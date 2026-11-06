//===------------------------------------------------------------*- c++ -*-===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
/// A tensor's start address can arrive as an input to the kernel rather than as
/// a constant: the host decides where the tensor lives, and the compiler is
/// told only that there is an address, not which one. In DataflowIR such an
/// address is a *symbol* -- `symbol.create_symbol` carrying the id that stands
/// for it -- and whatever runs the program writes the real value into every
/// instruction the symbol reached.
///
/// The symbol stands alone as the start address, so that what has to be written
/// over is a whole operand and code generation needs no form for an expression
/// over a placeholder. An address that is an expression over an input therefore
/// becomes a symbol of its own, and what it is computed from is said
/// separately, as arithmetic the IR carries alongside:
///
/// ```mlir
///   // -1 is the input; -3 is -1 + 24576.
///   symbol.create_id %arg0 {symbol_id = -1 : si64} : index
///   %0 = arith.addi %arg0, %c24576 : index
///   symbol.create_id %0 {symbol_id = -3 : si64} : index
///   %sym = symbol.create_symbol {SymbolId = -3 : i64} : index
///   %v = dataflow.get_logical_memory_view %unit, %sym {layout_map = ...}
/// ```
///
/// Nothing reads `%0`. It is there so that whatever resolves the symbols can
/// read what each one is from the IR rather than being told separately -- the
/// declarations *are* the table.
///
/// An address that varies with the grid element -- each one working on its own
/// slab of a tensor -- is one symbol per grid element rather than one overall.
/// Which slab that is can be said in either of two ways, and they come to the
/// same thing here: as arithmetic over the input at the view, so the address
/// the view is built on already differs; or as one view of the whole tensor
/// with an access tile whose offset into it is the grid element's, which is a
/// displacement of that view's address by the time it is lowered. Either way
/// the symbols are gathered into a uniform map keyed on the unit, so the start
/// address a program reads is the query, and each grid element's own symbol is
/// what gets written:
///
/// ```mlir
///   %m = uniform.def_immutable_mapping([%u0 -> %sym0], [%u1 -> %sym1]):index
///   %q = uniform.query_map(map:%m, key:%iter_arg) : index
///   %v = dataflow.get_logical_memory_view %unit, %q {layout_map = ...}
/// ```
///
/// In KTIR the address reaches `ktdp.construct_memory_view` as an expression
/// over block arguments: the kernel takes an argument in its own signature, or
/// launches the program from a loop of its own, and splitting the kernel into
/// per-schedule programs threads the argument or the loop variable down as an
/// argument of each program function that reads it. An address may be computed
/// from several of them -- a base the run hands in, advanced by the loop
/// variable -- and is then one symbol defined over all of them:
///
/// ```mlir
///   // -1 is the base, -2 the loop variable; -4 is (-1 + -2 * 128) / 64.
///   symbol.create_id %a {symbol_id = -1 : si64} : index
///   symbol.create_id %i {symbol_id = -2 : si64} : index
///   %0 = arith.muli %i, %c128 : index
///   symbol.create_id %0 {symbol_id = -3 : si64} : index
///   %1 = arith.addi %a, %0 : index
///   %2 = arith.divsi %1, %c64 : index
///   symbol.create_id %2 {symbol_id = -4 : si64} : index
/// ```
///
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_SYMBOLICSTARTADDRESS_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_SYMBOLICSTARTADDRESS_H_

#include <optional>
#include <utility>

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace scheduler {

/// What one operand of one step of an expression is.
///
/// Every operand says which of four things it is, rather than only whether it
/// is a constant, so that a key describes the tree of the expression and not
/// just its sequence of operators: with only "constant or not",
/// `a * 4 + b` and `a + b * 4` are the same list of steps, and two different
/// addresses would get one symbol.
struct OperandRef {
  enum class Kind : uint8_t {
    /// A compile-time integer; `value` is the integer.
    Constant,
    /// A value a symbol of the run already stands for -- in a program, one of
    /// its inputs; in a kernel, a root. `value` is its position in the roots of
    /// the expression, not its id, so that two copies of the same arithmetic
    /// over different values in two places are told apart by those values.
    Input,
    /// A value that differs per grid element but is known on each one; `value`
    /// is its position in the per-unit leaves, and what it is worth on a unit
    /// is in DerivedKey::terms.
    PerUnitLeaf,
    /// The result of another step of the same expression; `value` is that
    /// step's position, always lower than the step reading it.
    Step,
  };
  Kind kind = Kind::Constant;
  int64_t value = 0;

  bool operator==(const OperandRef& other) const {
    return kind == other.kind && value == other.value;
  }
  bool operator!=(const OperandRef& other) const { return !(*this == other); }
};

inline llvm::hash_code hash_value(const OperandRef& ref) {
  return llvm::hash_combine(static_cast<uint8_t>(ref.kind), ref.value);
}

/// One step of the expression a derived address is built by: which operation it
/// is, and what each of its operands is.
///
/// Not the operation itself. Two program units running one tensor's address are
/// two copies of the same arithmetic, so an `Operation*` would make them two
/// addresses and hand out a symbol each -- and the copies are erased once a
/// symbol stands for them, leaving the key pointing at freed memory.
/// `OperationName` is interned by the context, so it compares and hashes as a
/// pointer and outlives the IR the key was built from.
struct ExpressionStep {
  mlir::OperationName op;
  /// One entry per operand, in operand order. In order because the order is
  /// part of the address -- `root - 5` and `5 - root` are two different ones.
  llvm::SmallVector<OperandRef, 2> operands;

  bool operator==(const ExpressionStep& other) const {
    return op == other.op && operands == other.operands;
  }
};

/// So that a DerivedKey's expression can be hashed with the rest of it.
inline llvm::hash_code hash_value(const ExpressionStep& step) {
  return llvm::hash_combine(mlir::hash_value(step.op),
                            llvm::hash_combine_range(step.operands.begin(),
                                                     step.operands.end()));
}

/// What identifies a derived address: the values it is rooted at, the
/// expression over them, the offset added to it, the constants its per-unit
/// operands take, and the word size it is divided by. The word size is part of
/// it because one tensor read through units that address in different word
/// sizes is two different numbers.
struct DerivedKey {
  /// The values the expression's `OperandRef::Input`s refer to, by position:
  /// program arguments, or in a kernel its arguments and loop variables. Values
  /// rather than ids, so that an address is only ever shared within the
  /// function that reads it, which is where its definition has to be declared.
  llvm::SmallVector<mlir::Value, 2> roots;
  int64_t offset = 0;
  llvm::SmallVector<int64_t> terms;
  int64_t word_size = 1;
  /// The steps the address is built from its roots by, operands before the
  /// steps reading them. What tells two addresses apart that agree in roots,
  /// offset, terms and word size.
  ///
  /// `(1024 + (tile * 6 + row) * 4096) * 2`, for a row the run hands in taken
  /// to a grid element's slab of a tensor the compiler placed at 1024, is:
  ///
  /// ```
  ///   0: arith.muli  [leaf 0, 6]      the grid element's share, in rows
  ///   1: arith.addi  [step 0, input 0] displaced by the row the run hands in
  ///   2: arith.muli  [step 1, 4096]   rows to elements
  ///   3: arith.addi  [1024, step 2]   the address the compiler picked
  ///   4: arith.muli  [step 3, 2]      elements to bytes
  /// ```
  ///
  /// A second tensor read the same way differs in that 1024 and nothing else.
  llvm::SmallVector<ExpressionStep> expression;

  bool operator==(const DerivedKey& other) const {
    return roots == other.roots && offset == other.offset &&
           terms == other.terms && word_size == other.word_size &&
           expression == other.expression;
  }
};

/// Hands out the symbol ids of one run and says which id each of its inputs is.
///
/// A symbol chain starts at a *root*: an index-typed argument of the kernel --
/// the one function with a body in the orchestration module -- or the
/// induction variable of any loop in it. The kernel's arguments come first, in
/// signature order, then its loops' induction variables in pre-order.
/// Everything derived from them is numbered after, continuing downward in the
/// order it is asked for.
///
/// A loop variable is a root because the kernel's loops are where the programs
/// are launched from: a program reading one reads a number only the run knows,
/// as much as one reading an address. It is never folded or enumerated, even
/// where the trip count is known, so what this writes holds whether a later
/// stage keeps the loop or unrolls it.
///
/// Symbol ids are negative and are drawn from a single sequence: nothing sets a
/// range aside, so an id used for one thing cannot be used for another, and the
/// numbering has to be arrived at once for the whole run rather than per
/// function. That is why this is owned by the pass rather than by a walk.
///
/// Whatever resolves the symbols downstream has to arrive at the same numbering
/// for the roots, and to leave everything this allocator handed out alone. The
/// declarations this writes into the IR are what say so: a consumer reads the
/// ids rather than working them out again.
class SymbolAllocator {
 public:
  /// Numbers the roots of the kernel under \p top and declares on the value it
  /// is each one a call reads, so that what an id means is said in the IR: a
  /// kernel argument at the top of the kernel, a loop variable at the top of
  /// the loop's body. Derived ids start after the last root's.
  ///
  /// Before anything is numbered, every `symbol.create_id` and
  /// `symbol.create_symbol` anywhere under \p top is read, and numbering starts
  /// below the lowest id they use, so that no id an earlier run handed out is
  /// handed out again. A root already declared -- by an earlier run -- keeps
  /// the id it was declared as and is not declared a second time.
  ///
  /// Beyond that scan, only the kernel is walked, once; the programs are not.
  /// Where \p top has no
  /// orchestration module there is no kernel and so no roots: the allocator
  /// is empty, and no value of the run is one a symbol stands for.
  ///
  /// Every root is numbered, but declared only if a call is passed it or
  /// something the kernel computes from it: one no program reads keeps its id
  /// reserved and says nothing.
  ///
  /// A value the kernel computes and passes to a call is declared too, in place,
  /// as a symbol derived from the roots it is computed from, so that the
  /// program it is passed to can say what its argument is. Fails, having
  /// reported it at the operation responsible, when such a value is not
  /// something a symbol's definition can be written for -- the program would
  /// otherwise read an address nothing ever writes.
  ///
  /// An argument of any other type than index is not something a symbol can
  /// stand for and is passed over, though it still counts a position.
  static mlir::FailureOr<SymbolAllocator> build(mlir::ModuleOp top);

  /// Declares on \p func's own arguments the symbol each one arrives as.
  ///
  /// Splitting a kernel into programs hands each of them the subset of the
  /// values it reads, and a program has to say what its own arguments are for:
  /// what corrects it is written against what it declares, not against the
  /// kernel's signature. A kernel needs no call of this -- its roots were
  /// declared when this was built.
  ///
  /// Call it once the program units exist. Before that a declaration counts as
  /// work to be done on a unit and is copied into every one of them.
  void declareInputsIn(mlir::func::FuncOp func, mlir::OpBuilder& builder) const;

  /// Returns the symbol id standing for \p value if a symbol of the run stands
  /// for it, std::nullopt if none does.
  ///
  /// That is a root itself, or an argument of a function's entry block traced
  /// back through the calls that pass it along to what the kernel passed: a
  /// root, or a value the kernel computed from roots and that was declared when
  /// this was built. What the kernel passed is what identifies it, not the
  /// argument's position: a program function takes only the values it reads,
  /// so two programs' first arguments are in general two different tensors.
  std::optional<int64_t> inputSymbolIdFor(mlir::Value value) const;

  /// Returns the id for the symbol \p key describes, and whether that id is
  /// newly handed out.
  ///
  /// The same address asked for twice gets the same id. One
  /// `ktdp.construct_memory_view` is copied into every unit the program runs
  /// on, so without this one tensor's address would become as many symbols as
  /// there are units, all meaning the same thing and each resolved separately.
  std::pair<int64_t, bool> derivedIdFor(const DerivedKey& key);

 private:
  SymbolAllocator() = default;

  /// The next id of a root, and of a derived symbol, handed out and counted
  /// down.
  int64_t takeRootId();
  int64_t takeDerivedId();
  void numberKernelArguments(mlir::func::FuncOp kernel);
  /// Numbers the kernel's loop variables and, in the same walk, fills `calls_`
  /// and returns the kernel's calls in walk order.
  llvm::SmallVector<mlir::func::CallOp> numberLoopInductionVarsAndCollectCalls(
      mlir::func::FuncOp kernel);
  /// Which roots have been declared so far, and where the next declaration in
  /// each block goes. Lives only as long as one declareComputedCallOperands.
  struct RootDeclarations;

  /// Declares what each index operand of \p calls is: the roots it reads,
  /// and, where the kernel computes it, the derived symbol it is. Inserts
  /// through \p builder, restoring its insertion point.
  mlir::LogicalResult declareComputedCallOperands(
      llvm::ArrayRef<mlir::func::CallOp> calls, mlir::OpBuilder& builder);
  /// Declares what \p operand of \p call is, as declareComputedCallOperands
  /// does for each.
  mlir::LogicalResult declareCallOperand(mlir::func::CallOp call,
                                         mlir::Value operand,
                                         mlir::OpBuilder& builder,
                                         RootDeclarations& declared);
  /// Declares \p root at the top of the block it is an argument of, after any
  /// root declared there before it, unless it is declared already.
  void declareRoot(mlir::Value root, mlir::OpBuilder& builder,
                   RootDeclarations& declared);

  /// The one call to \p func, or null if there is no call or more than one.
  mlir::func::CallOp soleCallTo(mlir::func::FuncOp func) const;

  /// One past the last root id handed out.
  int64_t next_input_ = -1;
  int64_t next_derived_ = -1;
  /// Every root -- a kernel argument or a kernel loop variable, nothing else --
  /// and its id.
  llvm::DenseMap<mlir::Value, int64_t> roots_;
  /// Every value the kernel computes from roots and passes to a call, and the
  /// id of the derived symbol declared on it. Not roots: each is defined over
  /// them.
  llvm::DenseMap<mlir::Value, int64_t> derived_operands_;
  /// Each callee's call in the kernel, by the callee's name, or null where it
  /// has more than one. Built once: the pass never adds, removes or rewrites a
  /// call, so this stays right for as long as the allocator lives.
  llvm::StringMap<mlir::func::CallOp> calls_;
  /// The id handed out for each derived address. Nothing iterates it, so the
  /// order does not matter.
  llvm::DenseMap<DerivedKey, int64_t> derived_;
};

/// A start address that is an expression over values symbols stand for, taken
/// apart: which values it is rooted at, and what each grid element's own
/// address is.
struct SymbolicAddress {
  /// The values the address is computed from that symbols stand for, each
  /// once, in the order a walk of the expression meets them -- operand 0 first.
  /// Every per-unit expression is rebuilt over these values. Never empty.
  llvm::SmallVector<mlir::Value, 2> roots;
  /// The symbol id of each of `roots`, in the same order.
  llvm::SmallVector<int64_t, 2> inputs;
  /// One entry per unit the address is read on: the unit, and the constant each
  /// of `per_unit_leaves` takes there, in the same order. Empty when the
  /// address does not vary with the grid element, which is the common case.
  llvm::SmallVector<std::pair<mlir::Value, llvm::SmallVector<int64_t>>>
      per_unit_terms;
  /// The operations of the expression, every operand before the operations
  /// reading it, so rebuilding them in order works even where one value is read
  /// twice. Empty when the address is a root itself.
  llvm::SmallVector<mlir::Operation*> expression;
  /// `expression` as a key describes it, one entry per operation.
  llvm::SmallVector<ExpressionStep> steps;
  /// The operands of the expression whose value differs per unit -- each a
  /// `uniform.query_map` over per-unit constants. `per_unit_terms` gives what
  /// each is worth on each unit.
  llvm::SmallVector<mlir::Value> per_unit_leaves;
};

/// Takes \p address apart as an expression over values symbols stand for, or
/// returns std::nullopt if it is not one -- a constant, or rooted at nothing a
/// symbol stands for.
///
/// Fails, rather than returning std::nullopt, when the address *is* rooted at
/// a symbol but is not something a symbol can be written for: an operator whose
/// definition could not be read back, a value only known while the program
/// runs, or a per-grid term that is not a query over constants. Reporting that here is
/// deliberate -- the alternative is a program that resolves to the wrong
/// address at runtime.
mlir::FailureOr<std::optional<SymbolicAddress>> takeSymbolicAddressApart(
    mlir::Value address, mlir::dataflow::ProgramUnitOp pu,
    const SymbolAllocator& symbols);

/// The offset of the tile a program reads into the view it is taken from. It is
/// part of the address the symbol stands for, so it has to be a number here.
///
/// It is one number when every grid element reads the same tile of the view,
/// and one number per unit when the view is the whole tensor and each element
/// reads its own slab of it.
struct Displacement {
  /// The offset every unit shares. When `per_unit` is empty this is the offset.
  int64_t common_offset = 0;
  /// The offset on each unit, in the order the program unit lists them. Empty
  /// when the offset does not vary with the grid element, which is the common
  /// case.
  llvm::SmallVector<std::pair<mlir::Value, int64_t>> per_unit;

  /// Whether the offset differs between units.
  bool varies() const { return !per_unit.empty(); }

  /// The offset on \p unit.
  int64_t at(mlir::Value unit) const;

  /// Multiplies by \p factor, covering both `common_offset` and `per_unit`, to
  /// convert the unit the offsets are counted in.
  void scaleBy(int64_t factor);
};

/// Evaluates \p offset -- the offset of the tile a program reads into the view
/// it is taken from -- on each unit \p pu runs on.
///
/// A constant evaluates to itself on every unit. Otherwise \p offset has to be
/// an expression over constants and `uniform.query_map`s of per-unit constants,
/// which is what an offset computed from `ktdp.get_compute_tile_id` has become
/// by this point.
///
/// Fails when \p offset is neither, i.e. when it is only known while the
/// program runs -- computed from a run input or a loop variable, say. A symbol
/// whose base is a run input and whose offset is itself symbolic is not
/// supported; the offset would have to be part of an expression over several
/// symbols rather than a number per grid element.
mlir::FailureOr<Displacement> takeDisplacementApart(
    mlir::OpFoldResult offset, mlir::dataflow::ProgramUnitOp pu);

/// Returns the symbol id standing for \p value and declares in the entry block
/// of \p program what that symbol is.
///
/// A register filled with a runtime scalar uses a symbol the way a start
/// address does: the DataflowIR carries the id, and whatever resolves the
/// symbols writes the value in. The arithmetic computing the value is rebuilt
/// in the entry block
/// with one symbol per step, since a definition is a single operator over
/// symbols and constants.
///
/// If \p value is one a symbol already stands for, that symbol is returned and
/// nothing is declared.
///
/// Fails if \p value is not an expression over values symbols stand for, or if
/// a step of it cannot be rebuilt.
mlir::FailureOr<int64_t> declareScalarSymbol(mlir::Value value,
                                             mlir::func::FuncOp program,
                                             SymbolAllocator& symbols);

/// Emits the start address \p taken describes, displaced by \p displacement and
/// divided by \p word_size, at \p builder's insertion point, and returns what
/// the view should read.
///
/// \p word_size is the size in bytes of the word the unit reading the view
/// addresses in. The run hands its input addresses over as byte addresses, so
/// the symbol a program reads has to be the byte address divided by it -- the
/// division goes into the symbol's definition, after everything else, because
/// the operand the run writes over has to be the symbol alone.
///
/// One `symbol.create_symbol` where every grid element agrees, or one per
/// element gathered into a uniform map keyed on the unit -- which is what a
/// grid element's own displacement asks for just as much as an address that
/// already differs per element. The declaration saying what each of those
/// symbols is computed from goes at \p definitions_at, which has to be
/// somewhere \p taken's roots are in scope and outside the region a single grid
/// element's program is read from.
mlir::FailureOr<mlir::Value> emitSymbolicStartAddress(
    const SymbolicAddress& taken, const Displacement& displacement,
    int64_t word_size, mlir::dataflow::ProgramUnitOp pu,
    SymbolAllocator& symbols, mlir::OpBuilder& builder,
    mlir::OpBuilder& definitions_at, mlir::Location loc);

/// Emits the `symbol.create_symbol` standing for symbol \p symbol_id at
/// \p builder's insertion point.
mlir::Value createSymbolicStartAddress(mlir::OpBuilder& builder,
                                       mlir::Location loc, int64_t symbol_id);

}  // namespace scheduler

namespace llvm {

/// So that SymbolAllocator can keep its derived ids in a DenseMap.
template <>
struct DenseMapInfo<scheduler::DerivedKey> {
  static scheduler::DerivedKey getEmptyKey() {
    return {{DenseMapInfo<mlir::Value>::getEmptyKey()}, 0, {}, 1, {}};
  }
  static scheduler::DerivedKey getTombstoneKey() {
    return {{DenseMapInfo<mlir::Value>::getTombstoneKey()}, 0, {}, 1, {}};
  }
  static unsigned getHashValue(const scheduler::DerivedKey& key) {
    return static_cast<unsigned>(
        hash_combine(hash_combine_range(key.roots), key.offset,
                     hash_combine_range(key.terms),
                     key.word_size, hash_combine_range(key.expression)));
  }
  static bool isEqual(const scheduler::DerivedKey& lhs,
                      const scheduler::DerivedKey& rhs) {
    return lhs == rhs;
  }
};

}  // namespace llvm

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_SYMBOLICSTARTADDRESS_H_
