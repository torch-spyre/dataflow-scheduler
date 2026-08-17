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
///   symbol.create_id %arg0 {symbol_id = -1 : si64} : index   // the input
///   %0 = arith.addi %arg0, %c24576 : index
///   symbol.create_id %0 {symbol_id = -3 : si64} : index      // -3 = -1 +
///   24576 %sym = symbol.create_symbol {SymbolId = -3 : i64} : index %v =
///   dataflow.get_logical_memory_view %unit, %sym {layout_map = ...}
/// ```
///
/// Nothing reads `%0`. It is there so that whatever resolves the symbols can
/// read what each one is from the IR rather than being told separately -- the
/// declarations *are* the table.
///
/// An address that varies with the grid element -- each one working on its own
/// slab of a tensor -- is one symbol per grid element rather than one overall.
/// Those are gathered into a uniform map keyed on the unit, so the start
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
/// over a block argument: the kernel takes the argument in its own signature,
/// and splitting the kernel into per-schedule programs threads it down as an
/// argument of each program function that reads it.
///
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_SYMBOLICSTARTADDRESS_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_SYMBOLICSTARTADDRESS_H_

#include <map>
#include <optional>
#include <tuple>
#include <utility>

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Value.h"

namespace scheduler {

/// Hands out the symbol ids of one run and says which id each of its inputs is.
///
/// The run's inputs come first, one per argument of the functions nothing calls
/// -- the kernels -- and each is `-(position + 1)`. Everything derived from
/// them is numbered after, continuing downward in the order it is asked for.
///
/// Symbol ids are negative and are drawn from a single sequence: nothing sets a
/// range aside, so an id used for one thing cannot be used for another, and the
/// numbering has to be arrived at once for the whole run rather than per
/// function. That is why this is owned by the pass rather than by a walk.
///
/// Whatever resolves the symbols downstream has to arrive at the same numbering
/// for the inputs, and to leave everything this allocator handed out alone. The
/// declarations this writes into the IR are what say so: a consumer reads the
/// ids rather than working them out again.
class SymbolAllocator {
 public:
  /// Numbers the inputs of every kernel under \p top and declares each on the
  /// argument it arrives as, so that what an id means is said in the IR.
  /// Derived ids start after the last input's.
  ///
  /// A kernel is a function nothing under \p top calls. Its index-typed
  /// arguments are the run's inputs; an argument of any other type is not
  /// something a symbol can stand for and is passed over.
  explicit SymbolAllocator(mlir::ModuleOp top);

  /// Declares on \p func's own arguments the input each one arrives as.
  ///
  /// Splitting a kernel into programs hands each of them the subset of the
  /// inputs it reads, and a program has to say what its own arguments are for:
  /// what corrects it is written against what it declares, not against the
  /// kernel's signature. A kernel needs no call of this -- its arguments were
  /// declared when this was built.
  ///
  /// Call it once the program units exist. Before that a declaration counts as
  /// work to be done on a unit and is copied into every one of them.
  void declareInputsIn(mlir::func::FuncOp func, mlir::OpBuilder& builder) const;

  /// The symbol id standing for \p address if it is a run input, std::nullopt
  /// if it is not -- a constant, or anything computed.
  ///
  /// An input is a block argument of a function's entry block, traced back
  /// through the calls that pass it along until the kernel is reached. What
  /// identifies it is its position in *that* signature, since a program
  /// function takes only the subset of inputs it reads and two programs' first
  /// arguments are in general two different tensors.
  std::optional<int64_t> inputSymbolIdFor(mlir::Value address) const;

  /// The id for a symbol derived from \p root, displaced by \p displacement,
  /// with
  /// \p terms for the per-grid leaves of its expression -- and whether it is
  /// new.
  ///
  /// The same address asked for twice gets the same id. One
  /// `ktdp.construct_memory_view` is copied into every unit the program runs
  /// work on, so without this one tensor's address would become as many symbols
  /// as there are units, all meaning the same thing and all to be resolved
  /// separately.
  std::pair<int64_t, bool> derivedIdFor(mlir::Value root, int64_t displacement,
                                        llvm::ArrayRef<int64_t> terms);

  /// How many inputs the run takes, i.e. how many ids are the inputs' own.
  int64_t numInputs() const { return -next_input_ - 1; }

 private:
  /// One past the last input id handed out, so `-(count + 1)`.
  int64_t next_input_ = -1;
  int64_t next_derived_ = -1;
  /// What each address already asked for was given, keyed by what makes it that
  /// address: where it is rooted, what displaces it, and its per-grid terms.
  std::map<std::tuple<const void*, int64_t, llvm::SmallVector<int64_t>>,
           int64_t>
      derived_;
};

/// A start address that is an expression over a run input, taken apart: which
/// input it is rooted at, and what each grid element's own address is.
struct SymbolicAddress {
  /// The input the address is computed from, as a symbol id.
  int64_t input = 0;
  /// The value the expression is rooted at -- the argument the input arrives
  /// as, which every per-element expression is written over.
  mlir::Value root;
  /// The units this address is read on, and for each the constant every
  /// per-grid term of the expression takes there. Empty where the address is
  /// the same everywhere, which is the common case.
  llvm::SmallVector<std::pair<mlir::Value, llvm::SmallVector<int64_t>>>
      perUnitTerms;
  /// The ops of the expression, roots-last, each to be rebuilt per unit. Empty
  /// where the address is the input itself and there is nothing to derive.
  llvm::SmallVector<mlir::Operation*> expression;
  /// The per-grid leaves of the expression, in the order `perUnitTerms` lists
  /// their values.
  llvm::SmallVector<mlir::Value> perUnitLeaves;
};

/// Takes \p address apart as an expression over a run input, or returns
/// std::nullopt if it is not one -- a constant, or rooted at something that is
/// not an input.
///
/// Fails, rather than returning std::nullopt, when the address *is* rooted at
/// an input but is not something a symbol can be written for: more than one
/// input in it, an operator whose definition could not be read back, or a
/// per-grid term that is not a query over constants. Reporting that here is
/// deliberate -- the alternative is a program that resolves to the wrong
/// address at runtime.
mlir::FailureOr<std::optional<SymbolicAddress>> takeSymbolicAddressApart(
    mlir::Value address, mlir::dataflow::ProgramUnitOp pu,
    const SymbolAllocator& symbols);

/// Emits the start address \p taken describes, displaced by \p displacement, at
/// \p builder's insertion point, and returns what the view should read.
///
/// One `symbol.create_symbol` where every grid element agrees, or one per
/// element gathered into a uniform map keyed on the unit. The declaration
/// saying what each of those symbols is computed from goes at \p
/// definitions_at, which has to be somewhere \p taken's root is in scope and
/// outside the region a single grid element's program is read from.
mlir::FailureOr<mlir::Value> emitSymbolicStartAddress(
    const SymbolicAddress& taken, int64_t displacement,
    mlir::dataflow::ProgramUnitOp pu, SymbolAllocator& symbols,
    mlir::OpBuilder& builder, mlir::OpBuilder& definitions_at,
    mlir::Location loc);

/// Emits the `symbol.create_symbol` standing for symbol \p symbol_id at
/// \p builder's insertion point.
mlir::Value createSymbolicStartAddress(mlir::OpBuilder& builder,
                                       mlir::Location loc, int64_t symbol_id);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_SYMBOLICSTARTADDRESS_H_
