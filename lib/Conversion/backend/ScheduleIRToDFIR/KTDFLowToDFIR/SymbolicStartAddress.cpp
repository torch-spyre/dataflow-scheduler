//===----------------------------------------------------------------------===//
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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/SymbolicStartAddress.h"

#include <limits>

#include "dataflow-scheduler/Dialect/Symbol/Symbol.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Utils/ProgramModules.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define DEBUG_TYPE "symbolic-start-address"

using namespace scheduler;

namespace {

/// \p value as an argument of its function's entry block, or a null
/// BlockArgument if it is not one. A block argument of some inner region -- a
/// loop's induction variable, a program unit's unit iter_arg -- is not an
/// input.
mlir::BlockArgument asFunctionArgument(mlir::Value value) {
  auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!arg) return {};
  auto func =
      mlir::dyn_cast_or_null<mlir::func::FuncOp>(arg.getOwner()->getParentOp());
  if (!func || func.getBody().empty() ||
      arg.getOwner() != &func.getBody().front())
    return {};
  return arg;
}

/// How many call edges an input is followed along before giving up. One is
/// enough for the pipeline as it stands -- kernel to program -- so this only
/// guards against a cycle in malformed IR.
constexpr unsigned kMaxCallDepth = 8;

/// Whether \p op is one whose definition can be read back out of the IR.
///
/// The set of operators that can be *written* is larger than the set that can
/// be read, and a symbol written outside the readable set is a hard failure in
/// whatever resolves it rather than a silent loss. So only these are used, and
/// anything else in an address expression is reported here.
bool readableAsDefinition(mlir::Operation* op) {
  return mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp,
                   mlir::arith::MulIOp, mlir::arith::DivSIOp,
                   mlir::arith::RemSIOp, mlir::arith::MinSIOp,
                   mlir::arith::CeilDivSIOp>(op);
}

/// The per-unit constants of a `uniform.query_map` over a mapping whose values
/// are all constants, keyed by the unit each is for. std::nullopt when \p value
/// is not such a query.
///
/// This is what a `ktdp.get_compute_tile_id` has become by the time addresses
/// are lowered, and multiplying or offsetting one is how an address comes to
/// vary with the grid element.
std::optional<llvm::DenseMap<mlir::Value, int64_t>> perUnitConstantsOf(
    mlir::Value value) {
  auto query = value.getDefiningOp<mlir::uniform::QueryMapOp>();
  if (!query) return std::nullopt;
  auto mapping =
      query.getMap().getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
  if (!mapping) return std::nullopt;

  llvm::DenseMap<mlir::Value, int64_t> per_unit;
  for (auto [key, mapped] : llvm::zip(mapping.getKeys(), mapping.getValues())) {
    std::optional<int64_t> constant = mlir::getConstantIntValue(mapped);
    if (!constant) return std::nullopt;
    per_unit[key] = *constant;
  }
  return per_unit;
}

/// The units \p pu runs on, in the order it lists them -- which is the order a
/// uniform map over them has to be keyed in.
llvm::SmallVector<mlir::Value> unitsOf(mlir::dataflow::ProgramUnitOp pu) {
  return llvm::SmallVector<mlir::Value>(pu.getUnits().begin(),
                                        pu.getUnits().end());
}

/// What \p value works out to on \p unit, or std::nullopt where it is not
/// something that has a value there at all.
///
/// A leaf is a constant, worth the same everywhere, or a per-grid query, worth
/// what the mapping says for this unit; anything else is folded from its
/// operands by the operator it is. The operators are those a symbol's
/// definition can be written with, so that a displacement worked out here is
/// one the same expression could have said.
std::optional<int64_t> evaluateAtUnit(mlir::Value value, mlir::Value unit) {
  if (std::optional<int64_t> constant = mlir::getConstantIntValue(value))
    return constant;

  if (std::optional<llvm::DenseMap<mlir::Value, int64_t>> per_unit =
          perUnitConstantsOf(value)) {
    const auto found = per_unit->find(unit);
    if (found == per_unit->end()) return std::nullopt;
    return found->second;
  }

  mlir::Operation* op = value.getDefiningOp();
  if (!op || !readableAsDefinition(op) || op->getNumOperands() != 2)
    return std::nullopt;

  std::optional<int64_t> lhs = evaluateAtUnit(op->getOperand(0), unit);
  std::optional<int64_t> rhs = evaluateAtUnit(op->getOperand(1), unit);
  if (!lhs || !rhs) return std::nullopt;

  // Checked, and an overflow is std::nullopt like anything else that cannot be
  // worked out: what the caller does with that is report the offset as one no
  // symbol can be written for, which is the right answer for a number that does
  // not fit either.
  int64_t result = 0;
  if (mlir::isa<mlir::arith::AddIOp>(op))
    return llvm::AddOverflow(*lhs, *rhs, result) ? std::nullopt
                                                 : std::optional(result);
  if (mlir::isa<mlir::arith::SubIOp>(op))
    return llvm::SubOverflow(*lhs, *rhs, result) ? std::nullopt
                                                 : std::optional(result);
  if (mlir::isa<mlir::arith::MulIOp>(op))
    return llvm::MulOverflow(*lhs, *rhs, result) ? std::nullopt
                                                 : std::optional(result);
  if (mlir::isa<mlir::arith::MinSIOp>(op)) return std::min(*lhs, *rhs);
  // A division by zero is undefined where the program runs and is not something
  // to work out an address from here either.
  if (*rhs == 0) return std::nullopt;
  if (mlir::isa<mlir::arith::DivSIOp>(op)) return *lhs / *rhs;
  if (mlir::isa<mlir::arith::RemSIOp>(op)) return *lhs % *rhs;
  // Rounding up is only worked out for a positive numerator and denominator --
  // an offset into a tensor and a size -- rather than reproducing what the
  // operation does with a negative one.
  if (mlir::isa<mlir::arith::CeilDivSIOp>(op) && *lhs >= 0 && *rhs > 0)
    return (*lhs + *rhs - 1) / *rhs;
  return std::nullopt;
}

/// Gets the value \p id is already declared on in \p block, null if it is not.
mlir::Value declaredIn(mlir::Block& block, int64_t id) {
  for (auto declaration : block.getOps<mlir::symbol::CreateIdOp>())
    if (declaration.getSymbolId() == id) return declaration.getInput();
  return nullptr;
}

/// Gets the id a `symbol.create_id` on \p value declares it as, std::nullopt if
/// there is none.
std::optional<int64_t> declaredIdOn(mlir::Value value) {
  for (mlir::Operation* user : value.getUsers())
    if (auto declaration = mlir::dyn_cast<mlir::symbol::CreateIdOp>(user))
      if (declaration.getInput() == value) return declaration.getSymbolId();
  return std::nullopt;
}

/// The first id under \p top that no `symbol.create_id` or
/// `symbol.create_symbol` anywhere in it uses: one below the lowest id in use,
/// or -1 where there is none.
///
/// Scheduled program modules are scanned too. They are never rewritten, but the
/// ids an earlier run handed out there are still taken, and a new one equal to
/// one of them would make two different values the same symbol.
int64_t firstUnusedSymbolId(mlir::ModuleOp top) {
  int64_t lowest = 0;
  top.walk([&](mlir::Operation* op) {
    if (auto declaration = mlir::dyn_cast<mlir::symbol::CreateIdOp>(op)) {
      lowest = std::min(lowest, declaration.getSymbolId());
    } else if (mlir::isa<mlir::symbol::CreateSymbolOp>(op)) {
      // The id is a discardable attribute and need not be there.
      if (auto id = op->getAttrOfType<mlir::IntegerAttr>("SymbolId"))
        lowest = std::min(lowest, id.getInt());
    }
  });
  if (lowest == std::numeric_limits<int64_t>::min())
    llvm::report_fatal_error("symbol ids exhausted by the input");
  return lowest - 1;
}

/// The kernel: the one function with a body in the orchestration module under
/// \p top, or null where there is no orchestration module.
///
/// Extraction puts the kernel in the orchestration module beside a declaration
/// of each program it calls, and only ever one kernel; the stages before this
/// one assume as much, so this does too. Fails, having reported it at the
/// second, where there is more than one: which one's arguments the run's
/// inputs are could only be guessed.
mlir::FailureOr<mlir::func::FuncOp> kernelOf(mlir::ModuleOp top) {
  for (auto child : top.getOps<mlir::ModuleOp>()) {
    if (!isOrchestratorModule(child)) continue;
    mlir::func::FuncOp kernel;
    for (auto func : child.getOps<mlir::func::FuncOp>()) {
      if (func.getBody().empty()) continue;
      if (kernel) {
        return func.emitError()
               << "more than one function with a body in the orchestration "
                  "module; @"
               << kernel.getName() << " is already its kernel";
      }
      kernel = func;
    }
    return kernel;
  }
  return mlir::func::FuncOp();
}

//===----------------------------------------------------------------------===//
// Collecting an expression
//===----------------------------------------------------------------------===//

/// An expression over values symbols stand for, as the operations computing it
/// and what each of their operands is.
///
/// What it is for is up to whoever collected it: an address, a scalar, or a
/// value a kernel passes to a program each has its own rules for which
/// operations it may hold, and checks them against this.
struct ExpressionTree {
  /// The values symbols already stand for, each once, in the order the walk
  /// met them; `root_ids` are their ids.
  llvm::SmallVector<mlir::Value, 2> roots;
  llvm::SmallVector<int64_t, 2> root_ids;
  /// The per-grid queries, and what each is worth on each unit.
  llvm::SmallVector<mlir::Value> per_unit_leaves;
  llvm::SmallVector<llvm::DenseMap<mlir::Value, int64_t>> per_unit_constants;
  /// The operations, every operand before the operations reading it. The last
  /// one computes the value collected.
  llvm::SmallVector<mlir::Operation*> steps;
  /// What every value the walk met is.
  llvm::DenseMap<mlir::Value, OperandRef> refs;

  OperandRef refOf(mlir::Value value) const {
    const auto found = refs.find(value);
    assert(found != refs.end() && "an operand the walk never met");
    return found->second;
  }

  /// Gets step \p index as a key describes it.
  ExpressionStep stepAt(size_t index) const {
    mlir::Operation* op = steps[index];
    ExpressionStep step{op->getName(), {}};
    for (mlir::Value operand : op->getOperands()) {
      const OperandRef ref = refOf(operand);
      assert((ref.kind != OperandRef::Kind::Step ||
              static_cast<size_t>(ref.value) < index) &&
             "steps out of order: an operand is computed after its reader");
      step.operands.push_back(ref);
    }
    return step;
  }
};

/// Collects the expression computing \p value, or returns std::nullopt if no
/// value a symbol stands for is among what it is computed from.
///
/// A leaf is a value a symbol stands for, a constant, or a per-grid query over
/// constants; anything else has to be an operation that could be cloned to
/// rebuild it -- one result, no regions, no memory effects. Which of those
/// operations may be in a definition is not judged here.
///
/// Fails at a leaf that is none of those: a block argument nothing stands for,
/// such as a loop variable of a program or a unit's iter_arg, or an operation
/// that could not be cloned. Reports it if \p report is set -- at the operation,
/// or for a block argument at what computes \p value -- and fails silently if
/// not, for a caller to whom a rejection is not an error.
///
/// Depth-first, operands before results and operand 0 first, so the steps come
/// out in an order they can be rebuilt in even where a value is read twice, and
/// the same arithmetic always comes out the same.
mlir::FailureOr<std::optional<ExpressionTree>> collectExpression(
    mlir::Value value, const SymbolAllocator& symbols, bool report) {
  ExpressionTree tree;
  // Each entry is an operation whose operands are being walked, and the index
  // of the next one to walk.
  llvm::SmallVector<std::pair<mlir::Operation*, unsigned>> stack;
  llvm::SmallPtrSet<mlir::Operation*, 8> on_stack;

  // Records \p current if it is a leaf or was met already, returning null;
  // otherwise returns the operation to walk the operands of.
  auto enter = [&](mlir::Value current) -> mlir::FailureOr<mlir::Operation*> {
    if (tree.refs.contains(current)) return nullptr;

    if (std::optional<int64_t> id = symbols.inputSymbolIdFor(current)) {
      tree.refs[current] = {OperandRef::Kind::Input,
                            static_cast<int64_t>(tree.roots.size())};
      tree.roots.push_back(current);
      tree.root_ids.push_back(*id);
      return nullptr;
    }
    if (std::optional<int64_t> constant = mlir::getConstantIntValue(current)) {
      tree.refs[current] = {OperandRef::Kind::Constant, *constant};
      return nullptr;
    }
    if (std::optional<llvm::DenseMap<mlir::Value, int64_t>> per_unit =
            perUnitConstantsOf(current)) {
      tree.refs[current] = {OperandRef::Kind::PerUnitLeaf,
                            static_cast<int64_t>(tree.per_unit_leaves.size())};
      tree.per_unit_leaves.push_back(current);
      tree.per_unit_constants.push_back(std::move(*per_unit));
      return nullptr;
    }

    mlir::Operation* op = current.getDefiningOp();
    // Only a cycle meets an operation again while its operands are still
    // being walked, and no expression can be rebuilt from one.
    if (!op || op->getNumResults() != 1 || op->getNumRegions() != 0 ||
        !mlir::isMemoryEffectFree(op) || !on_stack.insert(op).second) {
      if (!report) return mlir::failure();
      if (op) {
        return op->emitError()
               << "a value computed by '" << op->getName()
               << "' cannot be rebuilt as a symbol expression";
      }
      // A block argument has no operation of its own to point at.
      mlir::Operation* site = value.getDefiningOp();
      return (site ? site->emitError() : mlir::emitError(current.getLoc()))
             << "a value computed from a block argument no symbol stands for, "
                "such as a loop variable of a program, cannot be rebuilt as a "
                "symbol expression";
    }
    return op;
  };

  mlir::FailureOr<mlir::Operation*> top = enter(value);
  if (mlir::failed(top)) return mlir::failure();
  if (*top) stack.push_back({*top, 0});

  while (!stack.empty()) {
    auto& [op, next] = stack.back();
    if (next == op->getNumOperands()) {
      on_stack.erase(op);
      tree.refs[op->getResult(0)] = {OperandRef::Kind::Step,
                                     static_cast<int64_t>(tree.steps.size())};
      tree.steps.push_back(op);
      stack.pop_back();
      continue;
    }
    // `op` and `next` are not used past the push, which may reallocate.
    mlir::FailureOr<mlir::Operation*> child = enter(op->getOperand(next++));
    if (mlir::failed(child)) return mlir::failure();
    if (*child) stack.push_back({*child, 0});
  }

  if (tree.roots.empty()) return std::optional<ExpressionTree>();
  return std::optional<ExpressionTree>(std::move(tree));
}

/// Whether a value a symbol stands for is among those \p value is computed
/// from, following any operation's operands.
///
/// Separate from collectExpression, and looking through what that rejects,
/// because whether an operation that cannot be read back is an error depends on
/// this: over constants the address is just a constant, and over a symbol it is
/// one no definition can be written for.
bool reachesRoot(mlir::Value value, const SymbolAllocator& symbols) {
  llvm::DenseSet<mlir::Value> reached;
  llvm::SmallVector<mlir::Value> pending{value};
  while (!pending.empty()) {
    mlir::Value current = pending.pop_back_val();
    if (!reached.insert(current).second) continue;
    if (symbols.inputSymbolIdFor(current)) return true;
    if (mlir::Operation* op = current.getDefiningOp())
      for (mlir::Value operand : op->getOperands()) pending.push_back(operand);
  }
  return false;
}

/// The first step of \p tree a definition cannot be written with, null if every
/// one can.
mlir::Operation* firstUnreadableStep(const ExpressionTree& tree) {
  for (mlir::Operation* step : tree.steps)
    if (!readableAsDefinition(step)) return step;
  return nullptr;
}

}  // namespace

//===----------------------------------------------------------------------===//
// SymbolAllocator
//===----------------------------------------------------------------------===//

mlir::FailureOr<SymbolAllocator> SymbolAllocator::build(mlir::ModuleOp top) {
  SymbolAllocator symbols;

  // Before anything is given an id: whatever ids the input already uses,
  // whether an earlier run or a caller handed them out, are taken. On a first
  // run there are none, and numbering starts at -1.
  symbols.next_input_ = firstUnusedSymbolId(top);
  symbols.next_derived_ = symbols.next_input_;

  // The kernel is the one function with a body in the orchestration module;
  // the rest of it is the declarations of the programs it calls. Without an
  // orchestration module nothing is launched, so there is nothing a symbol
  // could stand for.
  mlir::FailureOr<mlir::func::FuncOp> found_kernel = kernelOf(top);
  if (mlir::failed(found_kernel)) return mlir::failure();
  mlir::func::FuncOp kernel = *found_kernel;
  if (!kernel) return symbols;

  // In this order, which is what the ids are: the kernel's arguments, then its
  // loop variables, then whatever is derived from them.
  symbols.numberKernelArguments(kernel);
  llvm::SmallVector<mlir::func::CallOp> calls =
      symbols.numberLoopInductionVarsAndCollectCalls(kernel);
  symbols.next_derived_ = symbols.next_input_;
  mlir::OpBuilder builder(top.getContext());
  if (mlir::failed(symbols.declareComputedCallOperands(calls, builder)))
    return mlir::failure();

  return symbols;
}

int64_t SymbolAllocator::takeRootId() {
  // Ids only ever descend, and a negative id is what marks one as a symbol, so
  // a wrap would hand out a positive number that means something else. It
  // takes 2^63 roots to get here, but an id that is silently not a symbol is
  // not a failure anything downstream would catch.
  if (next_input_ == std::numeric_limits<int64_t>::min())
    llvm::report_fatal_error("symbol ids exhausted numbering the run's inputs");
  return next_input_--;
}

void SymbolAllocator::numberKernelArguments(mlir::func::FuncOp kernel) {
  for (mlir::BlockArgument arg : kernel.getArguments()) {
    // An argument of any other type is not an address and no symbol stands for
    // it. Numbering still counts the position, so that what an id is follows
    // from the signature rather than from which arguments happened to be
    // addresses.
    //
    // An argument an earlier run declared keeps that id: the programs it
    // scheduled read it by that id.
    const std::optional<int64_t> existing = declaredIdOn(arg);
    const int64_t id = existing ? *existing : takeRootId();
    if (!mlir::isa<mlir::IndexType>(arg.getType())) continue;
    roots_[arg] = id;
    LDBG(1) << "  " << kernel.getName() << " argument " << arg.getArgNumber()
            << " is symbol " << id;
  }
}

llvm::SmallVector<mlir::func::CallOp>
SymbolAllocator::numberLoopInductionVarsAndCollectCalls(
    mlir::func::FuncOp kernel) {
  llvm::SmallVector<mlir::func::CallOp> calls;

  // Every loop of the kernel, in one walk, whether or not a call sits in it:
  // an id a loop variable never needs costs nothing, and finding out which
  // loops launch a program would mean walking each loop's body again.
  kernel.walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    // Every call, by its callee's name. Matched on the name rather than
    // through a symbol table: a call sits beside a private declaration of its
    // callee in the orchestration module, which is a different operation from
    // the function, and the function is in a program module of its own. Only
    // the kernel's calls: a program calls nothing, since extraction never puts
    // control flow, and so never a call, inside one.
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
      calls.push_back(call);
      auto [found, inserted] = calls_.try_emplace(call.getCallee(), call);
      if (!inserted) found->second = mlir::func::CallOp();
      return;
    }

    auto loop = mlir::dyn_cast<mlir::LoopLikeOpInterface>(op);
    if (!loop) return;
    std::optional<llvm::SmallVector<mlir::Value>> ivs =
        loop.getLoopInductionVars();
    if (!ivs) return;

    // Numbered whether or not a program reads it, so that which id a loop
    // variable is follows from the loops rather than from what the calls in
    // them happen to pass. Declared only once a call is found to read it.
    for (mlir::Value iv : *ivs) {
      // A block argument, because a root is declared at the top of the block
      // it is an argument of.
      if (!mlir::isa<mlir::BlockArgument>(iv) ||
          !mlir::isa<mlir::IndexType>(iv.getType()))
        continue;
      // As for a kernel argument, one already declared keeps its id.
      const std::optional<int64_t> existing = declaredIdOn(iv);
      const int64_t id = existing ? *existing : takeRootId();
      roots_[iv] = id;
      LDBG(1) << "  a loop variable of " << kernel.getName() << " is symbol "
              << id;
    }
  });
  return calls;
}

struct SymbolAllocator::RootDeclarations {
  llvm::DenseSet<mlir::Value> roots;
  /// The last declaration put at the top of each block.
  llvm::DenseMap<mlir::Block*, mlir::Operation*> last;
};

mlir::LogicalResult SymbolAllocator::declareComputedCallOperands(
    llvm::ArrayRef<mlir::func::CallOp> calls, mlir::OpBuilder& builder) {
  // A root is declared once a call is found to read it, directly or through
  // arithmetic, and not otherwise: one nothing reads keeps its id reserved and
  // says nothing.
  mlir::OpBuilder::InsertionGuard guard(builder);
  RootDeclarations declared;
  for (mlir::func::CallOp call : calls)
    for (mlir::Value operand : call.getOperands())
      if (mlir::failed(declareCallOperand(call, operand, builder, declared)))
        return mlir::failure();
  return mlir::success();
}

void SymbolAllocator::declareRoot(mlir::Value root, mlir::OpBuilder& builder,
                                  RootDeclarations& declared) {
  // At the top of the block it is an argument of -- the kernel's entry for a
  // kernel argument, the loop's body for a loop variable -- after any root
  // declared there before it, so that the declarations in a block read in the
  // order they are first used.
  if (!declared.roots.insert(root).second) return;
  // Declared already, by an earlier run: its id was taken from that
  // declaration, and a second would say the same thing again.
  if (declaredIdOn(root)) return;
  mlir::Block* owner = mlir::cast<mlir::BlockArgument>(root).getOwner();
  mlir::Operation*& last = declared.last[owner];
  if (last)
    builder.setInsertionPointAfter(last);
  else
    builder.setInsertionPointToStart(owner);
  last = mlir::symbol::CreateIdOp::create(builder, root.getLoc(), root,
                                          roots_.lookup(root))
             .getOperation();
}

mlir::LogicalResult SymbolAllocator::declareCallOperand(
    mlir::func::CallOp call, mlir::Value operand, mlir::OpBuilder& builder,
    RootDeclarations& declared) {
  // What a program can be told by a symbol: an index the kernel computed. A
  // constant, a root, or a value already declared needs nothing more.
  if (!mlir::isa<mlir::IndexType>(operand.getType())) return mlir::success();
  if (mlir::getConstantIntValue(operand)) return mlir::success();
  if (roots_.contains(operand)) {
    declareRoot(operand, builder, declared);
    return mlir::success();
  }
  if (derived_operands_.contains(operand)) return mlir::success();

  mlir::FailureOr<std::optional<ExpressionTree>> collected =
      collectExpression(operand, *this, /*report=*/true);
  if (mlir::failed(collected)) return mlir::failure();
  // Arithmetic over constants alone is a number the compiler knows, as much as
  // a constant is.
  if (!collected->has_value()) return mlir::success();
  const ExpressionTree& tree = **collected;
  // A per-grid query means nothing here: a kernel is not run per grid element,
  // its programs are.
  if (!tree.per_unit_leaves.empty()) {
    return tree.per_unit_leaves.front().getDefiningOp()->emitError()
           << "@" << call.getCallee()
           << " is passed a value computed from a per-grid query, which has no "
              "single value in a kernel";
  }
  if (mlir::Operation* unreadable = firstUnreadableStep(tree)) {
    return unreadable->emitError()
           << "@" << call.getCallee() << " is passed a value computed by '"
           << unreadable->getName()
           << "', which a symbol's definition cannot be written with";
  }

  // Declared in place, a symbol per step since a definition is one operator
  // over symbols and constants: the arithmetic is already in the kernel, next
  // to the roots it reads, so there is nothing to rebuild.
  //
  // One id per value rather than per key. Nothing is copied here, so a value is
  // what identifies a step, and a step another operand shares -- or the same
  // operand passed twice -- keeps the declaration it has.
  int64_t id = 0;
  for (mlir::Operation* step : tree.steps) {
    mlir::Value result = step->getResult(0);
    if (std::optional<int64_t> existing = declaredIdOn(result)) {
      id = *existing;
      continue;
    }
    id = takeDerivedId();
    builder.setInsertionPointAfter(step);
    mlir::symbol::CreateIdOp::create(builder, step->getLoc(), result, id);
  }
  derived_operands_[operand] = id;
  for (mlir::Value root : tree.roots) declareRoot(root, builder, declared);
  LDBG(1) << "  the kernel passes @" << call.getCallee() << " symbol " << id
          << ", over " << tree.roots.size() << " root(s)";
  return mlir::success();
}

int64_t SymbolAllocator::takeDerivedId() {
  if (next_derived_ == std::numeric_limits<int64_t>::min())
    llvm::report_fatal_error("symbol ids exhausted deriving addresses");
  return next_derived_--;
}

std::pair<int64_t, bool> SymbolAllocator::derivedIdFor(const DerivedKey& key) {
  const auto found = derived_.find(key);
  if (found != derived_.end()) return {found->second, false};
  const int64_t id = takeDerivedId();
  derived_.insert({key, id});
  return {id, true};
}

mlir::func::CallOp SymbolAllocator::soleCallTo(mlir::func::FuncOp func) const {
  return calls_.lookup(func.getName());
}

void SymbolAllocator::declareInputsIn(mlir::func::FuncOp func,
                                      mlir::OpBuilder& builder) const {
  for (mlir::BlockArgument arg : func.getArguments()) {
    std::optional<int64_t> input = inputSymbolIdFor(arg);
    // A constant the caller had: no symbol of the run stands for it.
    if (!input) continue;
    mlir::symbol::CreateIdOp::create(builder, arg.getLoc(), arg, *input);
  }
}

std::optional<int64_t> SymbolAllocator::inputSymbolIdFor(
    mlir::Value value) const {
  if (const auto root = roots_.find(value); root != roots_.end())
    return root->second;

  // Walk up the calls that pass this argument along, until what is passed is a
  // root or something the kernel declared as computed from roots.
  mlir::BlockArgument arg = asFunctionArgument(value);
  for (unsigned depth = 0; arg && depth <= kMaxCallDepth; ++depth) {
    auto func = mlir::cast<mlir::func::FuncOp>(arg.getOwner()->getParentOp());
    mlir::func::CallOp call = soleCallTo(func);
    if (!call) return std::nullopt;

    mlir::Value passed = call.getOperand(arg.getArgNumber());
    if (const auto root = roots_.find(passed); root != roots_.end())
      return root->second;
    if (const auto derived = derived_operands_.find(passed);
        derived != derived_operands_.end())
      return derived->second;
    // A constant, or anything else that is not an argument passed along.
    arg = asFunctionArgument(passed);
  }

  if (arg)
    LDBG(1) << "  gave up tracing an address through " << kMaxCallDepth
            << " calls";
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Taking an address apart
//===----------------------------------------------------------------------===//

mlir::FailureOr<std::optional<SymbolicAddress>>
scheduler::takeSymbolicAddressApart(mlir::Value address,
                                    mlir::dataflow::ProgramUnitOp pu,
                                    const SymbolAllocator& symbols) {
  // Whether a value a symbol stands for is among those this address is computed
  // from has to be settled before anything else in the expression can be
  // called an error. An operator whose definition cannot be read back is fine
  // over constants, where the address is just a constant, and fatal over a
  // symbol; only the whole expression says which of the two this is.
  if (!reachesRoot(address, symbols)) return std::optional<SymbolicAddress>();

  // Then the expression itself, with every leaf classified. A leaf is a value a
  // symbol stands for, a constant, or a per-grid query -- and nothing else,
  // because nothing else has a value this can say per grid element.
  mlir::FailureOr<std::optional<ExpressionTree>> collected =
      collectExpression(address, symbols, /*report=*/true);
  if (mlir::failed(collected)) return mlir::failure();
  if (!collected->has_value()) return std::optional<SymbolicAddress>();
  ExpressionTree& tree = **collected;

  if (mlir::Operation* unreadable = firstUnreadableStep(tree)) {
    return unreadable->emitError(
        "start address is computed from a run input by an operation whose "
        "definition cannot be written as a symbol expression");
  }

  SymbolicAddress taken;
  taken.roots = std::move(tree.roots);
  taken.inputs = std::move(tree.root_ids);
  taken.expression = tree.steps;
  for (size_t index = 0; index < tree.steps.size(); ++index)
    taken.steps.push_back(tree.stepAt(index));
  taken.per_unit_leaves = std::move(tree.per_unit_leaves);
  const auto& per_unit_constants = tree.per_unit_constants;

  // What each unit's copy of the per-grid leaves is. Every leaf has to say
  // something for every unit the program runs on, or the map cannot be keyed.
  if (!taken.per_unit_leaves.empty()) {
    for (mlir::Value unit : unitsOf(pu)) {
      llvm::SmallVector<int64_t> terms;
      for (const auto& per_unit : per_unit_constants) {
        const auto found = per_unit.find(unit);
        if (found == per_unit.end()) {
          return pu.emitError(
                     "a per-grid term of a symbolic start address says nothing "
                     "for one of the units the program runs on, so the address "
                     "cannot be resolved there")
                 << unit;
        }
        terms.push_back(found->second);
      }
      taken.per_unit_terms.emplace_back(unit, std::move(terms));
    }
  }

  LDBG(1) << "  start address is over " << taken.roots.size()
          << " symbol(s), first " << taken.inputs.front() << ", through "
          << taken.expression.size() << " op(s), "
          << taken.per_unit_terms.size() << " per-grid value(s)";
  return std::optional<SymbolicAddress>(std::move(taken));
}

//===----------------------------------------------------------------------===//
// Emitting it
//===----------------------------------------------------------------------===//

namespace {

/// Gets the key identifying the address \p taken, \p terms, \p displacement and
/// \p word_size describe, over the first \p through steps of the expression --
/// a step of it rather than the whole, where the step needs a symbol of its
/// own.
DerivedKey keyFor(const SymbolicAddress& taken, llvm::ArrayRef<int64_t> terms,
                  int64_t displacement, int64_t word_size, size_t through) {
  DerivedKey key;
  key.roots.assign(taken.roots.begin(), taken.roots.end());
  key.offset = displacement;
  key.terms.assign(terms.begin(), terms.end());
  key.word_size = word_size;
  key.expression.assign(taken.steps.begin(), taken.steps.begin() + through);
  return key;
}

/// Rebuilds \p taken's expression with each per-grid leaf replaced by the
/// constant \p terms gives it, adds \p displacement, declares the result as a
/// symbol of its own and returns that symbol's id.
///
/// Where what is left once it is folded is one of the roots and nothing is
/// displaced, the address is that root itself, so there is nothing to derive
/// and the root's own id is the answer.
int64_t declareDerived(const SymbolicAddress& taken,
                       llvm::ArrayRef<int64_t> terms, int64_t displacement,
                       int64_t word_size, SymbolAllocator& symbols,
                       mlir::OpBuilder& builder, mlir::Location loc) {
  // Whether the expression comes back to the input itself is only known once it
  // has been folded, and the id is only known to be new after that, so some of
  // what is built here can turn out not to be wanted. Those are erased rather
  // than left dead in the IR: what erases the IR later has no reason to take
  // them in an order that keeps a use alive, and a dead operand chain is how
  // one ends up destroyed while something still points at it.
  llvm::SmallVector<mlir::Operation*> built;
  auto note = [&built](mlir::Value value) {
    if (mlir::Operation* op = value.getDefiningOp()) built.push_back(op);
    return value;
  };
  auto discardBuilt = [&built] {
    for (mlir::Operation* op : llvm::reverse(built))
      if (op->use_empty()) op->erase();
  };

  mlir::IRMapping mapping;
  for (auto [leaf, term] : llvm::zip(taken.per_unit_leaves, terms))
    mapping.map(leaf,
                note(mlir::arith::ConstantIndexOp::create(builder, loc, term)));

  // Folded as it is built, because with only one leaf left unknown most of the
  // expression is constant here -- the grid element whose share starts at the
  // tensor's own base adds nothing, and its address is the input itself.
  // Leaving that as `input + 0` and letting canonicalization fold it later
  // would leave a symbol declared on a value nothing computes, which reads back
  // as a symbol with no definition at all.
  // An empty expression is an address that is a root itself, and so has one.
  assert((!taken.expression.empty() || taken.roots.size() == 1) &&
         "an address with no expression over more than one root");
  mlir::Value result = taken.roots.front();
  for (auto [index, op] : llvm::enumerate(taken.expression)) {
    mlir::Operation* clone = builder.clone(*op, mapping);
    llvm::SmallVector<mlir::Value> folded;
    if (mlir::succeeded(builder.tryFold(clone, folded)) && !folded.empty()) {
      mapping.map(op->getResult(0), folded.front());
      result = note(folded.front());
      continue;
    }
    result = note(clone->getResult(0));

    // A definition is one operator over symbols and constants, so a step read
    // by another step needs a symbol of its own to be read as an operand. The
    // last step needs none: what follows it reads it as part of a single
    // definition --
    // `(x + y) / word` -- or is the symbol below.
    const bool last = index + 1 == taken.expression.size();
    const bool read_by_what_follows =
        displacement == 0 &&
        (word_size == 1 ||
         mlir::isa<mlir::arith::AddIOp, mlir::arith::SubIOp>(clone));
    if (last && read_by_what_follows) continue;

    // On this rebuild of the step even where the id is not fresh: two addresses
    // sharing a step share its id, but each rebuilds it, and what the next step
    // reads is this rebuild.
    const int64_t step = symbols
                             .derivedIdFor(keyFor(taken, terms,
                                                  /*displacement=*/0,
                                                  /*word_size=*/1, index + 1))
                             .first;
    built.push_back(mlir::symbol::CreateIdOp::create(builder, loc, result, step)
                        .getOperation());
  }
  if (displacement != 0) {
    mlir::Value by =
        note(mlir::arith::ConstantIndexOp::create(builder, loc, displacement));
    result = note(mlir::arith::AddIOp::create(builder, loc, result, by));
  }

  // `result` is the byte address now: the roots, combined with this grid
  // element's share and whatever displaced it. Where that all folded away and
  // what is left is one of the roots, that root's own symbol stands for it.
  const auto* const same_as_root = llvm::find(taken.roots, result);

  if (word_size == 1) {
    if (same_as_root != taken.roots.end()) {
      discardBuilt();
      return taken.inputs[same_as_root - taken.roots.begin()];
    }
    const auto [id, fresh] = symbols.derivedIdFor(
        keyFor(taken, terms, displacement, word_size, taken.expression.size()));
    // Declared once. A second declaration of the same id would leave two
    // symbols meaning one address, so what was built to get here is thrown away
    // instead.
    if (!fresh) {
      discardBuilt();
      return id;
    }
    mlir::symbol::CreateIdOp::create(builder, loc, result, id);
    return id;
  }

  // The unit addresses in words of `word_size` bytes and the run hands its
  // inputs over as byte addresses, so the symbol a program reads is the byte
  // address divided by that -- last, after everything else the address is built
  // from.
  const auto [id, fresh] = symbols.derivedIdFor(
      keyFor(taken, terms, displacement, word_size, taken.expression.size()));
  if (!fresh) {
    discardBuilt();
    return id;
  }

  // The byte address stays unnamed, and the definition is the division over the
  // expression that built it -- `(input + share) / word_size` rather than a
  // symbol for the sum and a division over that. Whatever reads definitions has
  // an operator for that shape, and a symbol nothing waits for would be one
  // more thing for the run to resolve for no reason.
  mlir::Value bytes_per_word =
      mlir::arith::ConstantIndexOp::create(builder, loc, word_size);
  mlir::Value in_words =
      mlir::arith::DivSIOp::create(builder, loc, result, bytes_per_word);
  mlir::symbol::CreateIdOp::create(builder, loc, in_words, id);
  return id;
}

}  // namespace

int64_t scheduler::Displacement::at(mlir::Value unit) const {
  for (const auto& [key, offset] : per_unit)
    if (key == unit) return offset;
  return common_offset;
}

void scheduler::Displacement::scaleBy(int64_t factor) {
  common_offset *= factor;
  for (auto& [unit, offset] : per_unit) offset *= factor;
}

mlir::FailureOr<scheduler::Displacement> scheduler::takeDisplacementApart(
    mlir::OpFoldResult offset, mlir::dataflow::ProgramUnitOp pu) {
  Displacement displaced;

  // A constant, whether it arrived as one or was folded into one: the same
  // wherever the program runs, and the whole answer.
  if (auto attr = mlir::dyn_cast<mlir::Attribute>(offset)) {
    displaced.common_offset = llvm::cast<mlir::IntegerAttr>(attr).getInt();
    return displaced;
  }
  auto value = llvm::cast<mlir::Value>(offset);
  if (std::optional<int64_t> constant = mlir::getConstantIntValue(value)) {
    displaced.common_offset = *constant;
    return displaced;
  }

  // Otherwise the offset has to be worth something on every unit the program
  // runs on -- an expression over the grid element's own tile id and constants.
  llvm::SmallVector<std::pair<mlir::Value, int64_t>> per_unit;
  for (mlir::Value unit : unitsOf(pu)) {
    std::optional<int64_t> by = evaluateAtUnit(value, unit);
    if (!by) {
      mlir::Operation* site = value.getDefiningOp();
      return (site ? site->emitError() : pu.emitError())
             << "a start address from a run input, displaced by an offset "
                "that depends on a run input, a loop variable or anything else "
                "only known while the program runs, is not supported";
    }
    per_unit.emplace_back(unit, *by);
  }

  // Every unit came to the same number after all -- an access tile at a fixed
  // offset into the view, said the long way round. One number, and no map.
  const bool uniform = llvm::all_of(per_unit, [&](const auto& entry) {
    return entry.second == per_unit.front().second;
  });
  if (uniform) {
    displaced.common_offset = per_unit.front().second;
    return displaced;
  }

  displaced.per_unit = std::move(per_unit);
  LDBG(1) << "  displaced by a grid element's own offset, on "
          << displaced.per_unit.size() << " unit(s)";
  return displaced;
}

mlir::FailureOr<mlir::Value> scheduler::emitSymbolicStartAddress(
    const SymbolicAddress& taken, const Displacement& displacement,
    int64_t word_size, mlir::dataflow::ProgramUnitOp pu,
    SymbolAllocator& symbols, mlir::OpBuilder& builder,
    mlir::OpBuilder& definitions_at, mlir::Location loc) {
  // The same everywhere: one symbol, and no map to choose between copies of it.
  // Which is the common case -- a tensor read whole rather than a slab of it
  // per grid element -- and worth not building a map for.
  if (taken.per_unit_terms.empty() && !displacement.varies()) {
    const int64_t id = declareDerived(taken, {}, displacement.common_offset,
                                      word_size, symbols, definitions_at, loc);
    return createSymbolicStartAddress(builder, loc, id);
  }

  // The address differs per grid element, either because the expression it is
  // computed from does or because what displaces it does. Where only the
  // displacement differs the expression has no per-grid leaf and every unit
  // rebuilds the same one; the units are still what the map is keyed on.
  llvm::SmallVector<std::pair<mlir::Value, llvm::SmallVector<int64_t>>>
      per_unit_terms = taken.per_unit_terms;
  if (per_unit_terms.empty())
    for (mlir::Value unit : unitsOf(pu))
      per_unit_terms.emplace_back(unit, llvm::SmallVector<int64_t>{});

  // One per grid element, gathered into a map keyed on the unit and queried by
  // the iter_arg -- the same shape the units themselves are resolved through.
  llvm::SmallVector<mlir::Value> keys;
  llvm::SmallVector<mlir::Value> values;
  for (const auto& [unit, terms] : per_unit_terms) {
    const int64_t id = declareDerived(taken, terms, displacement.at(unit),
                                      word_size, symbols, definitions_at, loc);
    keys.push_back(unit);
    values.push_back(createSymbolicStartAddress(builder, loc, id));
  }

  auto mapping = mlir::uniform::DefImmutableMappingOp::create(
      builder, loc, builder.getIndexType(), keys, values);
  auto query = mlir::uniform::QueryMapOp::create(
      builder, loc, builder.getIndexType(), mapping.getResult(),
      pu.getRegion().front().getArgument(0));
  return query.getResult();
}

mlir::Value scheduler::createSymbolicStartAddress(mlir::OpBuilder& builder,
                                                  mlir::Location loc,
                                                  int64_t symbol_id) {
  auto symbol = mlir::symbol::CreateSymbolOp::create(builder, loc,
                                                     builder.getIndexType());
  // The id lives in an attribute rather than in the operation's own arguments;
  // `SymbolId` is the name symbol::CreateSymbolOp::getSymbolID reads.
  symbol->setAttr("SymbolId", builder.getI64IntegerAttr(symbol_id));
  return symbol.getResult();
}

//===----------------------------------------------------------------------===//
// Symbols for runtime scalars
//===----------------------------------------------------------------------===//

mlir::FailureOr<int64_t> scheduler::declareScalarSymbol(
    mlir::Value value, mlir::func::FuncOp program, SymbolAllocator& symbols) {
  // A per-grid query is refused: a register is filled once for the program,
  // not once per grid element.
  // Not reported: to the pattern asking, a value that is not a symbol is a
  // fill this lowering does not apply to, not an error.
  mlir::FailureOr<std::optional<ExpressionTree>> collected =
      collectExpression(value, symbols, /*report=*/false);
  if (mlir::failed(collected) || !collected->has_value() ||
      !(*collected)->per_unit_leaves.empty()) {
    LDBG(1) << "  " << value << " is not an expression over symbols";
    return mlir::failure();
  }
  const ExpressionTree& tree = **collected;

  // The arithmetic is rebuilt next to the program: the DataflowIR handed to
  // code generation must not read a value only a caller could supply.
  mlir::Block& entry = program.getBody().front();
  mlir::OpBuilder builder(program.getContext());
  builder.setInsertionPoint(entry.getTerminator());

  // What the program's own arguments are, where nothing has said so yet: a
  // program that builds no memory view has not had its inputs declared, and a
  // definition over an argument nothing declares could not be resolved. Only
  // the program's own arguments: a kernel's are declared by the allocator, once
  // a call is found to read them.
  for (auto [root, root_id] : llvm::zip(tree.roots, tree.root_ids)) {
    auto arg = mlir::dyn_cast<mlir::BlockArgument>(root);
    if (!arg || arg.getOwner() != &entry || declaredIdOn(arg)) continue;
    mlir::symbol::CreateIdOp::create(builder, arg.getLoc(), arg, root_id);
  }

  // The value is a root itself.
  if (tree.steps.empty()) return tree.root_ids.front();

  mlir::IRMapping mapping;
  DerivedKey key;
  key.roots.assign(tree.roots.begin(), tree.roots.end());
  int64_t id = 0;
  for (auto [index, step] : llvm::enumerate(tree.steps)) {
    // A constant is a leaf of the expression rather than a step, but it can be
    // defined anywhere in the program, so it is copied beside the step too.
    for (mlir::Value operand : step->getOperands()) {
      if (tree.refOf(operand).kind != OperandRef::Kind::Constant ||
          mapping.contains(operand))
        continue;
      if (mlir::Operation* constant = operand.getDefiningOp())
        builder.clone(*constant, mapping);
    }
    mlir::Operation* clone = builder.clone(*step, mapping);
    key.expression.push_back(tree.stepAt(index));
    if (clone->hasTrait<mlir::OpTrait::ConstantLike>()) continue;

    id = symbols.derivedIdFor(key).first;

    // One declaration per id, since two would say the symbol is two different
    // things.
    if (mlir::Value declared = declaredIn(entry, id)) {
      mapping.map(step->getResult(0), declared);
      clone->erase();
      continue;
    }
    mlir::symbol::CreateIdOp::create(builder, step->getLoc(),
                                     clone->getResult(0), id);
  }
  return id;
}
