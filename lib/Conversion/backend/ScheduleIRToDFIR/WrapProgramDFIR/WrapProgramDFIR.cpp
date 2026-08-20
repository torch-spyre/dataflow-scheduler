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
//
/// WrapProgramDFIR pass: gives each program two levels -- what it is, and what
/// it is compiled from.
///
/// A program of the run is two different things to two different readers. To
/// whatever launches it, it is a name, a signature saying which of the run's
/// inputs it is about, and the symbols its instructions still wait for. To code
/// generation it is DataflowIR and nothing else: a schedule over units, with no
/// arguments, because an argument is a value only the caller has and there is
/// no caller on a device.
///
/// So the program becomes a function of its own name holding what it is, and an
/// inner module holding a function of no arguments holding what it is compiled
/// from:
///
/// ```mlir
///   module @prog {
///     func.func @prog(%base: index) {
///       symbol.create_id %base {symbol_id = -1 : si64} : index
///       %0 = arith.addi %base, %c24576 : index
///       symbol.create_id %0 {symbol_id = -3 : si64} : index
///       call @prog_body() : () -> ()
///       return
///     }
///     func.func private @prog_body()
///     module {
///       func.func @prog_body() { ...DataflowIR... }
///     }
///   }
/// ```
///
/// Which symbol each address is, and what each is computed from, therefore sits
/// in the outer function -- beside what the program is, where whatever resolves
/// the symbols reads it -- and not in the DataflowIR, which is handed on to
/// code generation with nothing in it that only a caller could supply.
///
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Symbol/Symbol.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "wrap-program-dfir"
#define DEBUG_TYPE PASS_NAME

namespace {
static llvm::cl::opt<bool> DisableThisPass(
    "wrap-program-dfir-disable",
    llvm::cl::desc("Leave each program as one level rather than splitting it "
                   "into what it is and what it is compiled from"),
    llvm::cl::init(false));
}  // namespace

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_WRAPPROGRAMDFIRPASS
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h.inc"
}  // namespace scheduler

namespace {

/// The suffix the inner function's name takes. The outer one keeps the
/// program's own name, because that is what the run calls; the call has to say
/// which of the two it means.
constexpr llvm::StringLiteral kBodySuffix = "_body";

/// Whether \p module holds DataflowIR of its own.
bool holdsDataflow(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::dataflow::ProgramUnitOp) {
    found = true;
    return mlir::WalkResult::interrupt();
  });
  return found;
}

/// Gets the ops of \p entry a symbol declaration is built from: every
/// `symbol.create_id`, and whatever computes the value each one points at. In
/// the order they appear in the block, so rebuilding them elsewhere in that
/// order always has its operands already rebuilt.
///
/// `erasable` gets the subset that nothing else reads, which is what may be
/// erased once the declarations have been rebuilt beside the program. The rest
/// is shared with the body -- a constant an address is divided by can be the
/// same one a transfer strides by -- and is copied rather than moved: erasing
/// it would take a value the DataflowIR still uses.
///
/// Fails if anything else uses one of \p entry's arguments. An argument is a
/// value only the caller has, so the DataflowIR left behind could not be
/// compiled, and failing here beats a diagnostic from inside code generation.
///
/// Only \p entry is looked at, and a declaration anywhere else in the function
/// is a failure rather than something followed. What writes them is
/// SymbolAllocator, which writes every one into the entry block, over an
/// argument or over arithmetic it puts there too -- so an expression that ran
/// through control flow or a nested region is not a shape this has to handle,
/// and it is refused rather than half-handled: left where it was, a declaration
/// would stay in the DataflowIR that code generation is handed, which is the
/// one thing this pass exists to prevent. Following definitions across regions
/// would take something like `PredecessorInfo`, and there is nothing yet to
/// follow.
mlir::FailureOr<llvm::SmallVector<mlir::Operation*>> symbolOpsOf(
    mlir::func::FuncOp program, mlir::Block& entry,
    llvm::DenseSet<mlir::Operation*>& erasable) {
  mlir::WalkResult elsewhere =
      program.walk([&](mlir::symbol::CreateIdOp declaration) {
        if (declaration->getBlock() == &entry)
          return mlir::WalkResult::advance();
        declaration.emitError()
            << PASS_NAME
            << ": a symbol is declared outside the program's entry block, so "
               "what it is could not be moved out of the DataflowIR";
        return mlir::WalkResult::interrupt();
      });
  if (elsewhere.wasInterrupted()) return mlir::failure();

  // Walked backwards from each declaration, so an op is collected only when a
  // declaration depends on it. A set rather than a list: the order comes from
  // the block below, and nothing iterates this.
  llvm::DenseSet<mlir::Operation*> wanted;
  llvm::SmallVector<mlir::Value> pending;
  for (mlir::Operation& op : entry) {
    if (!mlir::isa<mlir::symbol::CreateIdOp>(op)) continue;
    wanted.insert(&op);
    for (mlir::Value operand : op.getOperands()) pending.push_back(operand);
  }
  while (!pending.empty()) {
    mlir::Value value = pending.pop_back_val();
    mlir::Operation* op = value.getDefiningOp();
    if (!op) {
      // A block argument, and the only ones a declaration in this block may be
      // written over are the program's own -- those are rebuilt over the
      // caller's arguments. Anything else is a value that exists only inside
      // some region of the body, and cloning the declaration would have nothing
      // to give it.
      auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
      if (arg && arg.getOwner() == &entry) continue;
      return entry.getParentOp()->emitError()
             << PASS_NAME
             << ": a symbol is said over a value that exists only inside the "
                "program's body, so it cannot be declared beside the program";
    }
    if (op->getBlock() != &entry) continue;
    if (!wanted.insert(op).second) continue;
    for (mlir::Value operand : op->getOperands()) pending.push_back(operand);
  }

  // An argument may be read only by the ops collected above. Anything else
  // reading one would be left in the DataflowIR, which is compiled with no
  // arguments to give it.
  for (mlir::BlockArgument arg : entry.getArguments()) {
    for (mlir::Operation* user : arg.getUsers()) {
      if (wanted.contains(user)) continue;
      return user->emitError()
             << PASS_NAME
             << ": this reads an argument of the program. The DataflowIR the "
                "program is compiled from takes no arguments, so only a symbol "
                "declaration may read one";
    }
  }

  llvm::SmallVector<mlir::Operation*> ordered;
  for (mlir::Operation& op : entry)
    if (wanted.contains(&op)) ordered.push_back(&op);

  // What only the declarations read may go with them. Anything with a reader
  // outside the set stays: it is the body's too.
  for (mlir::Operation* op : ordered) {
    const bool only_ours = llvm::all_of(
        op->getUsers(), [&](auto* user) { return wanted.contains(user); });
    if (only_ours) erasable.insert(op);
  }
  return ordered;
}

struct WrapProgramDFIRPass
    : public impl::WrapProgramDFIRPassBase<WrapProgramDFIRPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    mlir::ModuleOp top = getOperation();

    // Collected before any is wrapped, because wrapping a program module adds a
    // module inside it and this walk is over the modules of `top`.
    llvm::SmallVector<mlir::ModuleOp> programs;
    for (auto program_module : top.getOps<mlir::ModuleOp>()) {
      // Only the programs. The declaration module is the unnamed one and holds
      // the kernels rather than any DataflowIR.
      if (!program_module.getSymName().has_value()) continue;
      if (!holdsDataflow(program_module)) continue;
      // Already two levels, so nothing to do -- a pathway that arrives this way
      // is left alone.
      if (!program_module.getOps<mlir::ModuleOp>().empty()) continue;
      programs.push_back(program_module);
    }

    for (mlir::ModuleOp program_module : programs)
      if (mlir::failed(wrap(program_module))) return signalPassFailure();
  }

 private:
  static mlir::LogicalResult wrap(mlir::ModuleOp program_module) {
    const llvm::StringRef name = program_module.getSymName().value();
    const std::string body_name = name.str() + kBodySuffix.str();

    mlir::func::FuncOp program;
    for (mlir::func::FuncOp func : program_module.getOps<mlir::func::FuncOp>())
      if (func.getSymName() == name) program = func;
    if (!program) {
      return program_module.emitError()
             << PASS_NAME << ": program '" << name
             << "' holds no function of its own name to compile from";
    }
    const mlir::FunctionType type = program.getFunctionType();
    mlir::Block& body_entry = program.getBody().front();

    // What the body says about symbols, which belongs beside what the program
    // is rather than in what it is compiled from.
    llvm::DenseSet<mlir::Operation*> erasable;
    mlir::FailureOr<llvm::SmallVector<mlir::Operation*>> symbol_ops =
        symbolOpsOf(program, body_entry, erasable);
    if (mlir::failed(symbol_ops)) return mlir::failure();

    // Renamed through the symbol table, so the uses of the old name go with it.
    // A name already taken is reported rather than uniqued with a suffix: what
    // a program's body is called is a contract with whatever reads the compiled
    // program -- it looks the body up as the program's name plus `_body` -- so
    // a second `<name>_body` here is a collision to be told about, not one to
    // paper over with `<name>_body_0`.
    mlir::SymbolTable program_symbols(program_module);
    if (program_symbols.lookup(body_name)) {
      return program_module.emitError()
             << PASS_NAME << ": '" << name << "' already holds a '" << body_name
             << "', which is the name the DataflowIR it is compiled from has "
                "to "
                "take";
    }
    if (mlir::failed(program_symbols.rename(program, body_name))) {
      return program_module.emitError() << PASS_NAME << ": could not rename '"
                                        << name << "' to '" << body_name << "'";
    }

    // A block of its own for what the program module will hold, so that what it
    // holds now moves into the inner module whole rather than op by op. The
    // inner module goes in the new block; everything else is what moves.
    mlir::IRRewriter rewriter(program_module.getContext());
    mlir::Block& contents = program_module.getBodyRegion().front();
    mlir::Block* outer = rewriter.createBlock(
        &program_module.getBodyRegion(), program_module.getBodyRegion().end());

    rewriter.setInsertionPointToEnd(outer);
    auto dfir_module =
        mlir::ModuleOp::create(rewriter, program_module.getLoc());
    rewriter.inlineBlockBefore(&contents, dfir_module.getBody(),
                               dfir_module.getBody()->end());

    // Symbol-dce deletes private symbols which are not called, so visibility
    // should be set to public.
    program.setVisibility(mlir::SymbolTable::Visibility::Public);

    rewriter.setInsertionPoint(dfir_module);
    auto body_type = rewriter.getFunctionType(/*inputs=*/{}, /*results=*/{});
    auto declaration = mlir::func::FuncOp::create(
        rewriter, program_module.getLoc(), body_name, body_type);
    declaration.setPrivate();

    auto caller = mlir::func::FuncOp::create(rewriter, program_module.getLoc(),
                                             name, type);
    caller.setVisibility(mlir::SymbolTable::Visibility::Public);
    mlir::Block* entry = caller.addEntryBlock();
    rewriter.setInsertionPointToEnd(entry);

    // The symbol declarations rebuilt over the caller's own arguments, in the
    // order they were written, and before the call: from here down a program is
    // corrected against what is declared beside what stands for its binary, and
    // the call is what that becomes.
    mlir::IRMapping mapping;
    for (mlir::BlockArgument arg : body_entry.getArguments())
      mapping.map(arg, entry->getArgument(arg.getArgNumber()));
    for (mlir::Operation* op : *symbol_ops) rewriter.clone(*op, mapping);

    mlir::func::CallOp::create(rewriter, program_module.getLoc(), declaration,
                               mlir::ValueRange{});
    mlir::func::ReturnOp::create(rewriter, program_module.getLoc());

    // The body is left with no arguments, which is what code generation needs.
    // Nothing reads them any more: the declarations that did were just moved.
    // Reverse order, so an operand goes after the last thing reading it.
    for (mlir::Operation* op : llvm::reverse(*symbol_ops))
      if (erasable.contains(op)) op->erase();
    body_entry.eraseArguments(0, body_entry.getNumArguments());
    program.setFunctionType(body_type);

    LDBG(1) << "  " << name << " is now itself and " << body_name << ", "
            << symbol_ops->size() << " symbol op(s) moved out of the latter";
    return mlir::success();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createWrapProgramDFIRPass() {
  return std::make_unique<WrapProgramDFIRPass>();
}
