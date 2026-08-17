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
/// arguments, because an argument is a value only the caller has and there is no
/// caller on a device.
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
/// the symbols reads it -- and not in the DataflowIR, which is handed on to code
/// generation with nothing in it that only a caller could supply.
///
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Symbol/Symbol.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "wrap-program-dfir"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_WRAPPROGRAMDFIRPASS
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h.inc"
}  // namespace scheduler

namespace {

/// The suffix the inner function's name takes. The outer one keeps the program's
/// own name, because that is what the run calls; the call has to say which of the
/// two it means.
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

/// The ops of \p entry that exist only to say what a symbol is -- every
/// `symbol.create_id` and, transitively, whatever computes what it points at.
/// In the order they appear, so rebuilding them elsewhere in that order always
/// has its operands already rebuilt.
///
/// Fails where something else uses one of \p entry's arguments: an argument is a
/// value only the caller has, so the DataflowIR left behind could not be
/// compiled, and saying so here beats a diagnostic from deep inside code
/// generation.
mlir::FailureOr<llvm::SmallVector<mlir::Operation*>> symbolOpsOf(
    mlir::Block& entry) {
  // Backwards from each declaration, so an op is collected only if a declaration
  // is what it is for.
  llvm::SetVector<mlir::Operation*> wanted;
  llvm::SmallVector<mlir::Value> pending;
  for (mlir::Operation& op : entry) {
    if (!mlir::isa<mlir::symbol::CreateIdOp>(op)) continue;
    wanted.insert(&op);
    for (mlir::Value operand : op.getOperands()) pending.push_back(operand);
  }
  while (!pending.empty()) {
    mlir::Value value = pending.pop_back_val();
    mlir::Operation* op = value.getDefiningOp();
    if (!op || op->getBlock() != &entry) continue;
    if (!wanted.insert(op)) continue;
    for (mlir::Value operand : op->getOperands()) pending.push_back(operand);
  }

  // Every use of an argument has to be one of them, or the DataflowIR needs a
  // value it will not have.
  for (mlir::BlockArgument arg : entry.getArguments()) {
    for (mlir::Operation* user : arg.getUsers()) {
      if (wanted.contains(user)) continue;
      return user->emitError()
             << PASS_NAME
             << ": this reads an argument of the program, which the DataflowIR "
                "compiled from it cannot be given -- only what a symbol is may "
                "be said over one";
    }
  }

  llvm::SmallVector<mlir::Operation*> ordered;
  for (mlir::Operation& op : entry)
    if (wanted.contains(&op)) ordered.push_back(&op);
  return ordered;
}

struct WrapProgramDFIRPass
    : public impl::WrapProgramDFIRPassBase<WrapProgramDFIRPass> {
  void runOnOperation() override {
    mlir::ModuleOp top = getOperation();

    llvm::SmallVector<mlir::ModuleOp> programs;
    for (mlir::Operation& op : top.getBodyRegion().front()) {
      auto program_module = mlir::dyn_cast<mlir::ModuleOp>(op);
      if (!program_module) continue;
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

    // What the body says about symbols, which belongs beside what the program is
    // rather than in what it is compiled from.
    mlir::FailureOr<llvm::SmallVector<mlir::Operation*>> symbol_ops =
        symbolOpsOf(body_entry);
    if (mlir::failed(symbol_ops)) return mlir::failure();

    program.setSymName(body_name);

    llvm::SmallVector<mlir::Operation*> contents;
    for (mlir::Operation& op : program_module.getBodyRegion().front())
      contents.push_back(&op);

    mlir::OpBuilder builder(program_module.getContext());
    builder.setInsertionPointToEnd(program_module.getBody());
    auto dfir_module = mlir::ModuleOp::create(builder, program_module.getLoc());

    mlir::Block* inner = dfir_module.getBody();
    for (mlir::Operation* op : contents) op->moveBefore(inner, inner->end());

    // Neither of the two definitions this leaves has a caller inside its own
    // module -- each is called from a module out, which is a symbol table of its
    // own -- so both are public, and said to be rather than left to a default.
    // A private symbol nothing in its own module calls is what symbol-dce
    // deletes, and dropping either of these means dropping the program.
    //
    // The forward declaration the call resolves against is the exception: it is
    // in the same module as the call, and is private precisely so that it reads
    // as standing in for a definition elsewhere.
    program.setVisibility(mlir::SymbolTable::Visibility::Public);

    builder.setInsertionPoint(dfir_module);
    auto body_type = builder.getFunctionType(/*inputs=*/{}, /*results=*/{});
    auto declaration = mlir::func::FuncOp::create(
        builder, program_module.getLoc(), body_name, body_type);
    declaration.setPrivate();

    auto caller = mlir::func::FuncOp::create(builder, program_module.getLoc(),
                                             name, type);
    caller.setVisibility(mlir::SymbolTable::Visibility::Public);
    mlir::Block* entry = caller.addEntryBlock();
    builder.setInsertionPointToEnd(entry);

    // The symbol declarations rebuilt over the caller's own arguments, in the
    // order they were written, and before the call: from here down a program is
    // corrected against what is declared beside what stands for its binary, and
    // the call is what that becomes.
    mlir::IRMapping mapping;
    for (mlir::BlockArgument arg : body_entry.getArguments())
      mapping.map(arg, entry->getArgument(arg.getArgNumber()));
    for (mlir::Operation* op : *symbol_ops) builder.clone(*op, mapping);

    mlir::func::CallOp::create(builder, program_module.getLoc(), declaration,
                               mlir::ValueRange{});
    mlir::func::ReturnOp::create(builder, program_module.getLoc());

    // And the body left with no arguments, which is what makes it something code
    // generation can take: the originals are gone with the declarations that
    // were the only things reading them.
    for (mlir::Operation* op : llvm::reverse(*symbol_ops)) op->erase();
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
