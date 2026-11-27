//===-- ProgramModules.h ----------------------------------------*- c++ -*-===//
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
// This file declares what the scheduler decides about the child modules of the
// top-level module: which one is the orchestration module, which program
// modules a run has already scheduled, and which are new.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_UTILS_PROGRAMMODULES_H_
#define DATAFLOW_SCHEDULER_UTILS_PROGRAMMODULES_H_

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>

namespace scheduler {

/// The unit attribute that says a program module went through the whole
/// pipeline. Dialect-prefixed because builtin.module rejects a discardable
/// attribute without one.
inline constexpr llvm::StringLiteral kScheduledAttrName =
    "dataflow_scheduler.scheduled";

/// Whether \p module carries the marker identifying the module as scheduled.
bool isScheduled(mlir::ModuleOp module);

/// Marks \p module as scheduled. Also how a caller marks a program module it
/// added without the scheduler, so that a run leaves it alone. The last pass
/// of every run marks what it scheduled.
void markScheduled(mlir::ModuleOp module);

/// Whether \p child is the orchestration module: the one child of the top-level
/// module that carries no name.
bool isOrchestratorModule(mlir::ModuleOp child);

/// The child module of the top-level module that \p op is in, or is, or null
/// when it is in none. A scheduled program nests a module of its own inside its
/// program module, so the nearest enclosing module is not always the one.
mlir::ModuleOp getTopLevelChildModule(mlir::Operation* op);

/// Whether \p op is in a program module that carries the marker, so that a
/// pass leaves it alone. A pass asks this of the function, pipeline or op it is
/// about to work on.
bool isInScheduledProgram(mlir::Operation* op);

/// What a child module of the top-level module is to a run.
enum class ModuleKind {
  /// The orchestration module.
  Orchestration,
  /// A program module the run schedules.
  New,
  /// A program module an earlier run scheduled, or a caller marked.
  Scheduled,
};

/// Classifies \p module, a child module of the top-level module: the unnamed
/// module is Orchestration, a marked program module is Scheduled, and an
/// unmarked one is New.
ModuleKind classifyModule(mlir::ModuleOp module);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_UTILS_PROGRAMMODULES_H_
