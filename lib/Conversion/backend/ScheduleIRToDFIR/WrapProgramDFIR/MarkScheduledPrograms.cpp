//===-- MarkScheduledPrograms.cpp -------------------------------*- c++ -*-===//
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
/// MarkScheduledPrograms pass: records, on each program module the run
/// scheduled, that it went through the whole pipeline.
///
/// The last pass of every run, so that the marker means the run reached its
/// end: a run that stops partway leaves nothing marked. A later run on the same
/// module reads the marker to tell the programs it has already scheduled from
/// the ones it has still to schedule.
///
/// The orchestration module is never marked.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Utils/ProgramModules.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "mark-scheduled-programs"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_MARKSCHEDULEDPROGRAMSPASS
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h.inc"
}  // namespace scheduler

namespace {

struct MarkScheduledProgramsPass
    : public impl::MarkScheduledProgramsPassBase<MarkScheduledProgramsPass> {
  void runOnOperation() override {
    LDBG(1) << "========= " PASS_NAME " =========";
    for (auto child : getOperation().getOps<mlir::ModuleOp>()) {
      if (isOrchestratorModule(child) || isScheduled(child)) continue;
      markScheduled(child);
      LDBG() << "  @" << child.getSymName().value_or("") << " marked";
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createMarkScheduledProgramsPass() {
  return std::make_unique<MarkScheduledProgramsPass>();
}
