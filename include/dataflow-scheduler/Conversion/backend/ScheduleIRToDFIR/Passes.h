//===-- Passes.h ------------------------------------------------*- c++ -*-===//
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
// This file declares all the scheduler conversion passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_SCHEDULEIRTODFIR_PASSES_H_
#define DATAFLOW_SCHEDULER_CONVERSION_SCHEDULEIRTODFIR_PASSES_H_

#include <mlir/Pass/PassRegistry.h>

#include <memory>

namespace mlir {
class Pass;
class OpPassManager;
}  // namespace mlir

namespace scheduler {
struct SchedulerExtContext;
}  // namespace scheduler

namespace scheduler {

std::unique_ptr<mlir::Pass> createKTDFToKTDFLoweringPass();

std::unique_ptr<mlir::Pass> createKTDFToKTDFLoweringPass(
    const scheduler::SchedulerExtContext& scheduler_ctx);

std::unique_ptr<mlir::Pass> createKTDFLowToDFIRPass();

std::unique_ptr<mlir::Pass> createKTDFLowToDFIRPass(
    const scheduler::SchedulerExtContext& scheduler_ctx);

std::unique_ptr<mlir::Pass> createWrapProgramDFIRPass();

std::unique_ptr<mlir::Pass> createSplitDFIROutputPass();
std::unique_ptr<mlir::Pass> createSplitDFIROutputPass(
    llvm::StringRef output_dir);

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h.inc"

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_SCHEDULEIRTODFIR_PASSES_H_

// Made with Bob
