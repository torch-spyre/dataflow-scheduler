//===-- Pipeline.h ----------------------------------------------*- c++ -*-===//
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
// This file declares the scheduler pipelie.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_PIPELINE_H_
#define DATAFLOW_SCHEDULER_PIPELINE_H_

namespace mlir {

class OpPassManager;

}  // namespace mlir

namespace scheduler {

struct SchedulerExtContext;

/// Builds the KTIR frontend stage: checks the incoming KTIR is legal and
/// converts it into an initial Schedule IR (KTDF) pipeline.
///
/// Passes come from `Conversion/frontend/KTIRToScheduleIR`.
void buildKTIRFrontendPipeline(
    mlir::OpPassManager& pm,
    const scheduler::SchedulerExtContext& scheduler_ctx);

/// Builds the scheduler optimization stage: legalizes and refines the Schedule
/// IR in place — routing data movement, tiling, coarsening, double buffering,
/// cross-instance parallelization, tile size selection, address assignment and
/// grid normalization.
///
/// Both the input and the output of this stage are Schedule IR (KTDF). Passes
/// come from `Transforms` and `Dialect/KTDF/Transforms`.
void buildSchedulerOptimizationPipeline(
    mlir::OpPassManager& pm,
    const scheduler::SchedulerExtContext& scheduler_ctx);

/// Builds the DFIR backend stage: lowers the optimized Schedule IR through
/// KTDFLow into Dataflow IR (DFIR).
///
/// Passes come from `Conversion/backend/ScheduleIRToDFIR`. Note that this does
/// not emit the DFIR output; callers that want the resulting DFIR written out
/// should append `createSplitDFIROutputPass()` themselves.
void buildDFIRBackendPipeline(
    mlir::OpPassManager& pm,
    const scheduler::SchedulerExtContext& scheduler_ctx);

/// Builds the full KTDP to DFIR pipeline with the given pass manager, by
/// chaining the KTIR frontend, scheduler optimization and DFIR backend stages.
///
/// Note: this does not emit the DFIR output. Callers that want the resulting
/// DFIR written out should append `createSplitDFIROutputPass()` themselves.
void buildKTDPToDFIRPipeline(
    mlir::OpPassManager& pm,
    const scheduler::SchedulerExtContext& scheduler_ctx);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_PIPELINE_H_
