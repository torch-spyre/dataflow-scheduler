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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_BUFFERPHASELOWERING_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_BUFFERPHASELOWERING_H_

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/UnitTypeDiscovery.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace scheduler {

/// Lower all ktdf.buffer_phase / ktdf.select_memref pairs in `func` into
/// iter-arg-based double buffering. Processes one pair at a time until none
/// remain.
mlir::LogicalResult lowerDoubleBuffering(
    mlir::func::FuncOp func, const ResourceToUnits& components,
    mlir::ktdf_arch::ResourceKinds& resource_kinds);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_BUFFERPHASELOWERING_H_
