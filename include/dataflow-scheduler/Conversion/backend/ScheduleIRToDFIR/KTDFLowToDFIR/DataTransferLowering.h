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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_DATATRANSFERLOWERING_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_DATATRANSFERLOWERING_H_

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/UnitTypeDiscovery.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "mlir/IR/PatternMatch.h"

namespace scheduler {

/// Name of the vector lanes attribute for compute operations.
///
/// Indicates the SIMD vector width of compute operations (e.g. linalg.generic),
/// derived from the compute unit's SIMD feature.
static constexpr llvm::StringLiteral kVectorLanesAttrName =
    "dataflow_scheduler.vector_lanes";

/// Name of the access granularity attribute for data transfers.
///
/// When lowering a data transfer, the lowering may decide to emit an
/// `agen.composite_load_and_store` operation, which is able to spread out a
/// transfer across the time domain. To be able to use this feature, the passes
/// before the lowering do not narrow transfers down to the throughput that the
/// actual compute achieves.
///
/// The access granularity attribute indicates, in number of elements as a
/// 64-bit int, the hardware step limitation that this data transfer has to
/// obey. This is set on pipeline construction / legalization, and gives the
/// maximum number of elements that may be transferred in a single time step.
static constexpr llvm::StringLiteral kAccessGranularityAttrName =
    "dataflow_scheduler.access_granularity";

/// Register LowerDataTransferPattern into the given pattern set.
void populateDataTransferLoweringPatterns(mlir::RewritePatternSet& patterns,
                                          const ResourceToUnits& components);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_DATATRANSFERLOWERING_H_
