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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_SCRATCHPADCONFLICTS_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_SCRATCHPADCONFLICTS_H_

#include <map>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/StageToUnitsMap.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/GlobalStageDAG.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/SmallVector.h"

namespace scheduler {

/// Step 3: Compute scratchpad conflicts between ordered stage pairs
mlir::LogicalResult computeScratchpadConflicts(
    const StageToUnitsMap& stage_to_units,
    const mlir::ktdf::StageDependencyDAG& dag,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds,
    std::map<std::pair<mlir::Operation*, mlir::Operation*>,
             llvm::SmallVector<scheduler::ResourceType, 2>>& conflicts);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFTOKTDFLOW_SCRATCHPADCONFLICTS_H_
