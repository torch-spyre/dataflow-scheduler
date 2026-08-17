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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_LOGICALMEMORYVIEWBUILDER_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_LOGICALMEMORYVIEWBUILDER_H_

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/SymbolicStartAddress.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace scheduler {

/// Lower ktdp.construct_memory_view chains and unrealized_conversion_cast
/// patterns (Source A and Source B) inside dataflow.program_unit bodies to
/// dataflow.get_logical_memory_view ops, resolving from_unit via
/// dataflow.get_unit + uniform map/query.
///
/// A start address computed from one of the run's inputs is lowered to the
/// symbol that stands for it, drawn from \p symbols -- see
/// SymbolicStartAddress.h.
///
/// Must be called after mlir::runRegionDCE has cleaned up dead ops.
mlir::LogicalResult buildLogicalMemoryViews(
    mlir::func::FuncOp func,
    const scheduler::arch_view::MemoryTree& memory_tree,
    const scheduler::SchedulerExtContext& ext_ctx, SymbolAllocator& symbols);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_LOGICALMEMORYVIEWBUILDER_H_
