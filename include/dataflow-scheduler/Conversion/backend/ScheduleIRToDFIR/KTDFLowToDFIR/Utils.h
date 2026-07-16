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

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_UTILS_H_
#define DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_UTILS_H_

#include <optional>

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace scheduler {

/// Enum representing the type of data transfer operation
enum class DataTransferType {
  kLoadAndStore,    // Memory to memory - use CompositeLoadAndStore
  kLoadAndSend,     // Memory to FIFO - use vector_load and send
  kReceiveAndStore  // FIFO to memory - use receive and vector_store
};

/// Walk up the parent chain of `op` to find the enclosing
/// dataflow::ProgramUnitOp and resolve its resource type from its first unit
/// operand.
/// Returns std::nullopt if no enclosing program_unit exists, it has no unit
/// operands, or the resource type cannot be resolved.
std::optional<scheduler::ResourceType> getEnclosingProgramUnitResourceType(
    mlir::Operation* op);

/// Helper to get flattened vector type from tensor or vector type.
/// Returns nullptr if the type is neither a RankedTensorType nor VectorType.
mlir::VectorType getFlattenedVectorType(
    mlir::Type type, arch_view::ResourceKinds& resource_kinds);

/// Match units by core ID between program_unit operands and target units,
/// create a def_immutable_mapping + query_map, and return the query result.
mlir::Value createQueryMapForComponent(
    mlir::OpBuilder& builder, mlir::dataflow::ProgramUnitOp program_unit,
    const llvm::SmallVector<mlir::Value, 4>& target_units, mlir::Location loc);

/// For the given program_unit, replace every in-region use of a single-result
/// ktdp.get_compute_tile_id with a uniform.query_map over a core map (keys =
/// the unit's operands, values = each operand get_unit's flat 'core' id),
/// keyed on the program_unit iterator argument. The map + query are built once
/// at the body start. The flat-core-id arith.constant values are shared across
/// units via `core_id_consts` (keyed by core id), materialized at
/// `const_builder`'s position (function scope, before the units) on first need.
/// Returns failure if an operand is not a dataflow.get_unit or lacks a 'core'
/// attribute.
mlir::LogicalResult replaceComputeTileIdWithCoreQuery(
    mlir::dataflow::ProgramUnitOp program_unit,
    llvm::DenseMap<int64_t, mlir::Value>& core_id_consts,
    mlir::OpBuilder& const_builder);

/// Determine the data transfer type based on source and destination types.
/// @param src_is_fifo True if source is a FIFO slot, false if memref
/// @param dst_is_fifo True if destination is a FIFO slot, false if memref
/// @return The data transfer type (LoadAndStore, LoadAndSend, or
/// ReceiveAndStore), or failure if both sides are FIFOs (unsupported).
llvm::FailureOr<scheduler::DataTransferType> getDataTransferType(
    bool src_is_fifo, bool dst_is_fifo);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_UTILS_H_
