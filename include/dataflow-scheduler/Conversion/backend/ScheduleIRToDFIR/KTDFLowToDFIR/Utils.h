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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/UnitTypeDiscovery.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

namespace scheduler {

/// Enum representing the type of data transfer operation
enum class DataTransferType {
  kLoadAndStore,    // Memory to memory - use CompositeLoadAndStore
  kLoadAndSend,     // Memory to FIFO - use vector_load and send
  kReceiveAndStore  // FIFO to memory - use receive and vector_store
};

/// Helper to get flattened vector type from tensor or vector type.
[[nodiscard]] auto getFlattenedVectorType(mlir::ShapedType type)
    -> mlir::VectorType;

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

/// Resolve a FIFO endpoint attribute to the corresponding unit query-map Value.
///
/// Given a raw `mlir::Attribute` taken from either FifoSlotType::getSrc() or
/// FifoSlotType::getDest(), this helper:
///   1. Casts it to StringAttr and upper-cases the value to a ResourceType.
///   2. Looks that type up in `components`.
///   3. Calls createQueryMapForComponent to build the uniform.query_map.
///
/// On any failure the error is emitted on `op_for_errors` and the function
/// returns failure.
llvm::FailureOr<mlir::Value> resolveUnitFromFifoAttr(
    mlir::Attribute fifo_attr, const ResourceToUnits& components,
    mlir::PatternRewriter& rewriter, mlir::dataflow::ProgramUnitOp program_unit,
    mlir::Location loc, mlir::Operation* op_for_errors);

/// Name of the attribute that says how many vectors a FIFO slot is moved as.
///
/// Set on a `ktdf.data_transfer` into or out of a FIFO slot by
/// `markWideConversionSlots`, before the work is split into program units, so
/// the unit at the memory end of the FIFO still knows after the split what the
/// compute end does. A transfer without it moves the slot as one vector.
static constexpr llvm::StringLiteral kVectorsPerSlotAttrName =
    "dataflow_scheduler.vectors_per_slot";

/// Gets whether \p generic converts between f16 and f32 and nothing else, and
/// so has a wide side twice its narrow one: a lane-wise linalg.generic on
/// tensors whose body converts its one input and yields the result.
[[nodiscard]] auto isFloatConversion(mlir::Operation* generic) -> bool;

/// Gets whether \p generic is a float conversion that widens, reading f16 and
/// writing f32; one that is not reads f32 and writes f16.
[[nodiscard]] auto isWideningFloatConversion(mlir::Operation* generic) -> bool;

/// Marks every transfer through a FIFO slot that is the wide side of a float
/// conversion in \p func with the number of vectors the slot is moved as.
///
/// The conversion takes one vector of the narrow element to two of the wide
/// one, so its wide slot is two vectors, which both ends of the FIFO have to
/// agree on. Fails, with an error, on a conversion whose wide side is not a
/// FIFO only it uses, or whose FIFO has another end that moves the slot as
/// one vector.
[[nodiscard]] auto markWideConversionSlots(mlir::func::FuncOp func)
    -> mlir::LogicalResult;

/// Gets the vector one send or receive moves for \p transfer, a
/// `ktdf.data_transfer` of \p vector_type into or out of a FIFO slot.
///
/// That is the whole of \p vector_type, unless the transfer is marked as moved
/// as several vectors.
[[nodiscard]] auto getFifoVectorType(mlir::Operation* transfer,
                                     mlir::VectorType vector_type)
    -> mlir::VectorType;

/// Determine the data transfer type based on source and destination types.
/// @param src_is_fifo True if source is a FIFO slot, false if memref
/// @param dst_is_fifo True if destination is a FIFO slot, false if memref
/// @return The data transfer type (LoadAndStore, LoadAndSend, or
/// ReceiveAndStore), or failure if both sides are FIFOs (unsupported).
llvm::FailureOr<scheduler::DataTransferType> getDataTransferType(
    bool src_is_fifo, bool dst_is_fifo);

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_CONVERSION_KTDFLOWTODFIR_UTILS_H_
