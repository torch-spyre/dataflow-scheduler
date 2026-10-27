//===-- ReductionUtils.h ----------------------------------------*- c++ -*-===//
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
// Utilities for reduction-related transforms that work with a Load / Compute /
// Store three-stage pipeline pattern:
//
//   findLoadStage     — upstream sibling via token chain.
//   findStoreStage    — downstream sibling via token chain.
//   findLoadTransfer  — trace the memref→fifo_in path.
//   findStoreTransfer — trace the fifo_out→memref path.
//
// These helpers assume the Load / Compute / Store pipeline shape produced by
// StageCoarsening and are intentionally kept in the transform layer rather
// than the dialect utilities, because they encode pipeline-shape assumptions
// that do not hold in general.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_REDUCTIONUTILS_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_REDUCTIONUTILS_H_

#include <optional>
#include <utility>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::ktdf {

// ---------------------------------------------------------------------------
// Return the loop dimension index that corresponds to the innermost reduction:
// the rightmost input dimension (input_rank - 1), if it is a reduction
// iterator. Returns std::nullopt if the rightmost input dimension is not a
// reduction.
// ---------------------------------------------------------------------------
std::optional<unsigned> findInnerDimLoopDim(linalg::GenericOp generic_op);

// ---------------------------------------------------------------------------
// Pipeline-shape queries.
//
// These assume the Load / Compute / Store three-stage pipeline shape produced
// by StageCoarsening.
// ---------------------------------------------------------------------------

/// Find the sibling stage of `compute_stage` inside `pipeline` that is
/// upstream (the Load stage) — i.e. the sibling whose `depends_out` token
/// appears in `compute_stage`'s `depends_in`.
/// Returns a null StageOp if not found.
auto findLoadStage(ktdf::PipelineOp pipeline, ktdf::StageOp compute_stage)
    -> StageOp;

/// Find the sibling stage of `compute_stage` inside `pipeline` that is
/// downstream (the Store stage) — i.e. the sibling that consumes a token
/// from `compute_stage`'s `depends_out`.
/// Returns a null StageOp if not found.
auto findStoreStage(ktdf::PipelineOp pipeline, ktdf::StageOp compute_stage)
    -> StageOp;

/// Trace the input data path for a reduction linalg.generic:
///   generic.inputs[0] ← ReadFromFifoOp(fifo_in) ←
///   DataTransferOp(src=input_memref, dst=fifo_in) in load_stage
///
/// Returns {load_transfer, read_from_fifo}, or {nullptr, nullptr} on
/// failure.
auto findLoadTransfer(ktdf::StageOp load_stage, linalg::GenericOp generic_op)
    -> std::pair<ktdf::DataTransferOp, ktdf::ReadFromFifoOp>;

/// Trace the output data path for a reduction linalg.generic:
///   generic.result(0) → WriteToFifoOp(fifo_out) →
///   DataTransferOp(src=fifo_out, dst=partial_memref) in store_stage
///
/// Returns {store_transfer, write_to_fifo}, or {nullptr, nullptr} on
/// failure.
auto findStoreTransfer(ktdf::StageOp compute_stage, ktdf::StageOp store_stage,
                       linalg::GenericOp generic_op)
    -> std::pair<ktdf::DataTransferOp, ktdf::WriteToFifoOp>;

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_REDUCTIONUTILS_H_
