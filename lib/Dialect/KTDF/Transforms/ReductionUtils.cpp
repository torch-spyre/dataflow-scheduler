//===-- ReductionUtils.cpp --------------------------------------*- c++ -*-===//
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
// Definitions for the reduction utilities declared in ReductionUtils.h.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Transforms/ReductionUtils.h"

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::ktdf {

// ---------------------------------------------------------------------------
// findInnerDimLoopDim
// ---------------------------------------------------------------------------
std::optional<unsigned> findInnerDimLoopDim(linalg::GenericOp generic_op) {
  auto input_type =
      dyn_cast<RankedTensorType>(generic_op.getInputs().front().getType());
  if (!input_type) return std::nullopt;

  int64_t last = input_type.getRank() - 1;

  AffineMap input_map = generic_op.getIndexingMapsArray().front();
  auto dim_expr = dyn_cast<AffineDimExpr>(input_map.getResult(last));
  if (!dim_expr) return std::nullopt;

  unsigned loop_dim = dim_expr.getPosition();
  auto iter_types = generic_op.getIteratorTypesArray();
  if (iter_types[loop_dim] == utils::IteratorType::reduction) return loop_dim;
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// Pipeline-shape queries
// ---------------------------------------------------------------------------

auto findLoadStage(ktdf::PipelineOp pipeline, ktdf::StageOp compute_stage)
    -> StageOp {
  for (Value tok : compute_stage.getDependsIn()) {
    for (auto sibling : pipeline.getStages()) {
      if (sibling == compute_stage) continue;
      for (Value out_tok : sibling.getDependsOut()) {
        if (out_tok == tok) return sibling;
      }
    }
  }
  return {};
}

auto findStoreStage(ktdf::PipelineOp pipeline, ktdf::StageOp compute_stage)
    -> StageOp {
  for (Value tok : compute_stage.getDependsOut()) {
    for (auto sibling : pipeline.getStages()) {
      if (sibling == compute_stage) continue;
      for (Value in_tok : sibling.getDependsIn()) {
        if (in_tok == tok) return sibling;
      }
    }
  }
  return {};
}

auto findLoadTransfer(ktdf::StageOp load_stage, linalg::GenericOp generic_op)
    -> std::pair<ktdf::DataTransferOp, ktdf::ReadFromFifoOp> {
  auto read_from_fifo =
      generic_op.getInputs().front().getDefiningOp<ktdf::ReadFromFifoOp>();
  if (!read_from_fifo) return {nullptr, nullptr};

  Value fifo_in = read_from_fifo.getFifoSlot();
  ktdf::DataTransferOp load_transfer;
  load_stage.getBody()->walk([&](ktdf::DataTransferOp dt) {
    if (dt.isSourceMemRef() && dt.isDestFifo() &&
        dt.getDestination() == fifo_in) {
      load_transfer = dt;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return {load_transfer, read_from_fifo};
}

auto findStoreTransfer(ktdf::StageOp compute_stage, ktdf::StageOp store_stage,
                       linalg::GenericOp generic_op)
    -> std::pair<ktdf::DataTransferOp, ktdf::WriteToFifoOp> {
  ktdf::WriteToFifoOp write_to_fifo;
  compute_stage.getBody()->walk([&](ktdf::WriteToFifoOp write) {
    if (write.getData() == generic_op.getResult(0)) {
      write_to_fifo = write;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!write_to_fifo) return {nullptr, nullptr};

  Value fifo_out = write_to_fifo.getFifoSlot();
  ktdf::DataTransferOp store_transfer;
  store_stage.getBody()->walk([&](ktdf::DataTransferOp dt) {
    if (dt.isSourceFifo() && dt.isDestMemRef() && dt.getSource() == fifo_out) {
      store_transfer = dt;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return {store_transfer, write_to_fifo};
}

}  // namespace mlir::ktdf
