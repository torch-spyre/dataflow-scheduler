//===----------------------------------------------------------------------===//
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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/DataTransferLowering.h"

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"

using namespace scheduler;

namespace {

/// Helper to build IntegerSet from static size array
/// Size of 1 corresponds to dk = 0, other sizes n correspond to 0 <= dk <= n-1
mlir::IntegerSet buildIntegerSetFromSizes(mlir::OpBuilder& builder,
                                          llvm::ArrayRef<int64_t> sizes) {
  if (sizes.empty()) {
    return mlir::IntegerSet::getEmptySet(0, 0, builder.getContext());
  }

  llvm::SmallVector<mlir::AffineExpr, 4> exprs;
  llvm::SmallVector<bool, 4> eq_flags;
  auto* context = builder.getContext();

  for (unsigned i = 0; i < sizes.size(); ++i) {
    auto dim = mlir::getAffineDimExpr(i, context);
    int64_t size = sizes[i];

    if (size == 1) {
      // Size 1: dk = 0
      exprs.push_back(dim);
      eq_flags.push_back(true);
    } else {
      // Size n: 0 <= dk <= n-1
      // This means: dk >= 0 and dk <= n-1
      exprs.push_back(dim);  // dk >= 0
      eq_flags.push_back(false);
      exprs.push_back(mlir::getAffineConstantExpr(size - 1, context) -
                      dim);  // n-1 - dk >= 0
      eq_flags.push_back(false);
    }
  }

  return mlir::IntegerSet::get(sizes.size(), 0, exprs, eq_flags);
}

/// Create a vectorchain.shuffle that broadcasts src_vec (vector<src_elements x
/// T>) to vector<dst_elements x T> using indices [0..src_elements-1] repeated
/// (dst_elements / src_elements) times.
mlir::Value insertSplatShuffle(mlir::PatternRewriter& rewriter,
                               mlir::Location loc, mlir::Value src_vec,
                               int64_t src_elements, int64_t dst_elements) {
  auto src_vec_type = mlir::cast<mlir::VectorType>(src_vec.getType());
  auto elem_type = src_vec_type.getElementType();

  // Build indices [0, 1, ..., src_elements-1]
  llvm::SmallVector<mlir::Attribute> index_attrs;
  for (int64_t i = 0; i < src_elements; ++i) {
    index_attrs.push_back(rewriter.getIntegerAttr(rewriter.getI32Type(), i));
  }
  auto indices_attr = rewriter.getArrayAttr(index_attrs);

  int32_t repetition = static_cast<int32_t>(dst_elements / src_elements);
  auto result_type = mlir::VectorType::get({dst_elements}, elem_type);

  return mlir::vectorchain::ShuffleOp::create(
             rewriter, loc, result_type, src_vec,
             /*mask=*/nullptr,
             /*dbgName=*/nullptr, indices_attr,
             rewriter.getI32IntegerAttr(repetition))
      .getOutput();
}

/// Pattern to lower ktdf.data_transfer operations
struct LowerDataTransferPattern
    : public mlir::OpRewritePattern<mlir::ktdf::DataTransferOp> {
  LowerDataTransferPattern(mlir::MLIRContext* context,
                           const ResourceToUnits& components)
      : OpRewritePattern(context), components_(components) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::DataTransferOp data_transfer_op,
      mlir::PatternRewriter& rewriter) const override {
    auto src = data_transfer_op.getSource();
    auto dst = data_transfer_op.getDestination();

    bool src_is_fifo = data_transfer_op.isSourceFifo();
    bool dst_is_fifo = data_transfer_op.isDestFifo();

    // Extract FIFO types if applicable
    // Determine transfer type based on source and destination
    auto transfer_type_or = getDataTransferType(src_is_fifo, dst_is_fifo);
    if (mlir::failed(transfer_type_or)) {
      data_transfer_op.emitError(
          "Unsupported data transfer: FIFO to FIFO transfers are not allowed");
      return mlir::failure();
    }
    auto transfer_type = *transfer_type_or;

    // Get source and destination indices
    auto src_indices = data_transfer_op.getSourceIndices();
    auto dst_indices = data_transfer_op.getDestIndices();

    // Get static sizes
    assert(data_transfer_op.hasAllStaticSourceSizes() &&
           "Expected static source sizes");
    assert(data_transfer_op.hasAllStaticDestSizes() &&
           "Expected static dest sizes");

    auto src_static_sizes = *data_transfer_op.getStaticSourceSizes();
    auto dst_static_sizes = *data_transfer_op.getStaticDestSizes();

    // Calculate total elements from static sizes
    int64_t src_total_elements = 1;
    for (int64_t size : src_static_sizes) {
      src_total_elements *= size;
    }

    int64_t dst_total_elements = 1;
    for (int64_t size : dst_static_sizes) {
      dst_total_elements *= size;
    }

    // For splat/pad transfers the source is smaller than the destination —
    // the hardware replicates or zero-pads to fill the vector. Skip the
    // equality check and use the destination size as the transfer width.
    auto transfer_mode_attr =
        data_transfer_op->getDiscardableAttr("transfer_mode");
    bool is_broadcast_transfer =
        transfer_mode_attr &&
        (llvm::cast<mlir::StringAttr>(transfer_mode_attr).getValue() ==
             "splat" ||
         llvm::cast<mlir::StringAttr>(transfer_mode_attr).getValue() == "pad");

    if (is_broadcast_transfer) {
      if (src_total_elements > dst_total_elements) {
        data_transfer_op.emitError(
            "source total elements must not exceed destination for "
            "splat/pad transfer");
        return mlir::failure();
      }
    } else if (src_total_elements != dst_total_elements) {
      data_transfer_op.emitError(
          "source and destination total elements must match");
      return mlir::failure();
    }

    int64_t total_elements = dst_total_elements;

    // Get element type (from memref or FIFO slot)
    mlir::Type elem_type;
    if (src_is_fifo) {
      elem_type =
          llvm::cast<mlir::ktdf::FifoSlotType>(src.getType()).getElementType();
    } else {
      elem_type = llvm::cast<mlir::MemRefType>(src.getType()).getElementType();
    }

    auto vector_type = mlir::VectorType::get({total_elements}, elem_type);

    // Handle different transfer types
    switch (transfer_type) {
      case DataTransferType::kLoadAndStore: {
        // Both are memrefs
        auto src_memref = src;
        auto dst_memref = dst;
        auto src_memref_type =
            llvm::cast<mlir::MemRefType>(src_memref.getType());
        auto dst_memref_type =
            llvm::cast<mlir::MemRefType>(dst_memref.getType());
        unsigned src_num_dims = src_memref_type.getRank();
        unsigned dst_num_dims = dst_memref_type.getRank();

        auto src_map = data_transfer_op.getSourceMap().value_or(
            mlir::AffineMap::getMultiDimIdentityMap(src_num_dims,
                                                    rewriter.getContext()));
        auto dst_map = data_transfer_op.getDestMap().value_or(
            mlir::AffineMap::getMultiDimIdentityMap(dst_num_dims,
                                                    rewriter.getContext()));

        return lowerAsLoadAndStore(
            rewriter, data_transfer_op, src_memref, dst_memref, src_indices,
            dst_indices, src_static_sizes, dst_static_sizes, src_num_dims,
            dst_num_dims, vector_type, src_map, dst_map);
      }

      case DataTransferType::kLoadAndSend: {
        // Source is memref, destination is FIFO
        auto src_memref = src;
        auto src_memref_type =
            llvm::cast<mlir::MemRefType>(src_memref.getType());
        unsigned num_dims = src_memref_type.getRank();

        auto identity_map = mlir::AffineMap::getMultiDimIdentityMap(
            num_dims, rewriter.getContext());
        auto src_map = data_transfer_op.getSourceMap().value_or(identity_map);

        auto dst_fifo_slot_type =
            llvm::cast<mlir::ktdf::FifoSlotType>(dst.getType());
        return lowerAsLoadAndSend(rewriter, data_transfer_op, src_memref,
                                  src_indices, src_static_sizes, num_dims,
                                  vector_type, src_map, dst_fifo_slot_type,
                                  is_broadcast_transfer, src_total_elements);
      }

      case DataTransferType::kReceiveAndStore: {
        // Source is FIFO, destination is memref
        auto dst_memref = dst;
        auto dst_memref_type =
            llvm::cast<mlir::MemRefType>(dst_memref.getType());
        unsigned num_dims = dst_memref_type.getRank();

        auto identity_map = mlir::AffineMap::getMultiDimIdentityMap(
            num_dims, rewriter.getContext());
        auto dst_map = data_transfer_op.getDestMap().value_or(identity_map);

        auto src_fifo_slot_type =
            llvm::cast<mlir::ktdf::FifoSlotType>(src.getType());
        return lowerAsReceiveAndStore(rewriter, data_transfer_op, dst_memref,
                                      dst_indices, dst_static_sizes, num_dims,
                                      vector_type, dst_map, src_fifo_slot_type);
      }
    }

    return mlir::failure();
  }

 private:
  const ResourceToUnits& components_;

  /// Lower as CompositeLoadAndStore
  mlir::LogicalResult lowerAsLoadAndStore(
      mlir::PatternRewriter& rewriter,
      mlir::ktdf::DataTransferOp data_transfer_op, mlir::Value src_memref,
      mlir::Value dst_memref, mlir::ValueRange src_indices,
      mlir::ValueRange dst_indices, llvm::ArrayRef<int64_t> src_static_sizes,
      llvm::ArrayRef<int64_t> dst_static_sizes, unsigned src_num_dims,
      unsigned dst_num_dims, mlir::VectorType vector_type,
      mlir::AffineMap src_map, mlir::AffineMap dst_map) const {
    // Build src_load_set from source sizes
    auto src_load_set = buildIntegerSetFromSizes(rewriter, src_static_sizes);

    // Build dst_store_set from destination sizes
    auto dst_store_set = buildIntegerSetFromSizes(rewriter, dst_static_sizes);

    // load_order and store_order must match their respective set
    // dimensionality.
    auto load_order = mlir::AffineMap::getMultiDimIdentityMap(
        src_num_dims, rewriter.getContext());
    auto store_order = mlir::AffineMap::getMultiDimIdentityMap(
        dst_num_dims, rewriter.getContext());

    // Build time_set: single iteration (d0) : d0=0
    llvm::SmallVector<mlir::AffineExpr, 1> time_exprs;
    llvm::SmallVector<bool, 1> time_eq_flags;
    time_exprs.push_back(mlir::getAffineDimExpr(0, rewriter.getContext()));
    time_eq_flags.push_back(true);  // equality constraint
    auto time_set = mlir::IntegerSet::get(1, 0, time_exprs, time_eq_flags);

    // Build time_order: identity map d0->d0
    auto time_order =
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());

    // time_addr_maps are zero-offset maps; result count must match the rank of
    // the corresponding memref (src for load side, dst for store side).
    auto makeZeroAddrMap = [&](unsigned rank) {
      llvm::SmallVector<mlir::AffineExpr> zero_exprs(
          rank, mlir::getAffineConstantExpr(0, rewriter.getContext()));
      return mlir::AffineMap::get(1, 0, zero_exprs, rewriter.getContext());
    };
    auto load_time_addr_map = makeZeroAddrMap(src_num_dims);
    auto store_time_addr_map = makeZeroAddrMap(dst_num_dims);

    // Create the CompositeLoadAndStoreOp
    mlir::agen::CompositeLoadAndStoreOp::create(
        rewriter, data_transfer_op.getLoc(), src_memref, dst_memref,
        /*dbgName=*/nullptr, src_map, src_indices, dst_map, dst_indices,
        src_load_set, load_order, dst_store_set, store_order, {}, time_set,
        time_order, load_time_addr_map, store_time_addr_map, vector_type);

    // Erase the original data_transfer operation
    rewriter.eraseOp(data_transfer_op);
    return mlir::success();
  }

  /// Lower as vector_load and send (L1 to FIFO)
  mlir::LogicalResult lowerAsLoadAndSend(
      mlir::PatternRewriter& rewriter,
      mlir::ktdf::DataTransferOp data_transfer_op, mlir::Value src_memref,
      mlir::ValueRange src_indices, llvm::ArrayRef<int64_t> src_static_sizes,
      unsigned num_dims, mlir::VectorType vector_type, mlir::AffineMap src_map,
      mlir::ktdf::FifoSlotType dst_fifo_slot_type, bool is_splat,
      int64_t src_total_elements) const {
    // Build load_set from source sizes
    auto load_set = buildIntegerSetFromSizes(rewriter, src_static_sizes);

    // Build load_order
    auto load_order = mlir::AffineMap::getMultiDimIdentityMap(
        num_dims, rewriter.getContext());

    // When splat: load only src_total_elements, then shuffle to full width.
    if (is_splat) {
      if (vector_type.getNumElements() % src_total_elements != 0) {
        data_transfer_op.emitError(
            "dst_total_elements must be divisible by src_total_elements "
            "for splat transfer");
        return mlir::failure();
      }
    }
    auto load_type = is_splat
                         ? mlir::VectorType::get({src_total_elements},
                                                 vector_type.getElementType())
                         : vector_type;

    // Create vector_load operation
    auto vector_load_op = mlir::agen::VectorLoadOp::create(
        rewriter, data_transfer_op.getLoc(), load_type, src_memref,
        /*dbgName=*/nullptr, src_map, src_indices, load_set, load_order);

    // For splat: broadcast loaded vector to full destination width.
    mlir::Value send_value = vector_load_op.getResult();
    if (is_splat) {
      send_value =
          insertSplatShuffle(rewriter, data_transfer_op.getLoc(), send_value,
                             src_total_elements, vector_type.getNumElements());
    }

    // Find the enclosing program_unit
    auto program_unit =
        data_transfer_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      data_transfer_op.emitError("data_transfer must be inside a program_unit");
      return mlir::failure();
    }

    // Resolve the destination unit from the FIFO dest attribute
    auto dest_unit_result = resolveUnitFromFifoAttr(
        dst_fifo_slot_type.getDest(), components_, rewriter, program_unit,
        data_transfer_op.getLoc(), data_transfer_op.getOperation());
    if (mlir::failed(dest_unit_result)) {
      return mlir::failure();
    }
    mlir::Value dest_unit = *dest_unit_result;

    // Create dataflow.send operation
    mlir::dataflow::SendOp::create(rewriter, data_transfer_op.getLoc(),
                                   dest_unit, send_value,
                                   /*dir=*/nullptr,
                                   /*dbgName=*/nullptr);

    // Erase the original data_transfer operation
    rewriter.eraseOp(data_transfer_op);
    return mlir::success();
  }

  /// Lower as receive and vector_store (FIFO to L1)
  mlir::LogicalResult lowerAsReceiveAndStore(
      mlir::PatternRewriter& rewriter,
      mlir::ktdf::DataTransferOp data_transfer_op, mlir::Value dst_memref,
      mlir::ValueRange dst_indices, llvm::ArrayRef<int64_t> dst_static_sizes,
      unsigned num_dims, mlir::VectorType vector_type, mlir::AffineMap dst_map,
      mlir::ktdf::FifoSlotType src_fifo_slot_type) const {
    // Build store_set from destination sizes
    auto store_set = buildIntegerSetFromSizes(rewriter, dst_static_sizes);

    // Build store_order
    auto store_order = mlir::AffineMap::getMultiDimIdentityMap(
        num_dims, rewriter.getContext());

    // Find the enclosing program_unit
    auto program_unit =
        data_transfer_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      data_transfer_op.emitError("data_transfer must be inside a program_unit");
      return mlir::failure();
    }

    // Resolve the source unit from the FIFO src attribute
    auto src_unit_result = resolveUnitFromFifoAttr(
        src_fifo_slot_type.getSrc(), components_, rewriter, program_unit,
        data_transfer_op.getLoc(), data_transfer_op.getOperation());
    if (mlir::failed(src_unit_result)) {
      return mlir::failure();
    }
    mlir::Value src_unit = *src_unit_result;

    // Create dataflow.receive operation
    auto receive_op = mlir::dataflow::ReceiveOp::create(
        rewriter, data_transfer_op.getLoc(), vector_type, src_unit,
        /*dbgName=*/nullptr);

    // Create vector_store operation
    mlir::agen::VectorStoreOp::create(
        rewriter, data_transfer_op.getLoc(), receive_op.getData(), dst_memref,
        /*dbgName=*/nullptr, dst_map, dst_indices, store_set, store_order);

    // Erase the original data_transfer operation
    rewriter.eraseOp(data_transfer_op);
    return mlir::success();
  }
};

}  // namespace

void scheduler::populateDataTransferLoweringPatterns(
    mlir::RewritePatternSet& patterns, const ResourceToUnits& components) {
  patterns.add<LowerDataTransferPattern>(patterns.getContext(), components);
}
