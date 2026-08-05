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

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
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

/// A dimension an AGEN composite transfer walks over its time axis: the index
/// position it occupies in the memref's static size list, its extent (the
/// number of time steps it contributes), and the coefficient that scales the
/// time-dimension induction variable to recover an offset in element units.
///
/// Every entry except (at most) the last has coefficient 1 -- it is a whole
/// non-unit outer dimension of the transfer. The last entry may instead be
/// the innermost dimension split into hardware vectors, in which case its
/// coefficient is the vector width (`lanes`): stepping that time dimension by
/// one advances by a whole vector, not by one element.
struct WalkedDim {
  unsigned position;
  int64_t extent;
  int64_t coeff;
};

/// Collect the dimensions a transfer walks over time, outermost first: every
/// non-unit dimension except the innermost (coefficient 1), followed by the
/// innermost dimension itself when it holds more than one hardware vector
/// (coefficient `lanes`, extent `innermost / lanes`). `sizes` must be
/// non-empty and its last (innermost) entry must be a multiple of `lanes` --
/// callers check both before calling this.
llvm::SmallVector<WalkedDim> collectWalkedDims(llvm::ArrayRef<int64_t> sizes,
                                               int64_t lanes) {
  llvm::SmallVector<WalkedDim> walked;
  for (unsigned i = 0, e = sizes.size() - 1; i < e; ++i) {
    if (sizes[i] != 1) {
      walked.push_back({i, sizes[i], /*coeff=*/1});
    }
  }
  const unsigned innermost_pos = sizes.size() - 1;
  const int64_t innermost_size = sizes[innermost_pos];
  const int64_t innermost_vectors = innermost_size / lanes;
  if (innermost_vectors > 1) {
    walked.push_back({innermost_pos, innermost_vectors, /*coeff=*/lanes});
  }
  return walked;
}

/// The time-dimension description of an AGEN composite transfer.
struct TimeDimensions {
  mlir::IntegerSet set;
  mlir::AffineMap order;
  mlir::AffineMap load_addr_map;
  mlir::AffineMap store_addr_map;
};

/// Build the AGEN time dimensions of a transfer split across `walked_src` /
/// `walked_dst` -- one entry per time dimension, outermost first (so the last
/// entry, if any, is the fastest-varying / innermost). Both lists must have
/// the same length and matching extents; the caller checks this.
///
/// The walked dimensions are listed slowest-first, so a plain identity
/// `time_order` map already visits them outermost-to-innermost -- no
/// permutation is needed.
///
/// `walked_src`/`walked_dst` are empty for a transfer that fits in a single
/// hardware vector; a single time step pinned to zero is emitted instead.
TimeDimensions buildTimeDimensions(mlir::MLIRContext* context,
                                   llvm::ArrayRef<WalkedDim> walked_src,
                                   llvm::ArrayRef<WalkedDim> walked_dst,
                                   unsigned src_rank, unsigned dst_rank) {
  const unsigned num_walked = walked_src.size();
  assert(walked_dst.size() == num_walked &&
         "expected the same number of walked dimensions on each side");
  const unsigned num_time_dims = std::max<unsigned>(1, num_walked);

  // time_set: walked dimensions span their extent; with no walked dimensions
  // the single time step is pinned to zero.
  llvm::SmallVector<mlir::AffineExpr> constraints;
  llvm::SmallVector<bool> eq_flags;
  if (num_walked == 0) {
    constraints.push_back(mlir::getAffineDimExpr(0, context));
    eq_flags.push_back(true);
  } else {
    for (unsigned dim = 0; dim < num_walked; ++dim) {
      auto expr = mlir::getAffineDimExpr(dim, context);
      constraints.push_back(expr);
      eq_flags.push_back(false);
      constraints.push_back(
          mlir::getAffineConstantExpr(walked_src[dim].extent - 1, context) -
          expr);
      eq_flags.push_back(false);
    }
  }
  auto time_set = mlir::IntegerSet::get(num_time_dims, 0, constraints,
                                        eq_flags);

  // time_order: identity. Correct because the walked dimensions above are
  // listed slowest-varying first, so time dimension 0 is already the
  // outermost and the last time dimension the innermost/fastest-varying.
  auto time_order =
      mlir::AffineMap::getMultiDimIdentityMap(num_time_dims, context);

  // The address maps have one result per memref dimension; a walked
  // dimension sits at the index position its side reported, scaled by its
  // coefficient, everything else is a zero offset from the base address.
  auto makeAddrMap = [&](unsigned rank, llvm::ArrayRef<WalkedDim> walked) {
    llvm::SmallVector<mlir::AffineExpr> results(
        rank, mlir::getAffineConstantExpr(0, context));
    for (auto [dim, w] : llvm::enumerate(walked)) {
      assert(w.position < rank && "walked dimension outside the memref rank");
      auto expr = mlir::getAffineDimExpr(dim, context);
      results[w.position] =
          w.coeff == 1 ? expr
                       : mlir::getAffineConstantExpr(w.coeff, context) * expr;
    }
    return mlir::AffineMap::get(num_time_dims, 0, results, context);
  };

  return TimeDimensions{time_set, time_order,
                        makeAddrMap(src_rank, walked_src),
                        makeAddrMap(dst_rank, walked_dst)};
}

/// Pattern to lower ktdf.data_transfer operations
struct LowerDataTransferPattern
    : public mlir::OpRewritePattern<mlir::ktdf::DataTransferOp> {
  LowerDataTransferPattern(mlir::MLIRContext* context,
                           const ResourceToUnits& components,
                           arch_view::ResourceKinds& resource_kinds)
      : OpRewritePattern(context),
        components_(components),
        resource_kinds_(resource_kinds) {}

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
  arch_view::ResourceKinds& resource_kinds_;

  /// Lower as CompositeLoadAndStore.
  ///
  /// An AGEN composite transfer moves at most one hardware vector per time
  /// step, so a transfer wider than that has to walk the remaining elements
  /// over AGEN time dimensions instead of widening `load_iv`. See
  /// `collectWalkedDims`/`buildTimeDimensions` above for how those time
  /// dimensions are derived.
  mlir::LogicalResult lowerAsLoadAndStore(
      mlir::PatternRewriter& rewriter,
      mlir::ktdf::DataTransferOp data_transfer_op, mlir::Value src_memref,
      mlir::Value dst_memref, mlir::ValueRange src_indices,
      mlir::ValueRange dst_indices, llvm::ArrayRef<int64_t> src_static_sizes,
      llvm::ArrayRef<int64_t> dst_static_sizes, unsigned src_num_dims,
      unsigned dst_num_dims, mlir::VectorType vector_type,
      mlir::AffineMap src_map, mlir::AffineMap dst_map) const {
    auto* context = rewriter.getContext();
    const int64_t total = vector_type.getNumElements();

    const auto lanes =
        getVectorLanes(vector_type.getElementType(), resource_kinds_);
    if (!lanes) {
      data_transfer_op.emitError(
          "cannot determine the hardware vector width: the architecture "
          "declares no compute resource kind");
      return mlir::failure();
    }

    // Sizes describing the elements covered by one AGEN vector transfer, and
    // the dimensions (if any) walked over time to cover the rest. Narrowed
    // below when the request is wider than one hardware vector.
    llvm::SmallVector<int64_t> load_sizes(src_static_sizes);
    llvm::SmallVector<int64_t> store_sizes(dst_static_sizes);
    mlir::VectorType load_iv_type = vector_type;
    llvm::SmallVector<WalkedDim> walked_src;
    llvm::SmallVector<WalkedDim> walked_dst;

    if (total > *lanes) {
      if (src_static_sizes.empty() || dst_static_sizes.empty()) {
        data_transfer_op.emitError()
            << "data transfer of " << total
            << " elements exceeds the hardware vector width of " << *lanes
            << " but has no dimensions to split";
        return mlir::failure();
      }

      if (src_static_sizes.back() % *lanes != 0 ||
          dst_static_sizes.back() % *lanes != 0) {
        data_transfer_op.emitError()
            << "data transfer of " << total
            << " elements exceeds the hardware vector width of " << *lanes
            << "; splitting requires the innermost source and destination "
               "sizes to be a multiple of the vector width, but they are "
            << src_static_sizes.back() << " and " << dst_static_sizes.back();
        return mlir::failure();
      }

      walked_src = collectWalkedDims(src_static_sizes, *lanes);
      walked_dst = collectWalkedDims(dst_static_sizes, *lanes);

      bool extents_match = walked_src.size() == walked_dst.size();
      if (extents_match) {
        for (auto [a, b] : llvm::zip(walked_src, walked_dst)) {
          if (a.extent != b.extent) {
            extents_match = false;
            break;
          }
        }
      }
      if (!extents_match) {
        data_transfer_op.emitError()
            << "source and destination walked dimensions must match to "
               "split a transfer of "
            << total << " elements across multiple vectors";
        return mlir::failure();
      }

      load_iv_type =
          mlir::VectorType::get({*lanes}, vector_type.getElementType());
      load_sizes.assign(src_static_sizes.size(), 1);
      load_sizes.back() = *lanes;
      store_sizes.assign(dst_static_sizes.size(), 1);
      store_sizes.back() = *lanes;
    }

    // Build src_load_set / dst_store_set from the per-vector sizes.
    auto src_load_set = buildIntegerSetFromSizes(rewriter, load_sizes);
    auto dst_store_set = buildIntegerSetFromSizes(rewriter, store_sizes);

    // load_order and store_order must match their respective set
    // dimensionality.
    auto load_order =
        mlir::AffineMap::getMultiDimIdentityMap(src_num_dims, context);
    auto store_order =
        mlir::AffineMap::getMultiDimIdentityMap(dst_num_dims, context);

    // Time dimensions: a single pinned step for a transfer of at most one
    // vector, one dimension per walked dimension otherwise.
    auto time_dims = buildTimeDimensions(context, walked_src, walked_dst,
                                        src_num_dims, dst_num_dims);

    // Create the CompositeLoadAndStoreOp
    mlir::agen::CompositeLoadAndStoreOp::create(
        rewriter, data_transfer_op.getLoc(), src_memref, dst_memref,
        /*dbgName=*/nullptr, src_map, src_indices, dst_map, dst_indices,
        src_load_set, load_order, dst_store_set, store_order, {},
        time_dims.set, time_dims.order, time_dims.load_addr_map,
        time_dims.store_addr_map, load_iv_type);

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
    mlir::RewritePatternSet& patterns, const ResourceToUnits& components,
    arch_view::ResourceKinds& resource_kinds) {
  patterns.add<LowerDataTransferPattern>(patterns.getContext(), components,
                                         resource_kinds);
}
