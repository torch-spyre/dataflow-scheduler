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
//
// This pass constructs a three-stage pipeline.
//
// This pass performs the following steps in order to create a three-stage
// pipeline:
//   Step 3: Create loops from linalg operations (tiling)
//   Step 4: Create pipeline with three stages (load, compute, store)
//   Step 5: Replace access tiles with memref.reinterpret_cast
//   Step 6: Cleanup
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Affine/Utils.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/IntegerSet.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <optional>

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/DataTransferLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDFTypes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"
#include "dataflow-scheduler/Transforms/Utils/CustomLinalgTiling.h"
#include "dataflow-scheduler/Transforms/Utils/Hoisting.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "ktir/Dialect/KTDP/KTDPTypes.h"

#define PASS_NAME "construct-three-stage-pipeline"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_CONSTRUCTTHREESTAGEPIPELINEPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable construction of three stage pipeline"),
    llvm::cl::init(false));

}  // unnamed namespace

namespace {

[[nodiscard]] auto getMemorySpace(mlir::TypedValue<mlir::MemRefType> memref)
    -> mlir::Attribute {
  if (const auto space = memref.getType().getMemorySpace(); space) {
    return space;
  }

  if (auto source = memref.getDefiningOp<mlir::ktdp::ConstructMemoryViewOp>()) {
    LDBG(2) << memref << " has erased memory space";
    return source.getMemorySpace();
  }

  if (auto source =
          memref.getDefiningOp<mlir::ktdp_lowering::ConstructMemoryViewOp>()) {
    LDBG(2) << memref << " has erased memory space (ktdp_lowering)";
    return source.getMemorySpace();
  }

  LDBG(2) << "missing memory space on " << memref;

  return nullptr;
}

void hoistOutOfLoops(mlir::Operation* op);

/// Get strides from the defining construct_memory_view op (ktdp or
/// ktdp_lowering). Falls back to the memref type's layout if no such defining
/// op exists or if dynamic strides are present (which are not supported yet).
template <typename OpT>
[[nodiscard]] static auto tryGetStridesFromOp(
    mlir::TypedValue<mlir::MemRefType> memref, llvm::StringRef op_name)
    -> std::optional<llvm::SmallVector<int64_t>> {
  if (auto source = memref.getDefiningOp<OpT>()) {
    LDBG(2) << memref << " getting strides from " << op_name;
    auto static_strides = source.getStaticStrides();
    if (llvm::any_of(static_strides, [](int64_t s) {
          return s == mlir::ShapedType::kDynamic;
        })) {
      LDBG(2) << "dynamic strides in " << op_name << ", falling back to type";
      return std::nullopt;
    }
    return llvm::to_vector(static_strides);
  }
  return std::nullopt;
}

/// Hoists each op in @p ops to just before @p anchor, then lets
/// hoistOutOfLoops() lift it as far as SSA allows.
///
/// An op is only moved when every non-batch operand's defining op already
/// dominates the anchor position.  An op dominates the anchor when it lives
/// either (a) in a strictly ancestor region of the anchor's region, or (b) in
/// the exact same block as the anchor and appears *before* it.  Ops in @p ops
/// are exempt from each other's dominance checks because they move together.
///
/// Any un-hoistable op emits a diagnostic and the function returns failure.
/// Well-formed IR (access tiles and memory views with loop-invariant operands)
/// will always succeed.
[[nodiscard]] static mlir::LogicalResult hoistOperations(
    llvm::ArrayRef<mlir::Operation*> ops, mlir::Operation* anchor) {
  // O(1) membership test: an op in this set is co-hoisted, so its results
  // dominate the anchor transitively once the batch has moved.
  llvm::SmallPtrSet<mlir::Operation*, 16> ops_set(ops.begin(), ops.end());

  mlir::Block* const anchor_block = anchor->getBlock();
  mlir::Region* const anchor_region = anchor_block->getParent();

  const auto dominatesAnchor = [&](mlir::Operation* def) -> bool {
    mlir::Region* const def_region = def->getParentRegion();
    // Case (a): def lives in a strictly ancestor region — always in scope.
    if (def_region->isProperAncestor(anchor_region)) return true;
    // Case (b): def is in the same block and strictly precedes the anchor.
    if (def->getBlock() == anchor_block) return def->isBeforeInBlock(anchor);
    // Any other arrangement does not dominate the anchor.
    return false;
  };

  for (auto* op : ops) {
    for (mlir::Value operand : op->getOperands()) {
      mlir::Operation* const def = operand.getDefiningOp();
      // Block arguments dominate the whole containing region — always safe.
      if (!def) continue;
      // Co-hoisted ops move together — dominance preserved transitively.
      if (ops_set.count(def)) continue;
      if (!dominatesAnchor(def)) {
        op->emitError(
            "cannot hoist op: an operand does not dominate the "
            "hoist anchor");
        return mlir::failure();
      }
    }
    op->moveBefore(anchor);
    hoistOutOfLoops(op);
  }
  return mlir::success();
}

[[nodiscard]] static auto getReassociationIndices(
    mlir::tensor::CollapseShapeOp collapse_op,
    mlir::tensor::ExpandShapeOp expand_op)
    -> llvm::SmallVector<mlir::ReassociationIndices> {
  return collapse_op ? collapse_op.getReassociationIndices()
                     : expand_op.getReassociationIndices();
}

[[nodiscard]] static auto findReshapeOps(mlir::Value value)
    -> std::pair<mlir::tensor::CollapseShapeOp, mlir::tensor::ExpandShapeOp> {
  mlir::tensor::CollapseShapeOp collapse_op;
  mlir::tensor::ExpandShapeOp expand_op;
  for (mlir::Operation* user : value.getUsers()) {
    if (!collapse_op) {
      if (auto c = mlir::dyn_cast<mlir::tensor::CollapseShapeOp>(user)) {
        collapse_op = c;
        if (expand_op) break;
        continue;
      }
    }
    if (!expand_op) {
      if (auto e = mlir::dyn_cast<mlir::tensor::ExpandShapeOp>(user)) {
        expand_op = e;
        if (collapse_op) break;
      }
    }
  }
  return {collapse_op, expand_op};
}

static void vectorizeToIndexAttrs(
    llvm::ArrayRef<int64_t> values, mlir::OpBuilder& builder,
    llvm::SmallVectorImpl<mlir::OpFoldResult>& out) {
  for (int64_t val : values) out.push_back(builder.getIndexAttr(val));
}

template <typename FindMapFunc>
[[nodiscard]] static auto findMapThroughExtractSlice(mlir::Value value,
                                                     FindMapFunc find_map)
    -> std::optional<mlir::AffineMap> {
  for (mlir::Operation* user : value.getUsers()) {
    if (auto extract = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user)) {
      if (auto map = find_map(extract.getResult())) return map;
    }
  }
  return std::nullopt;
}

[[nodiscard]] static auto validateTensorResults(mlir::Operation* op,
                                                unsigned expected_count,
                                                llvm::StringRef description)
    -> mlir::LogicalResult {
  if (op->getNumResults() < 1 || op->getNumResults() > expected_count) {
    op->emitError(description) << " expected 1-" << expected_count
                               << " results, got " << op->getNumResults();
    return mlir::failure();
  }
  if (!llvm::all_equal(op->getResultTypes())) {
    op->emitError(description) << " results must all be the same type";
    return mlir::failure();
  }
  return mlir::success();
}

[[nodiscard]] auto getStrides(mlir::TypedValue<mlir::MemRefType> memref)
    -> std::optional<llvm::SmallVector<int64_t>> {
  if (auto strides = tryGetStridesFromOp<mlir::ktdp::ConstructMemoryViewOp>(
          memref, "ktdp.construct_memory_view")) {
    return strides;
  }

  if (auto strides =
          tryGetStridesFromOp<mlir::ktdp_lowering::ConstructMemoryViewOp>(
              memref, "ktdp_lowering.construct_memory_view")) {
    return strides;
  }

  LDBG(2) << "no construct_memory_view found for " << memref
          << ", falling back to type";
  return std::nullopt;
}

[[nodiscard]] static auto getBaseMemrefFromAccessTile(
    mlir::TypedValue<mlir::ktdp::AccessTileType> access_tile)
    -> mlir::TypedValue<mlir::MemRefType> {
  mlir::Value tile_base;
  if (auto source =
          access_tile.getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>()) {
    tile_base = source.getBase();
  } else if (auto indirect =
                 access_tile.getDefiningOp<
                     mlir::ktdp_lowering::ConstructIndirectAccessTileOp>()) {
    tile_base = indirect.getBase();
  }

  return llvm::dyn_cast<mlir::TypedValue<mlir::MemRefType>>(tile_base);
}

[[nodiscard]] auto getMemorySpace(
    mlir::TypedValue<mlir::ktdp::AccessTileType> access_tile)
    -> mlir::Attribute {
  auto base = getBaseMemrefFromAccessTile(access_tile);
  if (!base) {
    LDBG(2) << access_tile << " is not derived from a buffer";
    return nullptr;
  }

  return getMemorySpace(base);
}

/// Hoists @p op as far out of the enclosing loops as SSA allows.
///
/// Loop-invariant memory views and full-shape casts end up before the outermost
/// loop, while a slice whose offset depends on a loop IV stays at its own
/// level.
void hoistOutOfLoops(mlir::Operation* op) {
  auto* const target =
      scheduler::findHoistingTarget(op, [](mlir::Region* region) -> bool {
        return llvm::isa_and_present<mlir::LoopLikeOpInterface>(
            region->getParentOp());
      });
  if (target != op) {
    op->moveBefore(target);
  }
}

/// The shape a transfer uses for an access tile of shape @p shape .
///
/// When @p collapse_leading_units is set, leading unit dimensions are dropped:
/// the indirect-address-buffer fill is rank-1 on both sides downstream, so a
/// pinned `1x32` row of addresses has to arrive as a plain `32`.
[[nodiscard]] auto transferShape(llvm::ArrayRef<int64_t> shape,
                                 bool collapse_leading_units)
    -> llvm::ArrayRef<int64_t> {
  if (collapse_leading_units) {
    while (shape.size() > 1 && shape.front() == 1) {
      shape = shape.drop_front();
    }
  }
  return shape;
}

/// A dimension-free map addressing the origin of a rank-@p rank memref.
///
/// Used where the whole offset is already folded into the reinterpret_cast.
[[nodiscard]] auto originMap(unsigned rank, mlir::MLIRContext* ctx)
    -> mlir::AffineMap {
  llvm::SmallVector<mlir::AffineExpr> origin(
      rank, mlir::getAffineConstantExpr(0, ctx));
  return mlir::AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, origin, ctx);
}

}  // namespace

namespace {

struct ConstructThreeStagePipelinePass
    : public impl::ConstructThreeStagePipelinePassBase<
          ConstructThreeStagePipelinePass> {
  ConstructThreeStagePipelinePass(const SchedulerExtContext& scheduler_ctx)
      : scheduler_ctx_(scheduler_ctx) {}

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    ConstructThreeStagePipelinePassBase::getDependentDialects(registry);
  }

  void runOnOperation() final;

 private:
  const SchedulerExtContext& schedulerExtContext() const {
    return scheduler_ctx_;
  }

  // Process a single function
  void runOnFunc(mlir::func::FuncOp func_op);

  // Reset state between function processing
  void resetState();

  // Determine tile sizes based on vector_length from linalg operation
  llvm::SmallVector<int64_t> determineTileSizes(
      mlir::linalg::LinalgOp linalgOp);

  // Create loops from linalg operations by tiling
  mlir::LogicalResult createLoopsFromLinalg(
      llvm::SmallVectorImpl<mlir::linalg::GenericOp>& linalg_ops);

  // Create a 3-stage pipeline inside innermost_loop, with one stage for loads,
  // computes, stores.
  mlir::LogicalResult createPipeline(mlir::scf::ForOp innermost_loop,
                                     mlir::ktdf_arch::ExecutionUnitOp compute);

  // Create linalg compute operations in stage 2
  void createComputeOps(mlir::OpBuilder& builder, mlir::Location loc,
                        mlir::ktdf::PrivateOp private_op);

  // Create data transfer operations for loads or stores
  // is_load=true: transfer from access_tile to FIFO slot
  // is_load=false: transfer from FIFO slot to access_tile
  // private_result_offset: index into private_op results where FIFO slots for
  //   this transfer start.
  void createDataTransfers(mlir::OpBuilder& builder, mlir::Location loc,
                           mlir::ktdf::PrivateOp private_op,
                           llvm::ArrayRef<int64_t> tile_sizes, bool is_load,
                           size_t private_result_offset,
                           mlir::ktdf_arch::ExecutionUnitOp compute);

  // Record the indirect access tiles of this function together with the
  // indirect-address-buffer fill that populates the buffer they read.
  // Must run before load/store collection: the fill is a ktdp.load/ktdp.store
  // pair that is not a pipeline load/store, and counting it as one would
  // misalign the FIFO slots createPrivateOp derives from the linalg operands.
  mlir::LogicalResult classifyIndirectAccess(mlir::func::FuncOp func_op);

  // True when `op` is one half of the indirect-address-buffer fill.
  [[nodiscard]] bool isIndAddrBufFillOp(mlir::Operation* op) {
    return ind_addr_buf_fill_ &&
           (op == ind_addr_buf_fill_->load.getOperation() ||
            op == ind_addr_buf_fill_->store.getOperation());
  }

  // True when `op` defines one of the access tiles of the fill, whose transfer
  // is rank-1 on both sides (see transferShape).
  [[nodiscard]] bool isIndAddrBufFillTile(mlir::Operation* op) {
    return ind_addr_buf_fill_ &&
           (op == ind_addr_buf_fill_->source_tile.getDefiningOp() ||
            op == ind_addr_buf_fill_->dest_tile.getDefiningOp());
  }

  // Move the indirect-address-buffer fill into the stage at the builder's
  // insertion point and rewrite it as a ktdf.data_transfer.
  void createIndAddrBufFill(mlir::OpBuilder& builder, mlir::Location loc,
                            mlir::ktdf_arch::ExecutionUnitOp compute);

  // Elements a transfer of `element_type` may move per time step, bounded by
  // the compute unit's vector width.  A type the unit declares no lanes for --
  // the buffer's index elements -- falls back to one element per step.
  //
  // FIXME: The DataTransferLowering needs to know the amount of elements it is
  //        allowed to transfer in a single time step. This is called the
  //        throttle, and is determined by the vector width of the target
  //        compute. However, path expansion will have to update the throttle on
  //        the whole chain of transfers based on the bottleneck!
  [[nodiscard]] int64_t getThrottle(mlir::Type element_type,
                                    mlir::ktdf_arch::ExecutionUnitOp compute) {
    return std::max<int64_t>(
        1, compute.getFeature<mlir::ktdf_arch::feature::SIMD>().getLanes(
               element_type));
  }

  // Emit the ktdf.ind_data_transfer for `ind_tile`, projecting the tiled loop
  // IVs through its per-dimension subscript maps.  The FIFO side reuses
  // `fifo_slot`/`fifo_sizes`.  Returns nullptr after emitting a diagnostic.
  mlir::ktdf::IndDataTransferOp createIndDataTransfer(
      mlir::OpBuilder& builder, mlir::Location loc,
      mlir::ktdp_lowering::ConstructIndirectAccessTileOp ind_tile,
      mlir::AffineMap indexing_map, llvm::ArrayRef<int64_t> tile_sizes,
      llvm::ArrayRef<mlir::Value> loop_ivs,
      llvm::ArrayRef<int64_t> variables_extents,
      mlir::TypedValue<mlir::ktdf::FifoSlotType> fifo_slot,
      llvm::ArrayRef<int64_t> fifo_sizes, bool is_load);

  // Emit the memory_space_cast (unless the space already maps to itself) plus
  // the memref.reinterpret_cast that address a `tile_dims`-shaped window of
  // `memory_view` at the element offset implied by `per_dim_indices` (empty for
  // offset 0).  Returns nullptr after emitting a diagnostic on `err_anchor`.
  mlir::Value emitReinterpretCast(
      mlir::OpBuilder& builder, mlir::Location loc,
      mlir::TypedValue<mlir::MemRefType> memory_view,
      llvm::SmallVector<mlir::Value>& per_dim_indices,
      llvm::ArrayRef<int64_t> tile_dims, mlir::Operation* err_anchor);

  // Create ktdf.private operation with FIFO slots and tokens
  // Returns the created private operation or failure
  llvm::FailureOr<mlir::ktdf::PrivateOp> createPrivateOp(
      mlir::OpBuilder& builder, mlir::Location loc,
      mlir::ktdf_arch::ExecutionUnitOp compute);

  // Annotate loops with loop_type attributes based on linalg iterator types
  mlir::LogicalResult annotateLoopsWithIteratorTypes(
      llvm::ArrayRef<mlir::Operation*> loops,
      mlir::linalg::GenericOp generic_op);

  // Delete op and recursively delete unused chain of operands
  void deleteOpAndUnusedChainOfOperands(mlir::Operation* op);

  // Collect loop induction variables from tiled loops
  [[nodiscard]] llvm::SmallVector<mlir::Value> collectLoopInductionVars() const;

  // Replace construct_access_tile with memref.reinterpret_cast
  mlir::LogicalResult replaceAccessTilesWithReinterpretCast(
      mlir::func::FuncOp func_op);

  [[nodiscard]] auto mapMemorySpace(mlir::Attribute declared_space) const
      -> mlir::Attribute {
    if (const auto mapped = mem_space_map_.lookup(declared_space); mapped) {
      return mapped;
    }
    return declared_space;
  }

  // Compute offset for reinterpret_cast from indices and strides.
  // An offset that is statically zero is returned as an attribute rather than a
  // materialized arith.constant, so the cast's only operand stays the memory
  // view and hoistOutOfLoops can lift it out of the loop nest.
  mlir::OpFoldResult computeReinterpretCastOffset(
      mlir::OpBuilder& builder, mlir::Location loc,
      llvm::SmallVector<mlir::Value>& indices,
      llvm::SmallVector<int64_t>& strides);

  // Clean up operations after pipeline creation
  void cleanupOperations();

  // Member variables
  const SchedulerExtContext& scheduler_ctx_;

  mlir::ktdf_arch::Mapping* mapping_;
  llvm::DenseMap<mlir::Attribute, mlir::Attribute> mem_space_map_;

  // Collected ktdp.load and ktdp.store operations
  llvm::SmallVector<mlir::ktdp::LoadOp> load_ops_;
  llvm::SmallVector<mlir::ktdp::StoreOp> store_ops_;
  llvm::SmallVector<mlir::linalg::LinalgOp> compute_ops_;

  // An indirect-address-buffer fill: a row of addresses loaded from global,
  // reshaped, and stored into the buffer view, usually under an scf.if that
  // limits the refill to the first iteration.
  struct IndAddrBufFill {
    mlir::ktdp::LoadOp load;
    mlir::ktdp::StoreOp store;
    mlir::scf::IfOp guard;  // nearest enclosing scf.if; null when unguarded
    // The access tiles the fill reads and writes.  Captured here rather than
    // re-read from the load and store: step 5 rewrites those operands to
    // memrefs while it is still iterating, and the generated accessors would
    // then fail their cast to an access tile.
    mlir::Value source_tile;
    mlir::Value dest_tile;
  };

  // The indirect access tile of the function (singular), the buffer views it
  // reads, and the fill that populates it. We expect at most one per
  // function.
  std::optional<mlir::ktdp_lowering::ConstructIndirectAccessTileOp> ind_tile_;
  mlir::Value iab_view_;
  std::optional<IndAddrBufFill> ind_addr_buf_fill_;

  // Whether the indirect tile feeds a load (gather) or a store (scatter), which
  // decides the stage the fill and the indirect transfer go into.
  std::optional<bool> ind_tile_is_load_;

  // Whole-view casts at the origin, keyed by memory view, shared across the
  // access tiles that ask for one (see emitReinterpretCast).
  llvm::DenseMap<mlir::Value, mlir::Value> origin_casts_;

  // Indirect transfers awaiting the reinterpret_casts created in step 5.
  llvm::SmallVector<
      std::pair<mlir::ktdf::IndDataTransferOp,
                mlir::ktdp_lowering::ConstructIndirectAccessTileOp>>
      ind_transfers_;

  // Tiled loops from linalg tiling (outermost to innermost)
  llvm::SmallVector<mlir::Operation*> tiled_loops_;

  // Tile sizes determined from linalg operation
  llvm::SmallVector<int64_t> tile_sizes_;

  // Operations to delete after pipeline creation
  llvm::SmallVector<mlir::Operation*> ops_to_delete_;

  // Builder for constants at function start
  std::optional<mlir::OpBuilder> const_builder_;
};

void ConstructThreeStagePipelinePass::resetState() {
  load_ops_.clear();
  store_ops_.clear();
  compute_ops_.clear();
  tiled_loops_.clear();
  tile_sizes_.clear();
  ops_to_delete_.clear();
  ind_tile_.reset();
  iab_view_ = nullptr;
  ind_addr_buf_fill_.reset();
  ind_tile_is_load_.reset();
  ind_transfers_.clear();
  origin_casts_.clear();
  const_builder_.reset();
}

mlir::LogicalResult ConstructThreeStagePipelinePass::classifyIndirectAccess(
    mlir::func::FuncOp func_op) {
  auto walk_result = func_op.walk(
      [&](mlir::ktdp_lowering::ConstructIndirectAccessTileOp tile) {
        if (ind_tile_) {
          tile.emitError(
              "more than one indirect access tile in a single pipeline is not "
              "supported");
          return mlir::WalkResult::interrupt();
        }
        ind_tile_ = tile;
        iab_view_ = tile.getIndAddrBufMemref();
        return mlir::WalkResult::advance();
      });

  if (walk_result.wasInterrupted()) return mlir::failure();

  if (!ind_tile_) return mlir::success();

  mlir::ktdp_lowering::ConstructIndirectAccessTileOp ind_tile = *ind_tile_;

  // The tile's single result must feed exactly one ktdp.load (gather) or one
  // ktdp.store (scatter).
  if (!ind_tile->hasOneUse())
    return ind_tile.emitError(
        "indirect access tile must have exactly one user");
  mlir::Operation* sole_user = *ind_tile->getUsers().begin();
  if (!mlir::isa<mlir::ktdp::LoadOp, mlir::ktdp::StoreOp>(sole_user))
    return ind_tile.emitError(
        "indirect access tile is not used by a ktdp.load or a ktdp.store");
  ind_tile_is_load_ = mlir::isa<mlir::ktdp::LoadOp>(sole_user);

  // The fill is the ktdp.store that writes into the buffer view.  Recognising
  // it structurally -- through the view the indirect tile itself names --
  // avoids guessing from tile ranks or from a memory-space string.
  llvm::SmallVector<mlir::ktdp::StoreOp> fill_stores;
  func_op.walk([&](mlir::ktdp::StoreOp store) {
    auto tile = store.getAccessTile()
                    .getDefiningOp<mlir::ktdp::ConstructAccessTilesOp>();
    if (tile && tile.getBase() == iab_view_) {
      fill_stores.push_back(store);
    }
  });
  if (fill_stores.empty()) {
    return ind_tile.emitError(
        "no indirect-address-buffer fill found for this indirect access tile");
  }
  if (fill_stores.size() > 1) {
    return fill_stores[1].emitError(
        "more than one indirect-address-buffer fill is not supported");
  }

  // Trace the stored data back through shape-only ops to the ktdp.load that
  // read the row of addresses from global.
  IndAddrBufFill fill;
  fill.store = fill_stores.front();
  llvm::SmallVector<mlir::Operation*> reshapes;
  for (mlir::Value data = fill.store.getDataTile(); !fill.load;) {
    mlir::Operation* def = data.getDefiningOp();
    if (!def) break;
    if (auto load = mlir::dyn_cast<mlir::ktdp::LoadOp>(def)) {
      fill.load = load;
      break;
    }
    if (!mlir::isa<mlir::tensor::CollapseShapeOp, mlir::tensor::ExpandShapeOp,
                   mlir::tensor::CastOp>(def)) {
      break;
    }
    reshapes.push_back(def);
    data = def->getOperand(0);
  }
  if (!fill.load) {
    return fill.store.emitError(
        "indirect-address-buffer fill does not trace back to a ktdp.load "
        "through shape-only operations");
  }

  fill.guard = fill.store->getParentOfType<mlir::scf::IfOp>();
  fill.source_tile = fill.load.getAccessTile();
  fill.dest_tile = fill.store.getAccessTile();

  // The fill transfer is emitted as rank-1 on both sides (transferShape drops
  // leading unit dims).  Verify that both tiles collapse to rank-1 here so
  // that a malformed tile is caught with a clear diagnostic rather than
  // silently producing a multi-dimensional transfer downstream.
  const auto checkCollapsesToRank1 =
      [](mlir::Value tile) -> mlir::LogicalResult {
    llvm::ArrayRef<int64_t> shape =
        mlir::cast<mlir::ktdp::AccessTileType>(tile.getType()).getShape();
    while (shape.size() > 1 && shape.front() == 1) shape = shape.drop_front();
    if (shape.size() != 1)
      return tile.getDefiningOp()->emitError(
                 "indirect-address-buffer fill tile must collapse to rank-1 "
                 "after "
                 "dropping leading unit dimensions, but shape ")
             << mlir::cast<mlir::ktdp::AccessTileType>(tile.getType())
             << " has non-unit leading dimension(s)";
    return mlir::success();
  };
  if (mlir::failed(checkCollapsesToRank1(fill.source_tile)) ||
      mlir::failed(checkCollapsesToRank1(fill.dest_tile)))
    return mlir::failure();

  ind_addr_buf_fill_ = fill;
  ops_to_delete_.append(reshapes);

  return mlir::success();
}

// Given a shape and a target vector_length, compute per-dimension tile sizes
// working rightmost-first until the product of covered dims reaches
// vector_length.  Uncovered dims (leftmost) get tile size 1 so a loop is
// generated for them; covered dims get the largest value that divides both
// the dim size and the remaining vector-length product.
static llvm::SmallVector<int64_t> computeTileSizesFromShape(
    llvm::ArrayRef<int64_t> shape, int64_t vector_length) {
  int64_t rank = shape.size();

  // Find how many rightmost dims are needed to cover vector_length.
  int64_t product = 1;
  int64_t covered_dims = 0;
  for (int64_t i = rank - 1; i >= 0; --i) {
    product *= shape[i];
    covered_dims++;
    if (product >= vector_length) break;
  }

  llvm::SmallVector<int64_t> tile_sizes(rank);

  // Uncovered dims (leftmost) get tile size 1 — a loop is generated that
  // iterates over the full extent one element at a time.
  for (int64_t i = 0; i < rank - covered_dims; ++i) tile_sizes[i] = 1;

  // Covered dims get a size that divides both the dim and the remaining
  // product.
  int64_t remaining = vector_length;
  for (int64_t i = rank - 1; i >= rank - covered_dims; --i) {
    int64_t ts = std::min(std::gcd(shape[i], remaining), shape[i]);
    tile_sizes[i] = ts;
    remaining /= ts;
    if (remaining <= 1) remaining = 1;
  }

  return tile_sizes;
}

llvm::SmallVector<int64_t> ConstructThreeStagePipelinePass::determineTileSizes(
    mlir::linalg::LinalgOp linalg_op) {
  assert(mlir::succeeded(validateTensorResults(linalg_op, 2, "linalg op")) &&
         "linalg op validation failed");

  auto compute = mapping_->getOrMap(
      linalg_op, mapping_->byKind().getDefaultCompute().getKind());
  assert(compute && "no default compute resource");

  mlir::ShapedType shaped_type =
      mlir::dyn_cast<mlir::ShapedType>(linalg_op->getResult(0).getType());
  assert(shaped_type && shaped_type.hasRank());

  mlir::Type elem_type = shaped_type.getElementType();

  auto simd_feature = compute.getFeature<mlir::ktdf_arch::feature::SIMD>();
  const auto vector_length =
      std::max(simd_feature.getLanes(elem_type), int64_t(1));

  // The tile-sizes vector must cover every loop dimension (getNumLoops()).
  // Reduction dimensions are skipped (tile size = 0) so the tiling infra does
  // not create a loop for them.  Only parallel dimensions are tiled, using the
  // same rightmost-first shape-based algorithm as before.
  const auto iterator_types = linalg_op.getIteratorTypesArray();
  const int64_t num_loops = static_cast<int64_t>(iterator_types.size());

  // Collect the loop bounds for each dimension by querying the op directly.
  // getStaticLoopRanges() scans all operand indexing maps so it works
  // regardless of whether any single operand has a full-rank map (e.g. when
  // an input has a broadcast/projection indexing map that skips some dims).
  llvm::SmallVector<int64_t> loop_ranges = linalg_op.getStaticLoopRanges();
  assert(static_cast<int64_t>(loop_ranges.size()) == num_loops &&
         "getStaticLoopRanges size must match number of loops");

  llvm::SmallVector<int64_t> parallel_loop_indices;
  llvm::SmallVector<int64_t> parallel_shape;
  for (int64_t i = 0; i < num_loops; ++i) {
    if (iterator_types[i] == mlir::utils::IteratorType::parallel) {
      parallel_loop_indices.push_back(i);
      parallel_shape.push_back(loop_ranges[i]);
    }
  }

  auto parallel_tile_sizes =
      computeTileSizesFromShape(parallel_shape, vector_length);

  llvm::SmallVector<int64_t> tile_sizes(num_loops, 0);
  for (unsigned k = 0; k < parallel_loop_indices.size(); ++k)
    tile_sizes[parallel_loop_indices[k]] = parallel_tile_sizes[k];

  LLVM_DEBUG({
    llvm::dbgs() << "    Parallel shape: [";
    for (int64_t i = 0; i < static_cast<int64_t>(parallel_shape.size()); ++i) {
      llvm::dbgs() << parallel_shape[i];
      if (i + 1 < static_cast<int64_t>(parallel_shape.size()))
        llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n    Tile sizes (all loops): [";
    for (int64_t i = 0; i < num_loops; ++i) {
      llvm::dbgs() << tile_sizes[i];
      if (i + 1 < num_loops) llvm::dbgs() << ", ";
    }
    llvm::dbgs() << "]\n";
  });

  return tile_sizes;
}

mlir::LogicalResult
ConstructThreeStagePipelinePass::annotateLoopsWithIteratorTypes(
    llvm::ArrayRef<mlir::Operation*> loops,
    mlir::linalg::GenericOp generic_op) {
  assert(generic_op);

  // The tiling infra only generates a loop for dims whose tile_size > 0.
  // Build the ordered list of iterator types for those dims so that loops[k]
  // maps to the k-th tiled dim (in loop-index order), not positionally to
  // iteratorTypes[k].
  const auto& iterator_types = generic_op.getIteratorTypesArray();
  llvm::SmallVector<mlir::utils::IteratorType> tiled_iterator_types;
  for (size_t i = 0; i < iterator_types.size(); ++i) {
    if (i < tile_sizes_.size() && tile_sizes_[i] > 0)
      tiled_iterator_types.push_back(iterator_types[i]);
  }

  for (size_t i = 0; i < loops.size() && i < tiled_iterator_types.size(); ++i) {
    auto for_op = mlir::dyn_cast<mlir::scf::ForOp>(loops[i]);
    assert(for_op);

    mlir::ktdf::LoopType loop_type;
    if (tiled_iterator_types[i] == mlir::utils::IteratorType::parallel) {
      loop_type = mlir::ktdf::LoopType::ParallelLoop;
    } else if (tiled_iterator_types[i] ==
               mlir::utils::IteratorType::reduction) {
      loop_type = mlir::ktdf::LoopType::ReductionLoop;
    } else {
      for_op->emitError("Unsupported iterator type");
      signalPassFailure();
      return mlir::failure();
    }

    auto loop_type_attr =
        mlir::ktdf::LoopTypeAttr::get(&getContext(), loop_type);
    for_op->setAttr("loop_type", loop_type_attr);

    LDBG(1) << "  Annotated loop " << i << " with loop_type: "
            << (loop_type == mlir::ktdf::LoopType::ParallelLoop ? "parallel"
                                                                : "reduction")
            << "";
  }
  return mlir::success();
}

mlir::LogicalResult ConstructThreeStagePipelinePass::createLoopsFromLinalg(
    llvm::SmallVectorImpl<mlir::linalg::GenericOp>& linalg_ops) {
  LDBG(1) << "Creating SCF loops from linalg.generic operations";

  assert(linalg_ops.size() <= 1 &&
         "Currently only supporting one linalg.generic operation after fusion");

  mlir::IRRewriter rewriter(&getContext());
  for (auto& linalg_op : linalg_ops) {
    LDBG(1) << "  Processing: " << linalg_op->getName() << "";

    rewriter.setInsertionPoint(linalg_op);

    // Determine tile sizes from output operand shape (needed for loop
    // creation)
    tile_sizes_ = determineTileSizes(linalg_op);

    if (tile_sizes_.empty()) {
      linalg_op.emitError("Could not determine tile sizes");
      signalPassFailure();
      return mlir::failure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "  Tile sizes: ";
      for (int64_t i : tile_sizes_) llvm::dbgs() << i << ", ";
      llvm::dbgs() << "\n";
    });

    // Tile the linalg.generic operation with the computed tile sizes
    // Use custom tiling that doesn't create iter_args.
    mlir::linalg::LinalgTilingOptions tiling_options;
    tiling_options.setTileSizes(tile_sizes_);
    tiling_options.setLoopType(mlir::linalg::LinalgTilingLoopType::Loops);

    llvm::FailureOr<mlir::linalg::TiledLinalgOp> tiled_result =
        customTileLinalgOp(rewriter, linalg_op, tiling_options);

    if (mlir::failed(tiled_result)) {
      linalg_op.emitError("Failed to tile linalg operation");
      signalPassFailure();
      return mlir::failure();
    }

    // Replace uses of original results with tiled results
    assert(tiled_result->tensorResults.size() == linalg_op->getNumResults() &&
           "Tiled result count must match original result count");
    for (size_t i = 0; i < linalg_op->getNumResults(); ++i) {
      rewriter.replaceAllUsesWith(linalg_op->getResult(i),
                                  tiled_result->tensorResults[i]);
    }

    // Erase the original linalg operation. It has been replaced by a loop nest
    // containing the tiled linalg operation.
    rewriter.eraseOp(linalg_op);
    linalg_op =
        mlir::cast<mlir::linalg::GenericOp>(tiled_result->op.getOperation());

    // Annotate loops with loop_type attributes based on iterator types
    if (!tiled_result->loops.empty()) {
      auto generic_op = mlir::dyn_cast<mlir::linalg::GenericOp>(
          tiled_result->op.getOperation());
      if (failed(
              annotateLoopsWithIteratorTypes(tiled_result->loops, generic_op)))
        return mlir::failure();
    }

    LDBG(1) << "  Successfully tiled linalg operation";

    // Store the tiled loops for later pipeline creation
    if (!tiled_result->loops.empty()) {
      tiled_loops_.assign(tiled_result->loops.begin(),
                          tiled_result->loops.end());
    }
  }
  return mlir::success();
}

[[nodiscard]] static auto computeNumElements(mlir::RankedTensorType tensor_type)
    -> int64_t {
  int64_t num_elements = 1;
  for (int64_t dim : tensor_type.getShape()) num_elements *= dim;
  return num_elements;
}

llvm::FailureOr<mlir::ktdf::PrivateOp>
ConstructThreeStagePipelinePass::createPrivateOp(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ktdf_arch::ExecutionUnitOp compute) {
  mlir::ktdf::TokenType token_type = mlir::ktdf::TokenType::get(&getContext());
  llvm::SmallVector<mlir::Type> private_result_types;

  // Group FIFO slot types by their FIFO type to preserve ordering
  llvm::MapVector<std::pair<mlir::Attribute, mlir::Attribute>,
                  llvm::SmallVector<mlir::ktdf::FifoSlotType>>
      fifo_type_groups;

  // Add FIFO slot types for load operations.
  // The FIFO must hold one per-iteration tile of the input.  compute_ops_[0]
  // is the tiled linalg inside the loop body; its DPS input operand types are
  // already the tiled shapes, so use them directly.
  assert(compute_ops_.size() == 1 &&
         "expected exactly one linalg compute op after fusion");
  for (size_t i = 0; i < load_ops_.size(); ++i) {
    mlir::ktdp::LoadOp load_op = load_ops_[i];
    auto tiled_type = mlir::dyn_cast<mlir::RankedTensorType>(
        compute_ops_[0].getDpsInputOperand(i)->get().getType());
    if (!tiled_type) {
      load_op.emitError("Expected RankedTensorType for tiled linalg input");
      signalPassFailure();
      return mlir::failure();
    }
    int64_t num_elements = computeNumElements(tiled_type);
    mlir::Type element_type = tiled_type.getElementType();

    // Get FIFO attributes for this specific load operation
    const auto load_src =
        mapMemorySpace(getMemorySpace(load_op.getAccessTile()));
    const auto load_dest = compute.getKind();
    auto load_key = std::make_pair(load_src, load_dest);

    auto fifo_slot_type = mlir::ktdf::FifoSlotType::get(
        &getContext(), load_src, load_dest, num_elements, element_type);
    fifo_type_groups[load_key].push_back(fifo_slot_type);
    private_result_types.push_back(fifo_slot_type);
  }

  // Add FIFO slot types for store operations.
  // Size from the tiled linalg DPS init operand type so that write_to_fifo
  // (stage 2) and data_transfer (stage 3) agree on the per-iteration tile
  // size.
  for (size_t i = 0; i < store_ops_.size(); ++i) {
    mlir::ktdp::StoreOp store_op = store_ops_[i];
    auto tiled_init_type = mlir::dyn_cast<mlir::RankedTensorType>(
        compute_ops_[0].getDpsInitOperand(i)->get().getType());
    if (!tiled_init_type) {
      store_op.emitError(
          "Expected RankedTensorType for tiled linalg init operand");
      signalPassFailure();
      return mlir::failure();
    }
    int64_t num_elements = computeNumElements(tiled_init_type);
    mlir::Type element_type = tiled_init_type.getElementType();

    // Get FIFO attributes for this specific store operation
    const auto store_src = compute.getKind();
    const auto store_dest =
        mapMemorySpace(getMemorySpace(store_op.getAccessTile()));
    auto store_key = std::make_pair(store_src, store_dest);

    auto fifo_slot_type = mlir::ktdf::FifoSlotType::get(
        &getContext(), store_src, store_dest, num_elements, element_type);
    fifo_type_groups[store_key].push_back(fifo_slot_type);
    private_result_types.push_back(fifo_slot_type);
  }

  // Add three token types for the three stages
  private_result_types.push_back(token_type);
  private_result_types.push_back(token_type);
  private_result_types.push_back(token_type);

  // Create ktdf.private operation
  auto private_op =
      mlir::ktdf::PrivateOp::create(builder, loc, private_result_types);
  mlir::Region& private_region = private_op.getRegion();
  mlir::Block* private_body = &private_region.front();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(private_body);

  // Create ktdf.fifo.allocate operations grouped by FIFO type
  // This preserves the order of load/stores within each FIFO type
  llvm::SmallVector<mlir::Value> fifo_results;
  for (auto& [fifo_type, slot_types] : fifo_type_groups) {
    // Create a single fifo.allocate with multiple results for this FIFO type
    llvm::SmallVector<mlir::Type> result_types(slot_types.begin(),
                                               slot_types.end());
    auto fifo_alloc_op = mlir::ktdf::FifoAllocateOp::create(
        builder, loc, mlir::TypeRange{result_types}, mlir::ValueRange{});

    // Add all results from this allocation to fifo_results
    for (unsigned i = 0; i < fifo_alloc_op.getNumResults(); ++i) {
      fifo_results.push_back(fifo_alloc_op.getResult(i));
    }
  }

  // Create three tokens
  auto t1 = mlir::ktdf::CreateTokenOp::create(builder, loc, token_type);
  auto t2 = mlir::ktdf::CreateTokenOp::create(builder, loc, token_type);
  auto t3 = mlir::ktdf::CreateTokenOp::create(builder, loc, token_type);
  fifo_results.push_back(t1.getResult());
  fifo_results.push_back(t2.getResult());
  fifo_results.push_back(t3.getResult());

  // Yield all results
  mlir::ktdf::PrivateYieldOp::create(builder, loc, fifo_results);

  return private_op;
}

mlir::LogicalResult ConstructThreeStagePipelinePass::createPipeline(
    mlir::scf::ForOp innermost_loop, mlir::ktdf_arch::ExecutionUnitOp compute) {
  LDBG(1) << "Creating ktdf.pipeline with three stages";

  compute_ops_.clear();
  innermost_loop.getBody()->walk([&](mlir::linalg::LinalgOp linalg_op) {
    compute_ops_.push_back(linalg_op);
  });

  LDBG(1) << "  Found " << compute_ops_.size() << " linalg compute operations";

  // Save original scf.yield operands to ops_to_delete_ before changing the
  // yield. This ensures tensor.insert_slice and linalg.generic get cleaned up
  // later.
  auto yield_op =
      mlir::cast<mlir::scf::YieldOp>(innermost_loop.getBody()->getTerminator());
  for (mlir::Value operand : yield_op.getOperands()) {
    if (auto* def_op = operand.getDefiningOp()) {
      ops_to_delete_.push_back(def_op);
    }
  }

  // Fix scf.yield to yield iter_args instead of linalg results. This allows
  // tensor.insert_slice to get cleaned up (no more uses) and the dead iter args
  // will be cleaned by a dead code elimination pass.
  mlir::OpBuilder yield_builder(yield_op);
  llvm::SmallVector<mlir::Value> new_yield_operands;
  for (mlir::BlockArgument iter_arg : innermost_loop.getRegionIterArgs()) {
    new_yield_operands.push_back(iter_arg);
  }
  mlir::scf::YieldOp::create(yield_builder, yield_op.getLoc(),
                             new_yield_operands);
  yield_op.erase();

  // Create ktdf.pipeline operation at the start of the loop body.
  mlir::OpBuilder builder(innermost_loop.getBodyRegion());

  mlir::ktdf::PipelineOp::create(
      builder, innermost_loop.getLoc(),
      [&](mlir::OpBuilder& builder, mlir::Location loc) {
        // Create ktdf.private operation with FIFO slots and tokens
        auto private_op_result = createPrivateOp(builder, loc, compute);
        if (mlir::failed(private_op_result)) return;
        auto private_op = *private_op_result;

        // Tokens are at the end: fifo_count + 0, fifo_count + 1, fifo_count + 2
        size_t fifo_count = load_ops_.size() + store_ops_.size();

        mlir::ktdf::StageOp::create(
            builder, loc,
            /*depends_in=*/{},
            /*depends_out=*/{private_op.getResult(fifo_count + 0U)},
            [&](mlir::OpBuilder& builder, mlir::Location loc) {
              // Add data transfer operations in stage1 for loads
              createDataTransfers(builder, loc, private_op, tile_sizes_,
                                  /*is_load=*/true,
                                  /*private_result_offset=*/0, compute);
            });

        mlir::ktdf::StageOp::create(
            builder, loc,
            /*depends_in=*/{private_op.getResult(fifo_count + 0U)},
            /*depends_out=*/{private_op.getResult(fifo_count + 1U)},
            [&](mlir::OpBuilder& builder, mlir::Location loc) {
              // Create read_from_fifos, compute operations, write_to_fifos in
              // stage 2
              createComputeOps(builder, loc, private_op);
            })
            .setApplicableUnitsAttr(builder.getArrayAttr(compute.getKind()));

        mlir::ktdf::StageOp::create(
            builder, loc,
            /*depends_in=*/{private_op.getResult(fifo_count + 1U)},
            /*depends_out=*/{private_op.getResult(fifo_count + 2U)},
            [&](mlir::OpBuilder& builder, mlir::Location loc) {
              // Add data transfer operations in stage3 for stores
              createDataTransfers(builder, loc, private_op, tile_sizes_,
                                  /*is_load=*/false,
                                  /*private_result_offset=*/load_ops_.size(),
                                  compute);
            });
      });
  return mlir::success();
}

void ConstructThreeStagePipelinePass::deleteOpAndUnusedChainOfOperands(
    mlir::Operation* op) {
  // Collect operands before erasing
  llvm::SmallVector<mlir::Value> operands(op->getOperands().begin(),
                                          op->getOperands().end());
  op->erase();

  // Recursively delete unused operand operations
  for (mlir::Value operand : operands) {
    if (auto* def_op = operand.getDefiningOp()) {
      if (def_op->use_empty()) deleteOpAndUnusedChainOfOperands(def_op);
    }
  }
}

void ConstructThreeStagePipelinePass::cleanupOperations() {
  LDBG(1) << "Cleaning up operations";

  for (auto* op : ops_to_delete_) {
    if (!op || !op->use_empty()) continue;
    deleteOpAndUnusedChainOfOperands(op);
  }
}

// For a load, find the linalg input operand whose value flows from the load
// result (directly, through a tensor.extract_slice, through a
// tensor.collapse_shape, or through a tensor.expand_shape) and return its
// matching indexing_map expressed in the rank of the access tile.
//
// When a reshape (collapse/expand) sits between the load result and the linalg
// operand, the linalg input map has the reshaped rank, but the caller needs a
// map with getNumResults() == access tile rank.  We rebuild it by contracting
// or expanding each linalg result dim across its reassociation group:
//
//   collapse (many input dims → one linalg dim):
//     The access tile has the pre-collapse (higher) rank.  For each group the
//     single non-unit input dim inherits the linalg affine expression; size-1
//     input dims in the group get constant-0.
//
//   expand (one input dim → many linalg dims):
//     The access tile has the pre-expand (lower) rank.  We compress back: for
//     each reassociation group emit the expression of the unique non-unit
//     linalg dim in that group (all size-1 linalg dims contribute 0 and are
//     skipped).
//
// Returns std::nullopt if the linalg operand cannot be found.
static std::optional<mlir::AffineMap> findIndexingMapForLoadResult(
    mlir::linalg::LinalgOp linalg_op, mlir::Value tensor_value,
    llvm::ArrayRef<int64_t> access_tile_shape) {
  // Helper to find indexing map for a value used by linalg_op
  auto findMapForValue = [&](mlir::Value v) -> std::optional<mlir::AffineMap> {
    for (mlir::OpOperand& operand : linalg_op->getOpOperands()) {
      if (operand.get() == v) return linalg_op.getMatchingIndexingMap(&operand);
    }
    return std::nullopt;
  };

  if (auto map = findMapForValue(tensor_value)) return map;
  if (auto map = findMapThroughExtractSlice(tensor_value, findMapForValue))
    return map;

  auto [collapse_op, expand_op] = findReshapeOps(tensor_value);
  if (!collapse_op && !expand_op) return std::nullopt;

  mlir::Value reshaped =
      collapse_op ? collapse_op.getResult() : expand_op.getResult();

  std::optional<mlir::AffineMap> base_map = findMapForValue(reshaped);
  if (!base_map) {
    base_map = findMapThroughExtractSlice(reshaped, findMapForValue);
  }
  if (!base_map) return std::nullopt;

  llvm::SmallVector<mlir::ReassociationIndices> reassoc =
      getReassociationIndices(collapse_op, expand_op);
  mlir::MLIRContext* ctx = base_map->getContext();
  llvm::SmallVector<mlir::AffineExpr> results;
  results.reserve(access_tile_shape.size());

  if (collapse_op) {
    // collapse: access tile has the pre-collapse (higher) rank.
    // base_map has one result per reassociation group (== one per output dim).
    // Expand back: for each group emit one result per input dim; size-1 input
    // dims → constant 0; the non-unit input dim → group's linalg expression.
    auto src_shaped =
        mlir::cast<mlir::ShapedType>(collapse_op.getSrc().getType());
    for (auto [group_idx, group] : llvm::enumerate(reassoc)) {
      mlir::AffineExpr group_expr = base_map->getResult(group_idx);
      for (int64_t in_dim : group) {
        if (src_shaped.getDimSize(in_dim) == 1)
          results.push_back(mlir::getAffineConstantExpr(0, ctx));
        else
          results.push_back(group_expr);
      }
    }
  } else {
    // expand: access tile has the pre-expand (lower) rank.
    // base_map has one result per linalg dim (post-expand); groups expand one
    // input dim to N linalg dims.  Compress back: for each group pick the
    // expression of the unique non-unit linalg dim.
    auto dst_shaped =
        mlir::cast<mlir::ShapedType>(expand_op.getResult().getType());
    for (auto& group : reassoc) {
      mlir::AffineExpr chosen = mlir::getAffineConstantExpr(0, ctx);
      for (int64_t out_dim : group) {
        if (dst_shaped.getDimSize(out_dim) != 1)
          chosen = base_map->getResult(out_dim);
      }
      results.push_back(chosen);
    }
  }

  return mlir::AffineMap::get(base_map->getNumDims(), /*symbolCount=*/0,
                              results, ctx);
}

// For a store, find the linalg DPS init operand whose value flows into the
// store's tensor input (directly or through a tensor.insert_slice,
// tensor.expand_shape, or tensor.collapse_shape) and return its matching
// indexing_map expressed in the rank of the access tile.
//
// When a reshape (expand/collapse) sits between the linalg result and the
// store, the linalg init map has the linalg result rank, but the caller needs
// a map with getNumResults() == access tile rank.  We rebuild it by expanding
// each linalg result dim across its reassociation group: size-1 output dims in
// the group become constant-0 results; the single non-unit dim inherits the
// linalg affine expression for that group.
//
// Returns std::nullopt if the linalg result cannot be found.
static std::optional<mlir::AffineMap> findIndexingMapForStoreSource(
    mlir::linalg::LinalgOp linalg_op, mlir::Value tensor_value,
    llvm::ArrayRef<int64_t> access_tile_shape) {
  mlir::Value v = tensor_value;

  mlir::tensor::ExpandShapeOp expand_op;
  mlir::tensor::CollapseShapeOp collapse_op;

  if (auto insert = v.getDefiningOp<mlir::tensor::InsertSliceOp>())
    v = insert.getSource();
  if (auto expand = v.getDefiningOp<mlir::tensor::ExpandShapeOp>()) {
    expand_op = expand;
    v = expand.getSrc();
  } else if (auto collapse = v.getDefiningOp<mlir::tensor::CollapseShapeOp>()) {
    collapse_op = collapse;
    v = collapse.getSrc();
  }
  if (auto insert = v.getDefiningOp<mlir::tensor::InsertSliceOp>())
    v = insert.getSource();

  for (int64_t i = 0, e = linalg_op->getNumResults(); i < e; ++i) {
    if (linalg_op->getResult(i) != v) continue;

    mlir::OpOperand* init = linalg_op.getDpsInitOperand(i);
    mlir::AffineMap base_map = linalg_op.getMatchingIndexingMap(init);

    if (!expand_op && !collapse_op) return base_map;

    llvm::SmallVector<mlir::ReassociationIndices> reassoc =
        getReassociationIndices(collapse_op, expand_op);
    // For expand: base_map has one result per reassociation group; each group
    // expands to N output dims.  Size-1 output dims → constant 0; the unique
    // non-unit output dim → the group's affine expression.
    // For collapse: base_map has one result per output dim; each group of
    // input dims collapses to one.  Size-1 input dims → skip (they contribute
    // constant 0 as a linalg result too); the non-unit input dim → keeps the
    // group's affine expression.
    mlir::MLIRContext* ctx = base_map.getContext();
    llvm::SmallVector<mlir::AffineExpr> results;
    results.reserve(access_tile_shape.size());

    if (expand_op) {
      // base_map result index == reassociation group index.
      for (auto [group_idx, group] : llvm::enumerate(reassoc)) {
        mlir::AffineExpr group_expr = base_map.getResult(group_idx);
        for (int64_t out_dim : group) {
          // A size-1 output dim in the expansion is pinned to 0.
          if (access_tile_shape[out_dim] == 1)
            results.push_back(mlir::getAffineConstantExpr(0, ctx));
          else
            results.push_back(group_expr);
        }
      }
    } else {
      // collapse_op: access_tile_shape is the collapsed (output) shape.
      // base_map has one result per input dim; groups collapse N inputs → 1.
      // We emit one result per output dim (== one per group), using the
      // non-unit input dim's expression from base_map.
      auto src_shaped =
          mlir::cast<mlir::ShapedType>(collapse_op.getSrc().getType());
      for (auto [group_idx, group] : llvm::enumerate(reassoc)) {
        mlir::AffineExpr chosen = mlir::getAffineConstantExpr(0, ctx);
        for (int64_t in_dim : group) {
          if (src_shaped.getDimSize(in_dim) != 1)
            chosen = base_map.getResult(in_dim);
        }
        results.push_back(chosen);
      }
    }

    return mlir::AffineMap::get(base_map.getNumDims(), /*symbolCount=*/0,
                                results, ctx);
  }
  return std::nullopt;
}

// Drop the dims of `map` that none of its results reference, appending to
// `surviving` the operands — one per dim of `map`, in dim order — for the dims
// that survived.  compressUnusedDims preserves the relative order of the dims
// it keeps, so appending in dim order matches the new numbering.
//
// Operands for dims that cannot survive (e.g. a dim already substituted with a
// constant) are never read, so a null Value is a legal placeholder for them.
static mlir::AffineMap compressMapAndOperands(
    mlir::AffineMap map, llvm::ArrayRef<mlir::Value> operands,
    llvm::SmallVectorImpl<mlir::Value>& surviving) {
  assert(operands.size() == map.getNumDims());

  llvm::SmallBitVector used(map.getNumDims(), false);
  for (mlir::AffineExpr result : map.getResults()) {
    result.walk([&](mlir::AffineExpr expr) {
      if (auto dim = mlir::dyn_cast<mlir::AffineDimExpr>(expr))
        used.set(dim.getPosition());
    });
  }

  for (unsigned d = 0, e = map.getNumDims(); d < e; ++d) {
    if (used.test(d)) surviving.push_back(operands[d]);
  }

  return mlir::compressUnusedDims(map);
}

// Return a compressed version of indexing_map suitable for use in a
// data_transfer op, together with the filtered set of loop IVs to pass as
// its dim operands.
//
// Two kinds of dims are removed from the domain:
//   1. Reduction dims (tile_sizes[k] == 0): substituted with constant 0,
//      then dropped because they are no longer referenced.
//   2. Parallel dims that do not appear in any result of this particular
//      operand's indexing map (e.g. a broadcast input that skips a loop dim):
//      also dropped by compressUnusedDims.
//
// loop_ivs must have one entry per tiled dim (tile_sizes[k] > 0), in
// loop-index order.  The returned map_ivs contains only the subset of
// loop_ivs that correspond to dims surviving compression.
static mlir::AffineMap buildDataTransferMap(
    mlir::AffineMap indexing_map, llvm::ArrayRef<int64_t> tile_sizes,
    llvm::ArrayRef<mlir::Value> loop_ivs,
    llvm::SmallVectorImpl<mlir::Value>& map_ivs) {
  assert(indexing_map.getNumSymbols() == 0);
  assert(indexing_map.getNumDims() == tile_sizes.size());

  mlir::MLIRContext* ctx = indexing_map.getContext();

  // Substitute reduction dims with constant 0 — they have no IV — and pair each
  // dim with the IV that drives it.
  llvm::SmallVector<mlir::AffineExpr> replacements;
  llvm::SmallVector<mlir::Value> operands;
  replacements.reserve(tile_sizes.size());
  operands.reserve(tile_sizes.size());
  for (unsigned k = 0, iv = 0; k < tile_sizes.size(); ++k) {
    if (tile_sizes[k] > 0) {
      replacements.push_back(mlir::getAffineDimExpr(k, ctx));
      operands.push_back(loop_ivs[iv++]);
    } else {
      replacements.push_back(mlir::getAffineConstantExpr(0, ctx));
      operands.push_back(nullptr);
    }
  }
  mlir::AffineMap substituted = indexing_map.replaceDimsAndSymbols(
      replacements, {}, indexing_map.getNumDims(), 0);

  return compressMapAndOperands(substituted, operands, map_ivs);
}

mlir::ktdf::IndDataTransferOp
ConstructThreeStagePipelinePass::createIndDataTransfer(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ktdp_lowering::ConstructIndirectAccessTileOp ind_tile,
    mlir::AffineMap indexing_map, llvm::ArrayRef<int64_t> tile_sizes,
    llvm::ArrayRef<mlir::Value> loop_ivs,
    llvm::ArrayRef<int64_t> variables_extents,
    mlir::TypedValue<mlir::ktdf::FifoSlotType> fifo_slot,
    llvm::ArrayRef<int64_t> fifo_sizes, bool is_load) {
  auto* ctx = builder.getContext();

  auto iab_memref = llvm::dyn_cast<mlir::MemRefType>(
      ind_tile.getIndAddrBufMemref().getType());
  if (!iab_memref || iab_memref.getRank() != 1) {
    ind_tile.emitError(
        "indirect address buffer must be rank 1 for ktdf.ind_data_transfer, "
        "which addresses a single entry; got rank ")
        << (iab_memref ? iab_memref.getRank() : -1);
    return nullptr;
  }

  mlir::ValueRange captured = ind_tile.getCapturedVariables();
  mlir::ValueRange intermediates = ind_tile.getIntermediateVariables();
  const unsigned num_captured = captured.size();
  const unsigned num_intermediates = intermediates.size();
  const unsigned num_loops = tile_sizes.size();
  assert(indexing_map.getNumDims() == num_loops &&
         "operand indexing map must be over the linalg loop domain");

  // The buffer subscript must name a captured value; an intermediate variable
  // would again mean one transfer per window rather than per entry.
  const auto iab_position =
      static_cast<unsigned>(ind_tile.getIndAddrBufDimPositions().front());
  if (iab_position >= num_captured) {
    ind_tile.emitError(
        "indirect address buffer subscript is an intermediate variable; "
        "ktdf.ind_data_transfer addresses a single entry, so the subscript "
        "must be a captured value");
    return nullptr;
  }
  mlir::Value iab_index = captured[iab_position];

  // variables_space_order names the intermediate variable driving each
  // dimension of the variables space, and the Linalg indexing map names the
  // loop dim driving each dimension of the operand.  Compose them to get, for
  // each intermediate variable, the loop dim it now ranges over and the extent
  // of its variables-space dimension.
  mlir::AffineMap order = ind_tile.getVariablesSpaceOrder();
  if (order.getNumDims() != num_intermediates ||
      order.getNumResults() != variables_extents.size() ||
      indexing_map.getNumResults() != variables_extents.size()) {
    ind_tile.emitError("variables_space_order maps ")
        << order.getNumDims() << " intermediate variable(s) to "
        << order.getNumResults() << " dimension(s), but there are "
        << num_intermediates << " intermediate variable(s), "
        << variables_extents.size()
        << " variables_space_set dimension(s) and a Linalg operand of rank "
        << indexing_map.getNumResults();
    return nullptr;
  }

  llvm::SmallVector<int64_t> var_loop_dim(num_intermediates, -1);
  llvm::SmallVector<int64_t> var_extent(num_intermediates, 0);
  for (unsigned p = 0; p < order.getNumResults(); ++p) {
    auto var_expr = mlir::dyn_cast<mlir::AffineDimExpr>(order.getResult(p));
    auto loop_expr =
        mlir::dyn_cast<mlir::AffineDimExpr>(indexing_map.getResult(p));
    if (!var_expr || !loop_expr) {
      ind_tile.emitError(
          "intermediate variables must project one-to-one onto Linalg loop "
          "dimensions; variables-space dimension ")
          << p << " does not";
      return nullptr;
    }
    const unsigned var = var_expr.getPosition();
    if (var_loop_dim[var] != -1) {
      ind_tile.emitError(
          "variables_space_order is not a permutation: intermediate variable ")
          << var << " drives more than one variables-space dimension";
      return nullptr;
    }
    var_loop_dim[var] = loop_expr.getPosition();
    var_extent[var] = variables_extents[p];
  }
  for (unsigned var = 0; var < num_intermediates; ++var) {
    if (var_loop_dim[var] == -1) {
      ind_tile.emitError("intermediate variable ")
          << var << " is not projected onto any variables-space dimension";
      return nullptr;
    }
  }

  // Rebase the subscript maps from the op's unified domain
  // (captured..., intermediate...) onto the transfer's domain
  // (loop dims..., captured...).  An intermediate variable becomes the tiled
  // loop IV that now enumerates it, or 0 when its dim is an untiled reduction
  // dim and the transfer therefore starts at the origin.
  llvm::SmallVector<mlir::AffineExpr> replacements;
  replacements.reserve(num_captured + num_intermediates);
  for (unsigned c = 0; c < num_captured; ++c) {
    replacements.push_back(mlir::getAffineDimExpr(num_loops + c, ctx));
  }
  for (unsigned var = 0; var < num_intermediates; ++var) {
    const auto loop_dim = static_cast<unsigned>(var_loop_dim[var]);
    replacements.push_back(tile_sizes[loop_dim] > 0
                               ? mlir::getAffineDimExpr(loop_dim, ctx)
                               : mlir::getAffineConstantExpr(0, ctx));
  }

  llvm::SmallVector<mlir::AffineExpr> dir_exprs;
  llvm::SmallVector<int64_t> dir_sizes;
  dir_exprs.reserve(ind_tile.getPerDimSubscriptMaps().size());
  dir_sizes.reserve(ind_tile.getPerDimSubscriptMaps().size());
  for (auto [dim, attr] : llvm::enumerate(ind_tile.getPerDimSubscriptMaps())) {
    mlir::AffineMap subscript =
        llvm::cast<mlir::AffineMapAttr>(attr).getValue();
    if (subscript.getNumResults() != 1) {
      ind_tile.emitError("per_dim_subscript_maps[")
          << dim << "] must have exactly one result, got "
          << subscript.getNumResults();
      return nullptr;
    }

    // Which intermediate variable, if any, this base dimension ranges over.
    // Two would make a single contiguous extent meaningless.
    std::optional<unsigned> var;
    bool ambiguous = false;
    subscript.getResult(0).walk([&](mlir::AffineExpr expr) {
      auto dim_expr = mlir::dyn_cast<mlir::AffineDimExpr>(expr);
      if (!dim_expr || dim_expr.getPosition() < num_captured) return;
      const unsigned candidate = dim_expr.getPosition() - num_captured;
      if (var && *var != candidate) ambiguous = true;
      var = candidate;
    });
    if (ambiguous) {
      ind_tile.emitError("per_dim_subscript_maps[")
          << dim
          << "] references more than one intermediate variable, so this base "
             "dimension has no single transfer extent";
      return nullptr;
    }

    dir_exprs.push_back(subscript
                            .replaceDimsAndSymbols(replacements, {},
                                                   num_loops + num_captured, 0)
                            .getResult(0));

    // A dimension the variables space does not range over is pinned to one
    // element.  Otherwise the tile covers tile_sizes[k] of it, falling back to
    // the whole variables-space extent for an untiled reduction dim.
    if (!var) {
      dir_sizes.push_back(1);
    } else {
      const auto loop_dim = static_cast<unsigned>(var_loop_dim[*var]);
      dir_sizes.push_back(tile_sizes[loop_dim] > 0 ? tile_sizes[loop_dim]
                                                   : var_extent[*var]);
    }
  }

  // Domain values for (loop dims..., captured...).  Reduction dims have no IV,
  // but they were substituted with a constant so they never survive.
  llvm::SmallVector<mlir::Value> domain_values(num_loops + num_captured);
  for (unsigned k = 0, iv = 0; k < num_loops; ++k) {
    if (tile_sizes[k] > 0) domain_values[k] = loop_ivs[iv++];
  }
  for (unsigned c = 0; c < num_captured; ++c) {
    domain_values[num_loops + c] = captured[c];
  }

  llvm::SmallVector<mlir::Value> dir_indices;
  mlir::AffineMap dir_map = compressMapAndOperands(
      mlir::AffineMap::get(num_loops + num_captured, 0, dir_exprs, ctx),
      domain_values, dir_indices);

  // The base and buffer operands are the pre-cast values; step 5 rewires them
  // to the reinterpret_casts it creates above the outermost tiled loop, which
  // cannot exist yet.
  mlir::Value base = ind_tile.getResult();
  mlir::Value iab = ind_tile.getIndAddrBufMemref();

  auto transfer =
      is_load ? mlir::ktdf::IndDataTransferOp::create(
                    builder, loc, /*ind_src_memref=*/iab,
                    /*ind_src_index=*/iab_index,
                    /*dir_src=*/base, dir_map, dir_indices, dir_sizes,
                    /*ind_dst_memref=*/nullptr, /*ind_dst_index=*/nullptr,
                    /*dir_dst=*/fifo_slot, /*dir_dst_map=*/mlir::AffineMap{},
                    mlir::ValueRange{}, fifo_sizes)
              : mlir::ktdf::IndDataTransferOp::create(
                    builder, loc, /*ind_src_memref=*/nullptr,
                    /*ind_src_index=*/nullptr, /*dir_src=*/fifo_slot,
                    /*dir_src_map=*/mlir::AffineMap{}, mlir::ValueRange{},
                    fifo_sizes, /*ind_dst_memref=*/iab,
                    /*ind_dst_index=*/iab_index, /*dir_dst=*/base, dir_map,
                    dir_indices, dir_sizes);

  ind_transfers_.emplace_back(transfer, ind_tile);
  return transfer;
}

void ConstructThreeStagePipelinePass::createIndAddrBufFill(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ktdf_arch::ExecutionUnitOp compute) {
  const IndAddrBufFill& fill = *ind_addr_buf_fill_;

  mlir::OpBuilder::InsertionGuard insertion_guard(builder);

  // Move the guard rather than rebuild it: that preserves its condition
  // verbatim, needs no clone of the arith.cmpi, and leaves nothing behind.  It
  // has to land in this stage — the buffer is local state of the unit running
  // the stage and does not survive a stage boundary.
  if (mlir::scf::IfOp guard = fill.guard) {
    guard->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
    builder.setInsertionPoint(guard.getThenRegion().front().getTerminator());
  }

  mlir::Value source = fill.source_tile;
  mlir::Value dest = fill.dest_tile;
  llvm::ArrayRef<int64_t> source_sizes = transferShape(
      llvm::cast<mlir::ktdp::AccessTileType>(source.getType()).getShape(),
      /*collapse_leading_units=*/true);
  llvm::ArrayRef<int64_t> dest_sizes = transferShape(
      llvm::cast<mlir::ktdp::AccessTileType>(dest.getType()).getShape(),
      /*collapse_leading_units=*/true);

  // Both sides address their origin: the row offset is folded into the
  // reinterpret_cast step 5 replaces each access tile with.
  mlir::MLIRContext* ctx = &getContext();
  auto transfer = mlir::ktdf::DataTransferOp::create(
      builder, loc, source, originMap(source_sizes.size(), ctx),
      mlir::ValueRange{}, source_sizes, dest, originMap(dest_sizes.size(), ctx),
      mlir::ValueRange{}, dest_sizes);

  transfer->setDiscardableAttr(
      kThrottleAttrName,
      builder.getI64IntegerAttr(
          getThrottle(llvm::cast<mlir::ktdp::AccessTileType>(source.getType())
                          .getElementType(),
                      compute)));
}

[[nodiscard]] llvm::SmallVector<mlir::Value>
ConstructThreeStagePipelinePass::collectLoopInductionVars() const {
  llvm::SmallVector<mlir::Value> loop_ivs;
  for (auto* loop_op : tiled_loops_) {
    auto for_op = llvm::cast<mlir::scf::ForOp>(loop_op);
    loop_ivs.push_back(for_op.getInductionVar());
  }
  return loop_ivs;
}

void ConstructThreeStagePipelinePass::createDataTransfers(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ktdf::PrivateOp private_op, llvm::ArrayRef<int64_t> tile_sizes,
    bool is_load, size_t private_result_offset,
    mlir::ktdf_arch::ExecutionUnitOp compute) {
  llvm::SmallVector<mlir::Value> loop_ivs = collectLoopInductionVars();

  LLVM_DEBUG({
    llvm::dbgs() << "  Creating " << (is_load ? "load" : "store")
                 << " data transfers with " << loop_ivs.size() << " loop IVs\n";
  });

  // Get the appropriate operation list
  size_t op_count = is_load ? load_ops_.size() : store_ops_.size();

  // The post-fusion linalg op whose indexing_maps describe how each
  // load/store operand maps onto the iteration space.
  assert(compute_ops_.size() == 1 &&
         "expected exactly one linalg compute op after fusion");
  mlir::linalg::LinalgOp linalg_op = compute_ops_[0];

  // The buffer fill belongs to the stage that dereferences it — stage 1 for a
  // gather, stage 3 for a scatter — and must precede the indirect transfer.
  if (ind_addr_buf_fill_ && ind_tile_is_load_ == is_load) {
    createIndAddrBufFill(builder, loc, compute);
  }

  // Create data_transfer for each operation
  for (size_t i = 0; i < op_count; ++i) {
    // Get the access_tile value (operand of load/store op) and the linalg
    // indexing map for the matching operand.
    mlir::Value access_tile_value;
    std::optional<mlir::AffineMap> indexing_map;
    mlir::Operation* err_anchor;
    if (is_load) {
      mlir::ktdp::LoadOp load_op = load_ops_[i];
      access_tile_value = load_op.getAccessTile();
      err_anchor = load_op.getOperation();
      llvm::ArrayRef<int64_t> tile_shape;
      if (auto at_type = mlir::dyn_cast<mlir::ktdp::AccessTileType>(
              access_tile_value.getType()))
        tile_shape = at_type.getShape();
      indexing_map = findIndexingMapForLoadResult(
          linalg_op, load_op.getResult(), tile_shape);
    } else {
      mlir::ktdp::StoreOp store_op = store_ops_[i];
      access_tile_value = store_op.getAccessTile();
      err_anchor = store_op.getOperation();
      llvm::ArrayRef<int64_t> tile_shape;
      if (auto at_type = mlir::dyn_cast<mlir::ktdp::AccessTileType>(
              access_tile_value.getType()))
        tile_shape = at_type.getShape();
      indexing_map = findIndexingMapForStoreSource(
          linalg_op, store_op.getDataTile(), tile_shape);
    }

    if (!indexing_map) {
      err_anchor->emitError(
          "could not locate matching linalg operand to project loop IVs and "
          "tile sizes through; the data_transfer rank would not match the "
          "underlying memref");
      signalPassFailure();
      return;
    }

    auto access_tile_type =
        mlir::dyn_cast<mlir::ktdp::AccessTileType>(access_tile_value.getType());
    if (!access_tile_type) {
      err_anchor->emitError("access tile operand is not an AccessTileType");
      signalPassFailure();
      return;
    }
    llvm::ArrayRef<int64_t> full_shape = access_tile_type.getShape();

    // For an indirect tile the extents come from variables_space_set rather
    // than from the result shape: the two are not guaranteed to agree (pinned
    // dims, variables_space_order, a result type carried over from the ktdp
    // variant), and the set is the op's own statement of the tile shape.
    auto ind_tile = access_tile_value.getDefiningOp<
        mlir::ktdp_lowering::ConstructIndirectAccessTileOp>();
    llvm::SmallVector<int64_t> variables_extents;
    if (ind_tile) {
      auto extents = scheduler::getSizesFromIntegerSet(
          ind_tile.getVariablesSpaceSet().getValue());
      if (mlir::failed(extents)) {
        ind_tile.emitError(
            "variables_space_set is not in box form, so the indirect transfer "
            "extents cannot be derived from it");
        signalPassFailure();
        return;
      }
      variables_extents = std::move(*extents);
      full_shape = variables_extents;
    }

    // The size loop below indexes full_shape by result position, so the operand
    // rank the indexing map describes has to match the extents we project
    // through it.
    if (indexing_map->getNumResults() != full_shape.size()) {
      err_anchor->emitError("linalg operand has rank ")
          << indexing_map->getNumResults() << " but "
          << (ind_tile ? "variables_space_set describes "
                       : "its access tile has ")
          << full_shape.size()
          << " dimension(s); the data_transfer rank would not match the "
             "underlying memref";
      signalPassFailure();
      return;
    }

    llvm::SmallVector<int64_t> access_tile_sizes;
    access_tile_sizes.reserve(indexing_map->getNumResults());
    for (mlir::AffineExpr result_expr : indexing_map->getResults()) {
      if (auto dim_expr = mlir::dyn_cast<mlir::AffineDimExpr>(result_expr)) {
        unsigned k = dim_expr.getPosition();
        // Parallel dim: use the tile size.
        // Reduction dim (tile_sizes[k] == 0): use the full dim extent.
        int64_t ts = tile_sizes[k];
        access_tile_sizes.push_back(
            ts > 0 ? ts : full_shape[access_tile_sizes.size()]);
      } else if (mlir::isa<mlir::AffineConstantExpr>(result_expr)) {
        // Broadcast/squeezed dim: size is 1.
        access_tile_sizes.push_back(1);
      } else {
        err_anchor->emitError(
            "unsupported affine expression in operand indexing map for "
            "data_transfer size computation");
        signalPassFailure();
        return;
      }
    }

    // Get the FIFO slot from ktdf.private results.
    const auto fifo_slot =
        llvm::cast<mlir::TypedValue<mlir::ktdf::FifoSlotType>>(
            private_op.getResult(private_result_offset + i));

    // For a load, the FIFO must hold the full input tile — including any
    // reduction dimensions — because the compute stage reads a tensor of that
    // full shape from the FIFO.  fifo_sizes therefore mirrors
    // access_tile_sizes.
    //
    // For a store, the output tile contains only parallel dimensions, so
    // reduction dims (tile_sizes[k] == 0) are absent from the output indexing
    // map and must not appear in fifo_sizes.
    llvm::SmallVector<int64_t> fifo_sizes;
    if (is_load) {
      fifo_sizes.assign(access_tile_sizes.begin(), access_tile_sizes.end());
    } else {
      for (mlir::AffineExpr result_expr : indexing_map->getResults()) {
        if (auto dim_expr = mlir::dyn_cast<mlir::AffineDimExpr>(result_expr)) {
          unsigned k = dim_expr.getPosition();
          if (tile_sizes[k] > 0) fifo_sizes.push_back(tile_sizes[k]);
        } else if (mlir::isa<mlir::AffineConstantExpr>(result_expr)) {
          fifo_sizes.push_back(1);
        }
      }
    }

    // The fifo side gets a null AffineMap; the access-tile (memref) side gets
    // the substituted+compressed map. map_ivs has exactly one entry per dim
    // that survived compression.
    mlir::Operation* transfer = nullptr;
    if (ind_tile) {
      transfer = createIndDataTransfer(builder, loc, ind_tile, *indexing_map,
                                       tile_sizes, loop_ivs, variables_extents,
                                       fifo_slot, fifo_sizes, is_load);
      if (!transfer) {
        signalPassFailure();
        return;
      }
    } else {
      // Build the data_transfer map and the matching IV list.
      llvm::SmallVector<mlir::Value> map_ivs;
      mlir::AffineMap substituted_map =
          buildDataTransferMap(*indexing_map, tile_sizes, loop_ivs, map_ivs);

      if (is_load) {
        transfer = mlir::ktdf::DataTransferOp::create(
            builder, loc, access_tile_value, substituted_map, map_ivs,
            access_tile_sizes, fifo_slot, {}, {}, fifo_sizes);
      } else {
        transfer = mlir::ktdf::DataTransferOp::create(
            builder, loc, fifo_slot, {}, {}, fifo_sizes, access_tile_value,
            substituted_map, map_ivs, access_tile_sizes);
      }
    }

    transfer->setDiscardableAttr(
        kThrottleAttrName, builder.getI64IntegerAttr(getThrottle(
                               fifo_slot.getType().getElementType(), compute)));
  }
}

}  // namespace

void ConstructThreeStagePipelinePass::createComputeOps(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::ktdf::PrivateOp private_op) {
  LDBG(1) << "Creating compute operations in stage 2";

  if (compute_ops_.size() != 1) {
    mlir::emitError(loc, "Expected exactly one compute op after fusion, got ")
        << compute_ops_.size();
    signalPassFailure();
    return;
  }

  // Get the tiled tensor type from the compute operation
  auto tiled_tensor_type = mlir::dyn_cast<mlir::RankedTensorType>(
      compute_ops_[0]->getResult(0).getType());
  if (!tiled_tensor_type) {
    compute_ops_[0]->emitError(
        "Expected RankedTensorType result from linalg operation");
    signalPassFailure();
    return;
  }

  // For each load op, find the chain that leads from the load result into the
  // tiled linalg op.  The chain may be any of:
  //   (A) direct:          load → extract_slice → linalg
  //   (B) reshape-direct:  load → reshape → linalg          (no tiling)
  //   (C) reshape-tiled:   load → reshape → extract_slice → linalg
  //
  // For each load we record:
  //   - reshape_op:  the collapse/expand between load and extract_slice (null
  //                  when the load feeds extract_slice or linalg directly)
  //   - extract_op:  the extract_slice that determines the per-tile type (null
  //                  when tiling did not introduce one, i.e. case B)
  //
  // The FIFO slot type (sized in createPrivateOp from the tiled linalg input
  // type) must match what read_from_fifo produces:
  //   - Cases A/C: FIFO type = extract_slice result type; map extract_slice
  //                result → read_from_fifo result; clone reshape (if any) into
  //                the stage so the cloned linalg can consume it.
  //   - Case B:    No extract_slice; FIFO type = reshape result type (which
  //                equals the linalg input type); map reshape result →
  //                read_from_fifo result.
  struct LoadChain {
    mlir::Operation* reshape_op = nullptr;    // collapse/expand, or null
    mlir::tensor::ExtractSliceOp extract_op;  // may be null (case B)
  };
  llvm::SmallVector<LoadChain> load_chains(load_ops_.size());

  for (size_t i = 0; i < load_ops_.size(); ++i) {
    auto v = load_ops_[i].getResult();
    auto& chain = load_chains[i];

    for (auto* user : v.getUsers()) {
      if (auto ex = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user)) {
        chain.extract_op = ex;
        break;
      }
    }
    if (chain.extract_op) continue;

    for (auto* user : v.getUsers()) {
      if (mlir::isa<mlir::tensor::CollapseShapeOp, mlir::tensor::ExpandShapeOp>(
              user)) {
        chain.reshape_op = user;
        break;
      }
    }
    if (!chain.reshape_op) continue;

    auto reshaped = chain.reshape_op->getResult(0);
    for (auto* user : reshaped.getUsers()) {
      if (auto ex = mlir::dyn_cast<mlir::tensor::ExtractSliceOp>(user)) {
        chain.extract_op = ex;
        break;
      }
    }
    // If still no extract_slice we are in case B — reshape feeds linalg
    // directly; extract_op remains null.
  }

  // Build a mapping from load results to read_from_fifo results, using the
  // per-operand tile type so the cloned linalg op's input ranks match its
  // indexing maps.
  mlir::DenseMap<mlir::Value, mlir::Value> value_map;
  mlir::IRMapping mapper;
  for (size_t i = 0; i < load_ops_.size(); ++i) {
    auto load_op = load_ops_[i];
    auto fifo_slot = private_op.getResult(i);
    auto& chain = load_chains[i];

    if (!chain.reshape_op && !chain.extract_op) {
      load_op.emitError(
          "no tensor.extract_slice or reshape user found; cannot determine "
          "the per-operand tile type for ktdf.read_from_fifo");
      signalPassFailure();
      return;
    }

    if (chain.extract_op) {
      // Cases A and C: FIFO carries the tiled (post-extract_slice) tensor.
      // In case C the tiling infra slices the collapsed tensor, so the tiled
      // linalg input references the extract_slice result directly — no reshape
      // clone is needed inside the stage.
      mlir::Type per_tile_type = chain.extract_op.getResult().getType();
      auto read_fifo_op = mlir::ktdf::ReadFromFifoOp::create(
          builder, loc, per_tile_type, fifo_slot);
      value_map[load_op.getResult()] = read_fifo_op.getResult();
      mapper.map(chain.extract_op.getResult(), read_fifo_op.getResult());
    } else {
      // Case B: no tiling; reshape feeds linalg directly.
      // FIFO carries the reshaped tensor type (== linalg input type).
      mlir::Value reshaped = chain.reshape_op->getResult(0);
      mlir::Type reshaped_type = reshaped.getType();
      auto read_fifo_op = mlir::ktdf::ReadFromFifoOp::create(
          builder, loc, reshaped_type, fifo_slot);
      value_map[load_op.getResult()] = read_fifo_op.getResult();
      // Map the reshape result → read_from_fifo so the cloned linalg sees
      // the correct value without needing the reshape inside the stage.
      mapper.map(reshaped, read_fifo_op.getResult());
    }
  }

  // Clone compute operation into stage 2 and create write_to_fifo for each
  // store
  mlir::linalg::LinalgOp compute_op = compute_ops_[0];
  LDBG(1) << "  Cloning compute op: " << compute_op->getName() << "";

  // A dummy tensor.empty per output operand, with the tiled tensor type. One
  // each: a compute accumulating a pair has an init per half, and mapping only
  // the first would leave the clone reading the other from outside the stage.
  auto linalg_op =
      mlir::dyn_cast<mlir::linalg::LinalgOp>(compute_op.getOperation());
  if (linalg_op) {
    for (auto& init : linalg_op.getDpsInitsMutable()) {
      auto empty_tensor = mlir::tensor::EmptyOp::create(
          builder, loc, tiled_tensor_type.getShape(),
          tiled_tensor_type.getElementType());
      mapper.map(init.get(), empty_tensor.getResult());
    }
  }

  // Any linalg op input not already remapped
  // that is defined by a tensor.extract_slice in the enclosing loop body
  // cannot be used inside the stage nested region (would result in use before
  // definition). Clone the extract_slice inside the stage so the cloned linalg
  // can reference it legally.
  if (linalg_op) {
    for (auto* input : linalg_op.getDpsInputOperands()) {
      auto v = input->get();
      if (mapper.contains(v)) continue;
      auto extract_op = v.getDefiningOp<mlir::tensor::ExtractSliceOp>();
      if (!extract_op) continue;
      // Clone the extract_slice at the current insertion point (inside the
      // stage), mapping its operands through the existing mapper so that any
      // loop IVs etc. resolve correctly.
      auto* cloned_extract = builder.clone(*extract_op, mapper);
      mapper.map(v, cloned_extract->getResult(0));
    }
  }

  auto* cloned = builder.clone(*compute_op, mapper);

  for (size_t i = 0; i < store_ops_.size(); ++i) {
    auto fifo_slot = private_op.getResult(load_ops_.size() + i);
    auto stored = mlir::dyn_cast<mlir::OpResult>(store_ops_[i].getDataTile());
    unsigned which = stored ? stored.getResultNumber() : 0;
    mlir::ktdf::WriteToFifoOp::create(builder, loc, cloned->getResult(which),
                                      fifo_slot);
  }
}

mlir::OpFoldResult
ConstructThreeStagePipelinePass::computeReinterpretCastOffset(
    mlir::OpBuilder& builder, mlir::Location loc,
    llvm::SmallVector<mlir::Value>& indices,
    llvm::SmallVector<int64_t>& strides) {
  // Calculate offset from access tile indices and memory view strides.
  // For an access tile %A_view[%idx0, %idx1, ...] with strides [stride0,
  // stride1, ...], the offset is: %idx0 * stride0 + %idx1 * stride1 + ...
  // This is computed using a sequence of arith.muli and arith.addi operations.

  if (indices.empty()) return builder.getIndexAttr(0);

  llvm::SmallVector<mlir::Value> terms;
  for (unsigned i = 0; i < indices.size(); ++i) {
    if (mlir::isZeroInteger(indices[i])) continue;

    mlir::Value term;
    if (strides[i] == 1) {
      term = indices[i];
    } else {
      mlir::Value stride_const =
          mlir::arith::ConstantIndexOp::create(builder, loc, strides[i]);
      term =
          mlir::arith::MulIOp::create(builder, loc, indices[i], stride_const);
    }
    terms.push_back(term);
  }

  if (terms.empty()) return builder.getIndexAttr(0);
  if (terms.size() == 1) return terms[0];

  mlir::Value offset = terms[0];
  for (size_t i = 1; i < terms.size(); ++i) {
    offset = mlir::arith::AddIOp::create(builder, loc, offset, terms[i]);
  }
  return offset;
}

[[nodiscard]] static auto getMemRefStrides(
    mlir::TypedValue<mlir::MemRefType> memory_view)
    -> llvm::SmallVector<int64_t> {
  if (auto opt_strides = getStrides(memory_view)) {
    return *opt_strides;
  }

  llvm::SmallVector<int64_t> strides;
  if (auto strided_layout = mlir::dyn_cast<mlir::StridedLayoutAttr>(
          memory_view.getType().getLayout())) {
    strides.assign(strided_layout.getStrides().begin(),
                   strided_layout.getStrides().end());
  } else {
    int64_t stride = 1;
    for (int i = memory_view.getType().getRank() - 1; i >= 0; --i) {
      strides.insert(strides.begin(), stride);
      stride *= memory_view.getType().getShape()[i];
    }
  }
  return strides;
}

mlir::Value ConstructThreeStagePipelinePass::emitReinterpretCast(
    mlir::OpBuilder& builder, mlir::Location loc,
    mlir::TypedValue<mlir::MemRefType> memory_view,
    llvm::SmallVector<mlir::Value>& per_dim_indices,
    llvm::ArrayRef<int64_t> tile_dims, mlir::Operation* err_anchor) {
  llvm::SmallVector<int64_t> strides = getMemRefStrides(memory_view);

  if (!per_dim_indices.empty() && per_dim_indices.size() != strides.size()) {
    err_anchor->emitError("Number of indices (")
        << per_dim_indices.size() << ") does not match number of strides ("
        << strides.size() << ")";
    return nullptr;
  }
  if (tile_dims.size() > strides.size()) {
    err_anchor->emitError("Access tile rank (")
        << tile_dims.size() << ") exceeds the memory view rank ("
        << strides.size() << ")";
    return nullptr;
  }

  // Calculate offset from per-dim indices and memory view strides
  mlir::OpFoldResult offset =
      computeReinterpretCastOffset(builder, loc, per_dim_indices, strides);

  // A window at the origin spanning the whole view depends on nothing but the
  // view, so every such cast hoists to the same place and one can serve them
  // all.  Sharing it is what keeps the buffer fill and the indirect transfer on
  // the same SSA value, which is how BroadcastPromotion recognises the fill.
  const bool whole_view_origin = mlir::isConstantIntValue(offset, 0) &&
                                 tile_dims == memory_view.getType().getShape();
  if (whole_view_origin) {
    if (mlir::Value cached = origin_casts_.lookup(memory_view)) {
      return cached;
    }
  }

  // Map the ktdp memory space to the device namespace using mem_space_mapping.
  // A space that already maps to itself — an "IAB" view is a flat string space
  // to begin with — needs no cast.
  const auto memory_space = mapMemorySpace(getMemorySpace(memory_view));
  mlir::Value source = memory_view;
  if (memory_space != memory_view.getType().getMemorySpace()) {
    const auto cast_source_type =
        mlir::MemRefType::get(memory_view.getType().getShape(),
                              memory_view.getType().getElementType(),
                              memory_view.getType().getLayout(), memory_space);
    source = mlir::memref::MemorySpaceCastOp::create(
                 builder, loc, cast_source_type, memory_view)
                 .getResult();
  }

  llvm::ArrayRef<int64_t> tile_strides =
      llvm::ArrayRef(strides).take_back(tile_dims.size());

  llvm::SmallVector<mlir::OpFoldResult> sizes;
  vectorizeToIndexAttrs(tile_dims, builder, sizes);
  llvm::SmallVector<mlir::OpFoldResult> reinterpret_strides;
  vectorizeToIndexAttrs(tile_strides, builder, reinterpret_strides);

  const auto result_type = mlir::MemRefType::get(
      tile_dims, memory_view.getType().getElementType(),
      mlir::StridedLayoutAttr::get(builder.getContext(),
                                   mlir::ShapedType::kDynamic, tile_strides),
      memory_space);

  auto cast = mlir::memref::ReinterpretCastOp::create(
      builder, loc, result_type, source, offset, sizes, reinterpret_strides);

  // The cast chain can often leave more loops than the access tile it replaces:
  // a window at the origin depends only on the memory view, while an indirect
  // tile is pinned by the loop IV that selects its buffer entry.  Hoist the
  // space cast first so the reinterpret_cast can follow it out.
  if (source != memory_view) hoistOutOfLoops(source.getDefiningOp());
  hoistOutOfLoops(cast);

  if (whole_view_origin) origin_casts_[memory_view] = cast.getResult();

  return cast.getResult();
}

mlir::LogicalResult
ConstructThreeStagePipelinePass::replaceAccessTilesWithReinterpretCast(
    mlir::func::FuncOp func_op) {
  llvm::SmallVector<mlir::ktdp::ConstructAccessTilesOp> access_tiles;
  func_op.walk([&](mlir::ktdp::ConstructAccessTilesOp access_tile) {
    access_tiles.push_back(access_tile);
  });

  if (access_tiles.empty() && !ind_tile_) return mlir::success();

  // Hoist construct_memory_view and construct_access_tile ops to just before
  // the outermost tiled loop (or the func terminator if no loops exist).
  // This ensures both the memory views and the access tiles — and therefore the
  // memory_space_cast / reinterpret_cast we are about to emit in their place —
  // all precede the pipeline that consumes them.
  // We use the outermost tiled loop as the insertion anchor because that is
  // where the pipeline lives; everything hoisted before it will dominate all
  // uses inside the loop body.
  assert(!tiled_loops_.empty() && "expecting tiled loops");
  mlir::Operation* hoist_before = tiled_loops_.front();

  // Anchoring at the outermost tiled loop is only a floor: hoistOutOfLoops then
  // lifts each op through as many enclosing loops as its own operands allow, so
  // loop-invariant views and full-shape casts leave the loop nest entirely
  // while a slice whose offset depends on an outer IV stays at that IV's level.
  {
    llvm::SmallVector<mlir::Operation*> mem_views;
    func_op.walk([&](mlir::Operation* op) {
      if (mlir::isa<mlir::ktdp::ConstructMemoryViewOp,
                    mlir::ktdp_lowering::ConstructMemoryViewOp>(op))
        mem_views.push_back(op);
    });
    if (mlir::failed(hoistOperations(mem_views, hoist_before))) {
      signalPassFailure();
      return mlir::failure();
    }
  }
  {
    llvm::SmallVector<mlir::Operation*> tiles;
    for (auto at : access_tiles) tiles.push_back(at.getOperation());
    if (ind_tile_) tiles.push_back(ind_tile_->getOperation());
    if (mlir::failed(hoistOperations(tiles, hoist_before))) {
      signalPassFailure();
      return mlir::failure();
    }
  }

  mlir::OpBuilder builder(func_op);

  for (auto& access_tile : access_tiles) {
    auto memory_view = llvm::dyn_cast<mlir::TypedValue<mlir::MemRefType>>(
        access_tile.getBase());
    if (!memory_view) {
      access_tile->emitError("Memory view is not a memref type");
      signalPassFailure();
      return mlir::failure();
    }

    auto raw_indices = access_tile.getIndices();
    auto base_map = access_tile.getBaseMap();

    auto access_tile_type = mlir::dyn_cast<mlir::ktdp::AccessTileType>(
        access_tile.getResult().getType());
    if (!access_tile_type) {
      access_tile->emitError("Result is not an access tile type");
      signalPassFailure();
      return mlir::failure();
    }

    // Get tile dimensions from the access tile type.  The fill's two tiles are
    // collapsed to rank 1, which is what the downstream lowering expects of it.
    // checkCollapsesToRank1() verified this previously.
    llvm::ArrayRef<int64_t> tile_dims =
        transferShape(access_tile_type.getShape(),
                      isIndAddrBufFillTile(access_tile.getOperation()));

    // Insert the cast sequence immediately before the (now-hoisted)
    // construct_access_tile.  Both the memory view and index operands have
    // already been moved above the outermost loop, so all operands dominate
    // this insertion point.
    builder.setInsertionPoint(access_tile);
    mlir::Location loc = access_tile.getLoc();

    // Apply base_map to materialize one index per source-memref dimension.
    // Operands to expandAffineMap are dim-values followed by symbol-values; the
    // op's symbol_operands feed both base_map symbols (if any) and the
    // access_tile_set, so they are appended after the raw indices.
    llvm::SmallVector<mlir::Value> expand_operands(raw_indices.begin(),
                                                   raw_indices.end());
    mlir::ValueRange symbol_operands = access_tile.getSymbolOperands();
    expand_operands.append(symbol_operands.begin(), symbol_operands.end());
    std::optional<llvm::SmallVector<mlir::Value, 8>> per_dim_indices =
        mlir::affine::expandAffineMap(builder, loc, base_map, expand_operands);
    if (!per_dim_indices) {
      access_tile.emitError(
          "Failed to expand base_map into per-dimension "
          "indices");
      signalPassFailure();
      return mlir::failure();
    }
    llvm::SmallVector<mlir::Value> indices(per_dim_indices->begin(),
                                           per_dim_indices->end());

    mlir::Value cast = emitReinterpretCast(builder, loc, memory_view, indices,
                                           tile_dims, access_tile);
    if (!cast) {
      signalPassFailure();
      return mlir::failure();
    }

    // Replace access tile with reinterpret_cast
    access_tile.replaceAllUsesWith(cast);
    ops_to_delete_.push_back(access_tile.getOperation());
  }

  // An indirect tile needs two casts, and neither carries a per-iteration
  // offset: the base is addressed at offset 0 in its full shape because the
  // origin lives in the transfer's dir_*_map, and the buffer is a whole window.
  if (ind_tile_) {
    auto& ind_tile = *ind_tile_;
    auto base =
        llvm::dyn_cast<mlir::TypedValue<mlir::MemRefType>>(ind_tile.getBase());
    if (!base) {
      ind_tile->emitError("Memory view is not a memref type");
      signalPassFailure();
      return mlir::failure();
    }
    auto iab = llvm::dyn_cast<mlir::TypedValue<mlir::MemRefType>>(
        ind_tile.getIndAddrBufMemref());
    if (!iab) {
      ind_tile->emitError("Indirect address buffer is not a memref");
      signalPassFailure();
      return mlir::failure();
    }

    builder.setInsertionPoint(ind_tile);
    auto loc = ind_tile.getLoc();
    llvm::SmallVector<mlir::Value> no_indices;

    mlir::Value base_cast = emitReinterpretCast(
        builder, loc, base, no_indices, base.getType().getShape(), ind_tile);
    mlir::Value iab_cast = emitReinterpretCast(
        builder, loc, iab, no_indices, iab.getType().getShape(), ind_tile);
    if (!base_cast || !iab_cast) {
      signalPassFailure();
      return mlir::failure();
    }

    // The base cast reaches the transfer through the tile's own uses; the
    // buffer view has other users (the fill, the tile itself), so rewire just
    // the transfer's operand.
    ind_tile.getResult().replaceAllUsesWith(base_cast);
    for (auto& [transfer, tile] : ind_transfers_) {
      if (tile != ind_tile) continue;
      (transfer.isGather() ? transfer.getIndSrcMemrefMutable()
                           : transfer.getIndDstMemrefMutable())
          .assign(iab_cast);
    }
    ops_to_delete_.push_back(ind_tile.getOperation());
  }
  return mlir::success();
}

void ConstructThreeStagePipelinePass::runOnFunc(mlir::func::FuncOp func_op) {
  LDBG(1) << "Processing function: " << func_op.getName() << "";

  // Initialize const_builder_ at start of function
  if (!func_op.empty()) {
    const_builder_.emplace(&getContext());
    const_builder_->setInsertionPointToStart(&func_op.front());
  }

  // Classify indirect access before collecting loads and stores, so that the
  // indirect-address-buffer fill can be kept out of the pipeline's load/store
  // lists.
  if (mlir::failed(classifyIndirectAccess(func_op))) {
    signalPassFailure();
    return;
  }

  // Collect ktdp load/store operations and linalg operations
  llvm::SmallVector<mlir::linalg::GenericOp> linalg_ops;
  func_op.walk([&](mlir::Operation* op) {
    if (auto load_op = mlir::dyn_cast<mlir::ktdp::LoadOp>(op)) {
      if (!isIndAddrBufFillOp(op)) load_ops_.push_back(load_op);
      ops_to_delete_.push_back(op);
    } else if (auto store_op = mlir::dyn_cast<mlir::ktdp::StoreOp>(op)) {
      if (!isIndAddrBufFillOp(op)) store_ops_.push_back(store_op);
      ops_to_delete_.push_back(op);
    } else if (auto linalg_op = mlir::dyn_cast<mlir::linalg::GenericOp>(op)) {
      linalg_ops.push_back(linalg_op);
    }
  });

  // Step 3: Create loops from linalg operations
  if (mlir::failed(createLoopsFromLinalg(linalg_ops))) return;

  LDBG(1) << "After loops created:\n" << func_op << "\n";

  // Step 4: Create pipeline if we have tiled loops
  if (!tiled_loops_.empty()) {
    auto innermost_loop = llvm::cast<mlir::scf::ForOp>(tiled_loops_.back());

    // FIXME: Handle multiple compute units involved in a pipeline.
    auto compute =
        mapping_->resolve<mlir::ktdf_arch::ExecutionUnitOp>(linalg_ops.front());
    if (!compute) {
      llvm::report_fatal_error("default compute was not assigned");
    }
    for (auto linalg_op : llvm::ArrayRef(linalg_ops).drop_front()) {
      if (mapping_->resolve(linalg_op) != compute) {
        auto diag =
            linalg_op.emitError()
            << "unable to map to two different units in a single pipeline";
        diag.attachNote(linalg_ops.front()->getLoc())
            << "previous op was mapped to " << *compute.getOperation();
        signalPassFailure();
        return;
      }
    }

    if (mlir::failed(createPipeline(innermost_loop, compute))) return;
  }

  LDBG(1) << "After pipeline created:\n" << func_op << "\n";

  // Step 5: Replace access tiles with reinterpret_cast after pipeline creation
  if (mlir::failed(replaceAccessTilesWithReinterpretCast(func_op))) return;

  // Step 6: Cleanup operations (removes original load/store/compute ops)
  cleanupOperations();

  LDBG(1) << "After cleanup:\n" << func_op << "\n";
}

void ConstructThreeStagePipelinePass::runOnOperation() {
  if (DisableThisPass) {
    return;
  }

  // Obtain the default device, emitting a diagnostic on failure.
  const auto& default_device = getAnalysis<mlir::ktdf_arch::DefaultDevice>();
  if (!default_device) {
    signalPassFailure();
    return;
  }

  // Construct a mapping adapter and ensure we have a default compute resource.
  mlir::ktdf_arch::Mapping mapping(default_device.getRef());
  if (!mapping.byKind().getDefaultCompute()) {
    mapping.getDevice().getDeclaration().emitError(
        "no (unambiguous) default compute resource");
    signalPassFailure();
    return;
  }

  // FIXME: Don't put per-invocation pass state in pass members.
  mapping_ = &mapping;
  {
    mem_space_map_.clear();
    if (const auto mapping =
            mapping_->getDevice().getAttrOfType<mlir::ktdf_arch::MapAttr>(
                "mem_space_mapping");
        mapping) {
      mem_space_map_.insert_range(mapping);
    }
  }

  LDBG(1) << "Starting ConstructThreeStagePipeline transformation";

  // Process each function in each nested module
  getOperation().walk([&](mlir::func::FuncOp func_op) {
    resetState();
    runOnFunc(func_op);
  });
}

std::unique_ptr<mlir::Pass> scheduler::createConstructThreeStagePipelinePass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<ConstructThreeStagePipelinePass>(scheduler_ctx);
}

std::unique_ptr<mlir::Pass> scheduler::createConstructThreeStagePipelinePass() {
  return std::make_unique<ConstructThreeStagePipelinePass>(
      SchedulerExtContext::dummyContext());
}
