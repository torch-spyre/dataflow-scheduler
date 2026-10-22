//===-- ReductionDimChunking.cpp --------------------------------*- c++ -*-===//
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
// ReductionDimChunking: split reduction dimensions of the inner linalg.generic
// into sequential chunks so that each chunk fits in the hardware FIFO path.
//
// When the chunk sizes are inferred from the threshold (--num-chunks not
// given), ReductionFootprint selects them: the reduction dimensions are
// shrunk from the outermost to the innermost in the layout of the inputs until
// the input tiles of a chunk fit the threshold.  The innermost overall loop
// dimension is never chunked.
//
// When --num-chunks is given explicitly, multi-dim chunking is applied using
// one value per reduction dimension (in iterator-type order); one nested
// scf.for is emitted per dim; dims with chunk count 1 produce no loop.
//
// The pass operates on the shape produced by StageCoarsening.  It expects a
// top-level ktdf.pipeline with three sibling stages:
//
//   stage-0 (Load)   : global memory → local memory
//   stage-1 (Compute): outer scf.for (batch) whose body is a single inner
//                       ktdf.pipeline with three stages:
//                         stage-a (Load)   : local memory → FIFO
//                         stage-b (Compute): linalg.generic reduce
//                         stage-c (Store)  : FIFO → local memory
//   stage-2 (Store)  : local memory → global memory
//
// The transformation wraps the single inner ktdf.pipeline in nested scf.for
// loops — one per reduction dimension that has more than one chunk — and
// rewrites it in place to process one chunk per innermost iteration.
// Dimensions whose num_chunks==1 produce no loop (the single chunk covers the
// entire dimension).  The compute stage computes its chunk with the op tiling
// with full-reduction semantics yields.  First-vs-rest accumulation behaviour
// is selected at runtime via
// %condition = AND(iv_j == 0 for all active chunk loop IVs):
//
//   Load stage   : transfers the current input chunk slice (memref → fifo_in)
//                  instead of the whole input.
//                  When !condition, also transfers the partial accumulator
//                  (local memory output buffer → fifo_partial) so the
//                  Compute stage can read it back.
//   Compute stage: unconditionally reads fifo_in and runs the tiled
//                  linalg.generic into the original init.  Then branches on
//                  condition:
//                    then (first iter): yield first generic result.
//                    else (later iters): read fifo_partial, run a second
//                      linalg.generic(input=partial, output=first result),
//                      yield second generic result.
//                  The conditional result is written to fifo_out.
//   Store stage  : unchanged: unconditionally writes fifo_out back to the
//                  local memory output buffer.
//
// The existing local memory output buffer (discovered via the original Store
// stage's data_transfer destination) is reused as the partial-accumulation
// buffer; no new memref is allocated.
//
//===----------------------------------------------------------------------===//

#include <mlir/Dialect/Linalg/IR/LinalgInterfaces.h>
#include <mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/Interfaces/TilingInterface.h>

#include <functional>
#include <memory>
#include <optional>
#include <tuple>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/ReductionFootprint.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/DataTransfers.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/ReductionUtils.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define PASS_NAME "reduction-dim-chunking"
#define DEBUG_TYPE PASS_NAME

using namespace mlir;
using namespace mlir::ktdf;

namespace mlir::ktdf {
#define GEN_PASS_DEF_REDUCTIONDIMCHUNKINGPASS
#include "dataflow-scheduler/Dialect/KTDF/Transforms/Passes.h.inc"
}  // namespace mlir::ktdf

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Reduction Dim Chunking pass"),
    llvm::cl::init(false));

namespace {

// ---------------------------------------------------------------------------
// Helper: find the unique linalg.generic with a reduction iterator inside a
// ktdf.stage body.  Returns nullptr if not found.
// ---------------------------------------------------------------------------
static linalg::GenericOp findReductionGenericOp(ktdf::StageOp stage) {
  linalg::GenericOp found;
  stage.getBody()->walk([&](linalg::GenericOp generic) {
    for (auto it : generic.getIteratorTypesArray()) {
      if (it == utils::IteratorType::reduction) {
        found = generic;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

// ---------------------------------------------------------------------------
// Pass implementation
// ---------------------------------------------------------------------------
struct ReductionDimChunkingPass
    : public ktdf::impl::ReductionDimChunkingPassBase<
          ReductionDimChunkingPass> {
  using ReductionDimChunkingPassBase<
      ReductionDimChunkingPass>::ReductionDimChunkingPassBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    ReductionDimChunkingPassBase::getDependentDialects(registry);
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    ModuleOp module = getOperation();
    if (failed(transformModule(module))) signalPassFailure();
  }

 private:
  // -----------------------------------------------------------------------
  // Walk the module for linalg.generic ops that have at least one reduction
  // iterator.  For each such op, locate the immediately enclosing ktdf.stage
  // (the Compute stage) and collect unique stages into a list.  The list is
  // processed after the walk to avoid mutating IR while walking.
  // -----------------------------------------------------------------------
  LogicalResult transformModule(ModuleOp module) {
    // Use a DenseSet to avoid O(n²) duplicate checks.
    llvm::DenseSet<ktdf::StageOp> seen;
    SmallVector<ktdf::StageOp> compute_stages;
    module.walk([&](linalg::GenericOp generic) {
      // Check if this generic has a reduction iterator.
      bool has_reduction = false;
      for (auto it : generic.getIteratorTypesArray()) {
        if (it == utils::IteratorType::reduction) {
          has_reduction = true;
          break;
        }
      }
      if (!has_reduction) return WalkResult::advance();

      // linalg.generic is always directly nested inside a ktdf.stage
      // (structural invariant enforced by StageCoarsening).
      auto stage = generic->getParentOfType<ktdf::StageOp>();
      assert(stage && "linalg.generic not inside a ktdf.stage");

      if (seen.insert(stage).second) compute_stages.push_back(stage);
      return WalkResult::advance();
    });

    for (auto compute_stage : compute_stages)
      if (failed(transformPipeline(compute_stage))) return failure();

    return success();
  }

  // -----------------------------------------------------------------------
  // Entry point for one Compute stage.  Determines the reduction dims, chunk
  // counts, and per-dim chunk sizes, then delegates to rewriteComputeStage().
  // -----------------------------------------------------------------------
  LogicalResult transformPipeline(ktdf::StageOp compute_stage) {
    // Stages are always direct children of a pipeline.
    auto inner_pipeline = cast<ktdf::PipelineOp>(compute_stage->getParentOp());

    // Verify there is exactly one inner pipeline inside the parent region.
    unsigned inner_pipeline_count = 0;
    inner_pipeline->getParentRegion()->walk<WalkOrder::PreOrder>(
        [&](ktdf::PipelineOp) {
          ++inner_pipeline_count;
          return WalkResult::skip();
        });
    assert(inner_pipeline_count == 1 &&
           "expected exactly 1 inner pipeline in parent region");

    ktdf::StageOp load_stage = findLoadStage(inner_pipeline, compute_stage);
    assert(load_stage && "could not find load stage upstream of compute stage");

    ktdf::StageOp store_stage = findStoreStage(inner_pipeline, compute_stage);
    assert(store_stage &&
           "could not find store stage downstream of compute stage");

    // Locate the linalg.generic with a reduction iterator.
    linalg::GenericOp generic_op = findReductionGenericOp(compute_stage);
    assert(generic_op && "no reduction linalg.generic in compute stage");

    // FIXME: Current implementation leads to invalid IR when processing
    //        contractions, so skip them for now to allow matmul to pass.
    if (linalg::isaContractionOpInterface(generic_op)) {
      LDBG() << "ignoring contraction";
      return success();
    }

    // ------------------------------------------------------------------
    // Determine reduction dims and per-dim chunk sizes.
    //
    // Threshold path (numChunks empty): ReductionFootprint selects the
    // chunk sizes; the innermost overall loop dim is excluded if it is a
    // reduction dim.
    //
    // Explicit path (numChunks non-empty): one value per reduction dim;
    // nested scf.for loops; dims with count 1 produce no loop.
    // ------------------------------------------------------------------

    // FIXME: This implementation assumes that there is an injective mapping
    //        from loop to shape space, i.e., for each iteration, there will be
    //        a distinct element read from the first input tensor.
    SmallVector<unsigned> reduction_dims;
    generic_op.getReductionDims(reduction_dims);
    SmallVector<int64_t> red_dim_sizes;
    SmallVector<int64_t> chunk_sizes;
    const auto loop_ranges = generic_op.getStaticLoopRanges();

    if (numChunks.empty()) {
      // Collect all reduction dims, excluding the innermost overall loop dim
      // if it is a reduction (the innermost loop dim must never be chunked,
      // regardless of iterator type).
      if (!reduction_dims.empty() &&
          reduction_dims.back() == generic_op.getNumLoops() - 1) {
        reduction_dims.pop_back();
      }
      if (reduction_dims.empty()) {
        LDBG(1) << PASS_NAME
            ": could not find a chunkable reduction dimension — skipping";
        return success();
      }

      const auto footprint = ReductionFootprint::get(generic_op);
      if (failed(footprint)) {
        inner_pipeline.emitError(PASS_NAME
                                 ": ReductionFootprint could not model the "
                                 "reduction");
        return failure();
      }
      const auto tile_sizes =
          footprint->getTileSizes(getBudget(load_stage, generic_op));
      if (!tile_sizes) {
        inner_pipeline.emitError(llvm::Twine(PASS_NAME
                                             ": no chunking of the reduction "
                                             "fits the threshold of ") +
                                 llvm::Twine(chunkSizeThreshold) + " bytes");
        return failure();
      }

      for (const auto dim : reduction_dims) {
        red_dim_sizes.push_back(loop_ranges[dim]);
        chunk_sizes.push_back((*tile_sizes)[dim]);
      }
    } else {
      // User-supplied --num-chunks path: collect reduction dims and validate.
      if (reduction_dims.empty()) {
        inner_pipeline.emitError(PASS_NAME
                                 ": could not find reduction dimension");
        return failure();
      }

      if (numChunks.size() != reduction_dims.size()) {
        inner_pipeline.emitError(
            llvm::Twine(PASS_NAME ": numChunks has ") +
            llvm::Twine(numChunks.size()) + " entries but there are " +
            llvm::Twine(reduction_dims.size()) + " reduction dims");
        return failure();
      }

      for (size_t j = 0; j < reduction_dims.size(); ++j) {
        const auto dim = reduction_dims[j];
        const auto dim_size = loop_ranges[dim];
        red_dim_sizes.push_back(dim_size);
        if (dim_size == ShapedType::kDynamic) {
          LDBG(1) << PASS_NAME
                  << ": dynamic reduction size not yet supported — skipping";
          return success();
        }
        unsigned nchunk = numChunks[j];
        if (nchunk == 0 || dim_size % nchunk != 0) {
          LDBG(1) << PASS_NAME ": num_chunks=" << nchunk
                  << " does not evenly divide dim " << dim
                  << " size=" << dim_size << " — skipping";
          return success();
        }
        chunk_sizes.push_back(dim_size / static_cast<int64_t>(nchunk));
      }
    }

    // Collect per-dim chunk counts in reduction-dim order.
    SmallVector<int64_t> per_dim_num_chunks;
    for (auto [dim_size, chunk_size] :
         llvm::zip_equal(red_dim_sizes, chunk_sizes)) {
      per_dim_num_chunks.push_back(dim_size / chunk_size);
    }

    LDBG(1) << PASS_NAME ": num_reduction_dims=" << reduction_dims.size();

    // One chunk on every dimension leaves the reduction as it is: the rewrite
    // below would rebuild the pipeline to emit the same program, differing only
    // in how the indices print. It also keeps the pass off a single-level
    // pipeline, which has no batch loop for rewriteComputeStage to index
    // against.
    if (llvm::all_of(per_dim_num_chunks,
                     [](int64_t chunks) { return chunks == 1; })) {
      LDBG(1) << PASS_NAME ": every reduction dim has one chunk, nothing to do";
      return success();
    }

    return rewriteComputeStage(inner_pipeline, load_stage, compute_stage,
                               store_stage, generic_op, reduction_dims,
                               chunk_sizes, per_dim_num_chunks);
  }

  // -----------------------------------------------------------------------
  // The budget a chunk of `generic_op` has: chunk-size-threshold bytes for
  // every input in the memory the load stage transfers it from.
  // -----------------------------------------------------------------------
  FootprintBudget getBudget(ktdf::StageOp load_stage,
                            linalg::GenericOp generic_op) const {
    FootprintBudget budget;
    for (OpOperand* input : generic_op.getDpsInputOperands()) {
      auto read = input->get().getDefiningOp<ktdf::ReadFromFifoOp>();
      if (!read) continue;
      load_stage.getBody()->walk([&](ktdf::DataTransferOp transfer) {
        if (!transfer.isSourceMemRef() ||
            transfer.getDestination() != read.getFifoSlot())
          return WalkResult::advance();
        Attribute memory =
            cast<MemRefType>(transfer.getSource().getType()).getMemorySpace();
        budget.place(*input, memory);
        budget.capacity[memory] = chunkSizeThreshold;
        return WalkResult::interrupt();
      });
    }
    return budget;
  }

  // -----------------------------------------------------------------------
  // Wrap inner_pipeline in one scf.for per chunked reduction dimension, and
  // rewrite it in place to process one chunk per iteration:
  //
  //   scf.for %iv_0 = 0 to num_chunks[0] {
  //     ...
  //       %first = (iv_0 == 0 && ...)
  //       ktdf.pipeline {
  //         Load:    + if !%first: partial_i <- where output i is stored
  //         Compute: inputs sliced to the chunk,
  //                  outputs accumulated into if %first {init_i}
  //                                           else {read partial_i}
  //         Store:   unchanged
  //       }
  //   }
  //
  // The data movement coalescing patterns carry the input slices over to the
  // load stage.
  // -----------------------------------------------------------------------
  LogicalResult rewriteComputeStage(
      ktdf::PipelineOp inner_pipeline, ktdf::StageOp load_stage,
      ktdf::StageOp compute_stage, ktdf::StageOp store_stage,
      linalg::GenericOp generic_op, ArrayRef<unsigned> reduction_dims,
      ArrayRef<int64_t> chunk_sizes, ArrayRef<int64_t> per_dim_num_chunks) {
    MLIRContext* context = inner_pipeline.getContext();
    IRRewriter rewriter(context);
    Location loc = inner_pipeline.getLoc();

    // The batch loop is a precondition: without it, the stages loading into
    // local memory share the pipeline, and would be repeated for every chunk.
    if (!isa<scf::ForOp>(inner_pipeline->getParentRegion()->getParentOp()))
      return inner_pipeline.emitError(PASS_NAME)
             << ": reduction needs chunking, but its pipeline is not nested in "
                "a batch scf.for, so there is no index to chunk against";

    // Every output must be written to a slot the store stage stores to memory,
    // where the next chunk reads it back from.
    SmallVector<ktdf::DataTransferOp> stores;
    for (OpResult result : generic_op->getResults()) {
      auto write = result.hasOneUse()
                       ? dyn_cast<ktdf::WriteToFifoOp>(*result.user_begin())
                       : nullptr;
      auto* consumer = write ? ktdf::getConsumer(write.getFifoSlot()) : nullptr;
      auto store = consumer
                       ? dyn_cast<ktdf::DataTransferOp>(consumer->getOwner())
                       : nullptr;
      if (!store || store->getParentOp() != store_stage ||
          !store.isDestMemRef() ||
          store.getTransferMode() != ktdf::TransferMode::Copy)
        return inner_pipeline.emitError(PASS_NAME)
               << ": output #" << result.getResultNumber()
               << " is not stored to memory by the store stage";
      stores.push_back(store);
    }

    // The partial results travel the way the first operand does.
    auto input_read =
        generic_op->getOperand(0).getDefiningOp<ktdf::ReadFromFifoOp>();
    if (!input_read)
      return inner_pipeline.emitError(PASS_NAME)
             << ": expected the first operand to be read from a FIFO";
    auto input_slot =
        cast<ktdf::FifoSlotType>(input_read.getFifoSlot().getType());
    auto input_result = cast<OpResult>(input_read.getFifoSlot());
    Operation* input_alloc = cast<ktdf::PrivateOp>(input_result.getOwner())
                                 .getYieldOp()
                                 ->getOperand(input_result.getResultNumber())
                                 .getDefiningOp();

    // The chunk loops, and the offset of each loop dim in the chunk.
    rewriter.setInsertionPoint(inner_pipeline);
    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    SmallVector<OpFoldResult> offsets(generic_op.getNumLoops(),
                                      rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes =
        getAsIndexOpFoldResult(context, generic_op.getStaticLoopRanges());
    Value first;
    for (auto [dim, chunk_size, num_chunks] :
         llvm::zip_equal(reduction_dims, chunk_sizes, per_dim_num_chunks)) {
      if (num_chunks == 1) continue;
      auto loop = scf::ForOp::create(
          rewriter, loc, c0,
          arith::ConstantIndexOp::create(rewriter, loc, num_chunks), c1);
      rewriter.setInsertionPointToStart(loop.getBody());
      Value iv = loop.getInductionVar();
      offsets[dim] = affine::makeComposedFoldedAffineApply(
          rewriter, loc, rewriter.getAffineDimExpr(0) * chunk_size, {iv});
      sizes[dim] = rewriter.getIndexAttr(chunk_size);
      Value is_first = arith::CmpIOp::create(rewriter, loc,
                                             arith::CmpIPredicate::eq, iv, c0);
      first = first ? arith::AndIOp::create(rewriter, loc, first, is_first)
                    : is_first;
    }
    rewriter.moveOpBefore(inner_pipeline, rewriter.getInsertionBlock(),
                          rewriter.getInsertionPoint());

    // The compute stage computes the chunk. Outputs not indexed by a chunked
    // dim accumulate over the chunks, the others are partitioned by them. The
    // index computations of the slices stay outside the pipeline, where the
    // load and store stages can see them.
    rewriter.setInsertionPoint(inner_pipeline);
    auto tiled = cast<TilingInterface>(generic_op.getOperation())
                     .getTiledImplementation(rewriter, offsets, sizes);
    if (failed(tiled))
      return inner_pipeline.emitError(PASS_NAME)
             << ": could not tile the reduction to a chunk";
    SmallVector<OpFoldResult> result_offsets, result_sizes;
    SmallVector<Value> results;
    for (OpResult result : generic_op->getResults()) {
      if (failed(cast<TilingInterface>(generic_op.getOperation())
                     .getResultTilePosition(rewriter, result.getResultNumber(),
                                            offsets, sizes, result_offsets,
                                            result_sizes)))
        return inner_pipeline.emitError(PASS_NAME)
               << ": could not tile output #" << result.getResultNumber();
      results.push_back(tiled->tiledValues[result.getResultNumber()]);
      auto type = cast<RankedTensorType>(result.getType());
      if (cast<RankedTensorType>(results.back().getType()) == type) continue;
      auto empty = tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                           type.getElementType());
      results.back() = tensor::InsertSliceOp::create(
          rewriter, loc, results.back(), empty, result_offsets, result_sizes,
          SmallVector<OpFoldResult>(type.getRank(), rewriter.getIndexAttr(1)));
      tiled->generatedSlices.push_back(empty);
      tiled->generatedSlices.push_back(results.back().getDefiningOp());
    }
    for (Operation* op : llvm::concat<Operation* const>(tiled->generatedSlices,
                                                        tiled->tiledOps))
      rewriter.moveOpBefore(op, generic_op);
    for (Value result : results) {
      if (auto insert = result.getDefiningOp<tensor::InsertSliceOp>())
        rewriter.moveOpAfter(insert, tiled->tiledOps.front());
    }
    rewriter.replaceOp(generic_op, results);
    generic_op = cast<linalg::GenericOp>(tiled->tiledOps.front());
    for (Operation* op : tiled->generatedSlices) {
      auto slice = dyn_cast<tensor::ExtractSliceOp>(op);
      if (slice && isIdentitySlice(slice))
        rewriter.replaceOp(slice, slice.getSource());
    }

    // Every accumulator starts from its init on the first chunk, and from the
    // partial result the previous chunk stored otherwise.
    {
      ktdf::PrivateBuilder slots(rewriter, inner_pipeline, loc);
      for (auto [init, store] :
           llvm::zip_equal(generic_op.getDpsInitsMutable(), stores)) {
        if (init.get().getDefiningOp<tensor::ExtractSliceOp>()) continue;
        auto type = cast<RankedTensorType>(init.get().getType());
        Value partial =
            slots
                .createFifo(ktdf::FifoSlotType::get(
                    context, input_slot.getSrc(), input_slot.getDest(),
                    type.getNumElements(), type.getElementType()))
                .front();
        rewriter.moveOpAfter(partial.getDefiningOp(), input_alloc);
        input_alloc = partial.getDefiningOp();

        rewriter.setInsertionPoint(generic_op);
        auto acc = scf::IfOp::create(rewriter, loc, TypeRange{type}, first,
                                     /*withElseRegion=*/true);
        rewriter.setInsertionPointToStart(acc.thenBlock());
        scf::YieldOp::create(rewriter, loc, init.get());
        rewriter.setInsertionPointToStart(acc.elseBlock());
        scf::YieldOp::create(
            rewriter, loc,
            ktdf::ReadFromFifoOp::create(rewriter, loc, type, partial)
                .getResult());
        rewriter.modifyOpInPlace(
            generic_op, [&, &init = init]() { init.set(acc.getResult(0)); });

        rewriter.setInsertionPointToEnd(load_stage.getBody());
        auto load = scf::IfOp::create(rewriter, loc, TypeRange{}, first,
                                      /*withElseRegion=*/true);
        rewriter.setInsertionPointToStart(load.elseBlock());
        OpOperand& dest = store.getDestinationMutable();
        const auto [map, indices] = store.getTiedOffset(&dest);
        ktdf::DataTransferOp::create(
            rewriter, loc, dest.get(), map,
            cloneOutOf(rewriter, indices, store_stage),
            store.getTiedSizes(&dest), partial, AffineMap{}, ValueRange{},
            ArrayRef<OpFoldResult>{
                rewriter.getIndexAttr(type.getNumElements())});
      }
    }

    // Splits the combine off the accumulation, where possible.
    for (OpOperand& init : generic_op.getDpsInitsMutable())
      (void)extractCombiner(rewriter, generic_op, init);

    RewritePatternSet patterns(context);
    scheduler::populateDataMovementCoalescingPatterns(patterns);
    GreedyRewriteConfig config;
    config.enableFolding(false).enableConstantCSE(false);
    auto* const scope =
        inner_pipeline->getParentWithTrait<OpTrait::IsIsolatedFromAbove>();
    if (failed(applyPatternsGreedily(scope, std::move(patterns), config)))
      return inner_pipeline.emitError(PASS_NAME)
             << ": data movement coalescing did not converge";
    if (!llvm::all_of(generic_op.getDpsInputs(), [](Value input) {
          return !input.getDefiningOp<tensor::ExtractSliceOp>();
        }))
      return inner_pipeline.emitError(PASS_NAME)
             << ": could not fold the chunk into the load transfer";

    DominanceInfo dominance;
    eliminateCommonSubExpressions(rewriter, dominance, load_stage);
    return success();
  }

  // Determines whether `slice` takes the whole of its source.
  static bool isIdentitySlice(tensor::ExtractSliceOp slice) {
    return slice.getSourceType() == slice.getResultType() &&
           llvm::all_of(slice.getMixedOffsets(),
                        [](OpFoldResult ofr) { return isZeroInteger(ofr); }) &&
           llvm::all_of(slice.getMixedStrides(),
                        [](OpFoldResult ofr) { return isOneInteger(ofr); });
  }

  // Returns `values`, cloning the pure ops computing them within `scope` to
  // the insertion point of `rewriter`.
  static SmallVector<Value> cloneOutOf(RewriterBase& rewriter,
                                       ValueRange values, Operation* scope) {
    IRMapping mapping;
    std::function<Value(Value)> get = [&](Value value) -> Value {
      if (Value mapped = mapping.lookupOrNull(value)) return mapped;
      Operation* def = value.getDefiningOp();
      if (!def || !scope->isProperAncestor(def)) return value;
      assert(isPure(def) && def->getNumRegions() == 0 &&
             "index computations are expected to be pure");
      for (Value operand : def->getOperands()) (void)get(operand);
      rewriter.clone(*def, mapping);
      return mapping.lookup(value);
    };
    return llvm::map_to_vector(values, get);
  }

  // Rewrites `op` accumulating into `init = scf.if %c {%v} else {%partial}`
  // to accumulate into %v, and to combine %partial into its result on the
  // else branch instead. Fails unless the body of `op` is a single binary op
  // of its block arguments, with only `init` as an output.
  static LogicalResult extractCombiner(RewriterBase& rewriter,
                                       linalg::GenericOp op, OpOperand& init) {
    auto acc = init.get().getDefiningOp<scf::IfOp>();
    Block* body = op.getBody();
    if (!acc || !acc->hasOneUse() || op->getNumResults() != 1 ||
        op.getNumDpsInputs() != 1 ||
        !llvm::hasSingleElement(body->without_terminator()))
      return failure();
    Operation& combiner = body->front();
    if (combiner.getNumOperands() != 2 || combiner.getNumResults() != 1 ||
        !llvm::is_contained(combiner.getOperands(), body->getArgument(0)) ||
        !llvm::is_contained(combiner.getOperands(), body->getArgument(1)) ||
        body->getTerminator()->getOperand(0) != combiner.getResult(0))
      return failure();

    Value value = acc.thenYield().getOperand(0);
    Value partial = acc.elseYield().getOperand(0);
    Value result = op->getResult(0);
    rewriter.modifyOpInPlace(op, [&]() { init.set(value); });
    rewriter.moveOpAfter(acc, op);
    rewriter.replaceAllUsesWith(result, acc.getResult(0));
    rewriter.modifyOpInPlace(acc.thenYield(),
                             [&]() { acc.thenYield()->setOperand(0, result); });

    // The combine has the body of `op`, as an elementwise op on the outputs.
    auto type = cast<RankedTensorType>(result.getType());
    AffineMap identity =
        rewriter.getMultiDimIdentityMap(static_cast<unsigned>(type.getRank()));
    rewriter.setInsertionPoint(acc.elseYield());
    auto combine = linalg::GenericOp::create(
        rewriter, op.getLoc(), TypeRange{type}, ValueRange{partial},
        ValueRange{result}, ArrayRef<AffineMap>{identity, identity},
        SmallVector<utils::IteratorType>(type.getRank(),
                                         utils::IteratorType::parallel));
    rewriter.cloneRegionBefore(op.getRegion(), combine.getRegion(),
                               combine.getRegion().end());
    rewriter.modifyOpInPlace(acc.elseYield(), [&]() {
      acc.elseYield()->setOperand(0, combine.getResult(0));
    });
    return success();
  }
};

}  // namespace

auto mlir::ktdf::createReductionDimChunkingPass() -> std::unique_ptr<Pass> {
  return std::make_unique<ReductionDimChunkingPass>();
}
