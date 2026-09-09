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

#include "dataflow-scheduler/Dialect/KTDF/Transforms/TileNormalized.h"

#include <mlir/Dialect/Utils/StaticValueUtils.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

using namespace mlir;
using namespace mlir::ktdf;

FailureOr<TileNestResult> mlir::ktdf::customTileNormalizedPerfectlyNested(
    mlir::scf::ForOp root_loop, llvm::ArrayRef<Value> tile_sizes,
    IRRewriter& rewriter) {
  llvm::SmallVector<scf::ForOp> nested_loops;
  getPerfectlyNestedLoops(nested_loops, root_loop);

  if (nested_loops.empty() || tile_sizes.size() != nested_loops.size())
    return failure();

  for (scf::ForOp loop : nested_loops) {
    auto lb = getConstantIntValue(loop.getLowerBound());
    auto step = getConstantIntValue(loop.getStep());
    if (!lb || *lb != 0 || !step || *step != 1) return failure();
  }

  Location loc = root_loop.getLoc();
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

  // For each dimension, extract the original upper bound, compute ancestry from
  // an existing tiling.derive_size if present, and compute tile trip count.
  TileNestResult result;
  SmallVector<Value> total_sizes;
  SmallVector<SmallVector<std::pair<Value, Value>>> ancestry(
      nested_loops.size());

  rewriter.setInsertionPoint(root_loop);
  for (size_t i = 0; i < nested_loops.size(); ++i) {
    Value ub = nested_loops[i].getUpperBound();
    Value total_size = ub;

    if (auto gts = ub.getDefiningOp<TilingDeriveSizeOp>()) {
      auto ivs = gts.getIvs();
      auto tss = gts.getTileSizes();
      for (auto [iv, ts] : llvm::zip(ivs, tss)) ancestry[i].push_back({iv, ts});
      total_size = gts.getTotalSize();
    }
    total_sizes.push_back(total_size);

    // Tile loop: for %ti = 0 to ceildivui(ub, tile_sizes[i]) step 1
    Value trip = arith::CeilDivUIOp::create(rewriter, loc, ub, tile_sizes[i]);
    auto tile_loop = scf::ForOp::create(rewriter, loc, zero, trip, one);
    tile_loop->setAttrs(nested_loops[i]->getAttrs());
    result.tile_loops.push_back(tile_loop);

    rewriter.setInsertionPointToStart(tile_loop.getBody());
  }

  // Now inside the innermost tile loop: emit tiling.derive_size + point loops.
  SmallVector<Value> point_sizes;
  for (size_t i = 0; i < nested_loops.size(); ++i) {
    SmallVector<Value> gts_ivs, gts_tss;
    for (auto [iv, ts] : ancestry[i]) {
      gts_ivs.push_back(iv);
      gts_tss.push_back(ts);
    }
    gts_ivs.push_back(result.tile_loops[i].getInductionVar());
    gts_tss.push_back(tile_sizes[i]);
    Value ps = TilingDeriveSizeOp::create(rewriter, loc, gts_ivs, gts_tss,
                                          total_sizes[i]);
    point_sizes.push_back(ps);
  }

  for (size_t i = 0; i < nested_loops.size(); ++i) {
    auto point_loop =
        scf::ForOp::create(rewriter, loc, zero, point_sizes[i], one);
    point_loop->setAttrs(nested_loops[i]->getAttrs());
    result.point_loops.push_back(point_loop);
    rewriter.setInsertionPointToStart(point_loop.getBody());
  }

  // Move original body into innermost point loop (body fixup in Task 4).
  Block* src = nested_loops.back().getBody();
  Block* dst = result.point_loops.back().getBody();
  dst->getOperations().splice(dst->getTerminator()->getIterator(),
                              src->getOperations(), src->begin(),
                              std::prev(src->end()));

  // Fix up body: update tiling.linearize_index ops and replace bare IV uses.
  // Reuse the `one` constant created above for stride-1 in bare IV subscripts.

  for (size_t i = 0; i < nested_loops.size(); ++i) {
    Value old_iv = nested_loops[i].getInductionVar();
    Value tile_iv = result.tile_loops[i].getInductionVar();
    Value point_iv = result.point_loops[i].getInductionVar();
    Value tile_size = tile_sizes[i];
    Block* body = result.point_loops.back().getBody();

    // Fix A: update existing TilingLinearizeIndexOps that use old_iv.
    SmallVector<TilingLinearizeIndexOp> sub_ops;
    body->walk([&](TilingLinearizeIndexOp op) { sub_ops.push_back(op); });

    for (TilingLinearizeIndexOp sub_op : sub_ops) {
      auto ivs = llvm::to_vector(sub_op.getIvs());
      auto strides = llvm::to_vector(sub_op.getStrides());
      bool changed = false;
      for (size_t k = 0; k < ivs.size(); ++k) {
        if (ivs[k] != old_iv) continue;
        Value old_stride = strides[k];
        ivs.erase(ivs.begin() + k);
        strides.erase(strides.begin() + k);
        ivs.insert(ivs.begin() + k, point_iv);
        strides.insert(strides.begin() + k, old_stride);
        ivs.insert(ivs.begin() + k, tile_iv);
        strides.insert(strides.begin() + k, tile_size);
        changed = true;
        break;
      }
      if (!changed) continue;
      rewriter.setInsertionPoint(sub_op);
      auto new_op = TilingLinearizeIndexOp::create(rewriter, sub_op.getLoc(),
                                                   ivs, strides);
      rewriter.replaceOp(sub_op, new_op.getResult());
    }

    // Fix B: replace remaining bare uses of old_iv with a new
    // tiling.linearize_index.
    if (old_iv.use_empty()) continue;
    SmallVector<Value> sub_ivs, sub_strides;
    for (auto [iv, ts] : ancestry[i]) {
      sub_ivs.push_back(iv);
      sub_strides.push_back(ts);
    }
    sub_ivs.push_back(tile_iv);
    sub_strides.push_back(tile_size);
    sub_ivs.push_back(point_iv);
    sub_strides.push_back(one);
    rewriter.setInsertionPointToStart(body);
    auto new_sub =
        TilingLinearizeIndexOp::create(rewriter, loc, sub_ivs, sub_strides);
    rewriter.replaceAllUsesWith(old_iv, new_sub.getResult());
  }

  // Erase original loop nest.
  rewriter.eraseOp(root_loop);

  return result;
}

// Made with Bob
