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
// This pass tiles perfectly nested scf.for loops with four execution modes:
//
// 1. Static mode with custom tiling (tile-sizes specified,
// fallback-scf-tiling=false):
//    Uses command-line tile sizes with custom normalized tiling scheme
// 2. Static mode with fallback SCF tiling (tile-sizes specified,
// fallback-scf-tiling=true):
//    Uses command-line tile sizes with SCF's standard tiling (for testing)
// 3. Dynamic mode with custom tiling (tile-sizes not specified,
// fallback-scf-tiling=false):
//    Uses LoopTilingAnalysis to identify candidates and custom normalized
//    tiling
// 4. Dynamic mode with fallback SCF tiling (tile-sizes not specified,
// fallback-scf-tiling=true):
//    Uses LoopTilingAnalysis with SCF's standard tiling (for testing)
//
// Custom tiling example (uses ktdf.tiling.derive_size and
// ktdf.tiling.linearize_index):
//   scf.for %i = 0 to 23 step 1 {
//     scf.for %j = 0 to 12 step 1 {
//       %sum = arith.addi %i, %j : index
//     }
//   }
// becomes (with tile sizes [5, 3]):
//   %n_tiles_i = arith.ceildivui %c23, %c5 : index
//   scf.for %tile_i = 0 to %n_tiles_i step 1 {
//     %n_tiles_j = arith.ceildivui %c12, %c3 : index
//     scf.for %tile_j = 0 to %n_tiles_j step 1 {
//       %tile_size_i = ktdf.tiling.derive_size [%tile_i : %c5], total_size =
//       %c23 %tile_size_j = ktdf.tiling.derive_size [%tile_j : %c3], total_size
//       = %c12 scf.for %point_i = 0 to %tile_size_i step 1 {
//         scf.for %point_j = 0 to %tile_size_j step 1 {
//           %i = ktdf.tiling.linearize_index[%tile_i : %c5], [%point_i : %c1]
//           %j = ktdf.tiling.linearize_index[%tile_j : %c3], [%point_j : %c1]
//           %sum = arith.addi %i, %j : index
//         }
//       }
//     }
//   }
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Analysis/LoopTiling.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/TileNormalized.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/SCFTilingUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_TILESCFFORLOOPSPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

struct TileSCFForLoopsPass
    : public impl::TileSCFForLoopsPassBase<TileSCFForLoopsPass> {
  using TileSCFForLoopsPassBase<TileSCFForLoopsPass>::TileSCFForLoopsPassBase;

  void runOnOperation() override {
    mlir::Operation* op = getOperation();

    if (!tileSizes.empty())
      runStaticMode(op);
    else
      runDynamicMode(op);
  }

 private:
  // Static mode: Use command-line tile sizes for testing
  void runStaticMode(mlir::Operation* op) {
    mlir::IRRewriter rewriter(op->getContext());

    // Collect all outermost scf.for loops using preorder walk with skip
    llvm::SmallVector<mlir::scf::ForOp> root_loops;
    op->walk<mlir::WalkOrder::PreOrder>([&](mlir::scf::ForOp for_op) {
      root_loops.push_back(for_op);
      return mlir::WalkResult::skip();
    });

    // Tile each perfectly nested loop nest
    for (mlir::scf::ForOp root_loop : root_loops) {
      if (mlir::failed(tileLoopNestStatic(root_loop, rewriter))) {
        return signalPassFailure();
      }
    }
  }

  // Dynamic mode: Use LoopTilingAnalysis to find candidates
  void runDynamicMode(mlir::Operation* op) {
    mlir::IRRewriter rewriter(op->getContext());

    // Get the shared LoopTilingAnalysis
    auto& analysis = getAnalysis<mlir::ktdf::LoopTilingAnalysis>();

    // Process each nest identified by the analysis
    for (const auto& nest : analysis.getAllNests()) {
      // Check if this nest is a candidate for tiling
      // Skip nests with invalidated loops
      bool is_candidate = true;

      for (mlir::scf::ForOp loop : nest.loops) {
        const auto* metadata = analysis.getMetadataForLoop(loop);
        if (!metadata || metadata->is_invalidated ||
            !metadata->isTilingCandidate()) {
          is_candidate = false;
          break;
        };
      }

      if (!is_candidate) continue;

      // Tile this nest with dynamic tile sizes
      tileLoopNestDynamic(nest, analysis, rewriter);

      // Mark all loops in this nest as invalidated
      for (mlir::scf::ForOp loop : nest.loops) {
        analysis.invalidateLoop(loop);
      }
    }

    // Preserve the analysis (except for entries that were transformed)
    markAnalysesPreserved<mlir::ktdf::LoopTilingAnalysis>();
  }

  // Tile a loop nest in static mode with command-line sizes
  mlir::LogicalResult tileLoopNestStatic(mlir::scf::ForOp root_loop,
                                         mlir::IRRewriter& rewriter) {
    // Collect the perfectly nested loops starting from root
    llvm::SmallVector<mlir::scf::ForOp> nested_loops;
    mlir::getPerfectlyNestedLoops(nested_loops, root_loop);

    if (nested_loops.empty()) return mlir::success();

    // Validate: number of tile sizes must match nest depth
    if (tileSizes.size() != nested_loops.size()) {
      return root_loop.emitError()
             << "Number of tile sizes (" << tileSizes.size()
             << ") does not match nest depth (" << nested_loops.size() << ")";
    }

    // Validate: all tile sizes must be positive
    for (size_t i = 0; i < tileSizes.size(); ++i) {
      if (tileSizes[i] <= 0) {
        return root_loop.emitError() << "Tile size at index " << i
                                     << " is non-positive: " << tileSizes[i];
      }
    }

    // Prepare tile sizes as Values
    llvm::SmallVector<mlir::Value> tile_size_values;
    rewriter.setInsertionPoint(root_loop);
    mlir::Location loc = root_loop.getLoc();

    for (int64_t tile_size : tileSizes) {
      tile_size_values.push_back(
          mlir::arith::ConstantIndexOp::create(rewriter, loc, tile_size));
    }

    applyTiling(root_loop, nested_loops, tile_size_values, rewriter);
    return mlir::success();
  }

  // Tile a loop nest in dynamic mode with LoopTilingAnalysis
  void tileLoopNestDynamic(const mlir::ktdf::PerfectNestingInfo& nest,
                           mlir::ktdf::LoopTilingAnalysis& analysis,
                           mlir::IRRewriter& rewriter) {
    if (nest.loops.empty()) return;

    mlir::scf::ForOp root_loop = nest.getOutermost();
    mlir::Location loc = root_loop.getLoc();

    // Materialize tiling.reserve_size ops above the outermost loop
    rewriter.setInsertionPoint(root_loop);

    llvm::SmallVector<mlir::Value> tile_size_values;
    for (mlir::scf::ForOp loop : nest.loops) {
      const auto* metadata = analysis.getMetadataForLoop(loop);

      // All loops in the nest should have metadata and be tiling candidates
      // since we checked this before calling this function
      assert(metadata && metadata->isTilingCandidate() &&
             "Expected all loops in nest to be tiling candidates");

      // Create ktdf.tiling.reserve_size op with metadata attributes
      auto reserve_size_op = mlir::ktdf::TilingReserveSizeOp::create(
          rewriter, loc, rewriter.getIndexType(),
          rewriter.getIndexAttr(metadata->min_value),
          rewriter.getIndexAttr(metadata->divisibility));

      tile_size_values.push_back(reserve_size_op.getResult());
    }

    // Collect the perfectly nested loops for tiling
    llvm::SmallVector<mlir::scf::ForOp> nested_loops;
    mlir::getPerfectlyNestedLoops(nested_loops, root_loop);

    applyTiling(root_loop, nested_loops, tile_size_values, rewriter);
  }

  // Apply tiling to a loop nest: try custom tiling first, fall back to SCF
  // tiling. Propagates loop_type attributes and normalizes loops when
  // requested. root_loop and nested_loops must be consistent (nested_loops
  // collected from root_loop). tile_size_values must be one per loop in
  // nested_loops. NOTE: root_loop may be erased by
  // customTileNormalizedPerfectlyNested.
  void applyTiling(mlir::scf::ForOp root_loop,
                   llvm::ArrayRef<mlir::scf::ForOp> nested_loops,
                   llvm::ArrayRef<mlir::Value> tile_size_values,
                   mlir::IRRewriter& rewriter) {
    // Try custom tiling first unless fallback-scf-tiling is set
    if (!useFallbackSCFTiling) {
      auto ktdf_result = mlir::ktdf::customTileNormalizedPerfectlyNested(
          root_loop, tile_size_values, rewriter);
      if (llvm::succeeded(ktdf_result)) return;
    }

    // Fall back to (or directly use) SCF's tilePerfectlyNested loops method.
    auto inner_loops = mlir::tilePerfectlyNested(root_loop, tile_size_values);

    // Propagate loop_type attribute from original loops to inner point loops
    if (!inner_loops.empty()) {
      propagateLoopTypeAttribute(nested_loops, inner_loops);
    }

    if (normalizeLoops && !inner_loops.empty()) {
      llvm::SmallVector<mlir::scf::ForOp> loops;
      loops.insert(loops.begin(), nested_loops.begin(), nested_loops.end());
      loops.insert(loops.end(), inner_loops.begin(), inner_loops.end());
      normalizeSCFLoops(loops, rewriter);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createTileSCFForLoopsPass() {
  return std::make_unique<TileSCFForLoopsPass>();
}

// Made with Bob
