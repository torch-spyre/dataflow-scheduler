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
// This pass performs strip-mining on scf.for loops.
// Strip-mining splits a loop into an outer loop and an inner loop,
// where the outer loop iterates over blocks and the inner loop
// iterates within each block.
//
// Two execution paths:
// 1. Static mode (when strip-mine-size specified): Use command-line size for
//    all loops (testing mode)
// 2. Dynamic mode (when strip-mine-size not specified): Use LoopTilingAnalysis
//    to identify strip-mining candidates and materialize tiling.reserve_size
//    ops.
//
// Example: scf.for %i = 0 to 100 step 1 { ... }
// becomes: scf.for %i0 = 0 to 100 step 32 {
//            scf.for %i1 = %i0 to min(%i0+32, 100) step 1 { ... }
//          }
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Analysis/LoopTiling.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/SCFTilingUtils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "strip-mine-scf-for-loops"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_STRIPMINESCFFORLOOPSPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {
const char VerboseDebug[] = DEBUG_TYPE "-verbose";
}  // namespace

static llvm::cl::opt<bool> DisableStripMineSCFForLoopsPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Strip Mine SCF For Loops pass"),
    llvm::cl::init(false));

namespace {

struct StripMineSCFForLoopsPass
    : public impl::StripMineSCFForLoopsPassBase<StripMineSCFForLoopsPass> {
  using StripMineSCFForLoopsPassBase<
      StripMineSCFForLoopsPass>::StripMineSCFForLoopsPassBase;

  void runOnOperation() override {
    if (DisableStripMineSCFForLoopsPass) return;
    DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << PASS_NAME " running\n");

    mlir::Operation* op = getOperation();

    if (!stripMineSize.empty())
      runStaticMode(op);
    else
      runDynamicMode(op);
  }

 private:
  // Static mode: Use command-line strip-mine size for all loops
  void runStaticMode(mlir::Operation* op) {
    mlir::IRRewriter rewriter(op->getContext());

    // Validate: strip-mine size must be positive
    if (!stripMineSize.empty() && stripMineSize[0] <= 0) {
      op->emitError() << "Strip-mine size must be positive: "
                      << stripMineSize[0];
      return signalPassFailure();
    }

    int64_t strip_mine_factor = stripMineSize[0];

    // Collect ALL scf.for loops (not just outermost)
    llvm::SmallVector<mlir::scf::ForOp> allLoops;
    op->walk([&](mlir::scf::ForOp for_op) { allLoops.push_back(for_op); });

    // Strip-mine each loop individually
    for (mlir::scf::ForOp for_op : allLoops) {
      stripMineLoopStatic(for_op, strip_mine_factor, rewriter);
    }
  }

  // Dynamic mode: Use LoopTilingAnalysis to find strip-mining candidates
  void runDynamicMode(mlir::Operation* op) {
    mlir::IRRewriter rewriter(op->getContext());

    // Get the shared LoopTilingAnalysis
    auto& analysis = getAnalysis<mlir::ktdf::LoopTilingAnalysis>();

    // Collect all loops that are strip-mining candidates
    llvm::SmallVector<mlir::scf::ForOp> candidate_loops;
    op->walk([&](mlir::scf::ForOp for_op) {
      const auto* metadata = analysis.getMetadataForLoop(for_op);
      if (metadata && !metadata->is_invalidated &&
          metadata->isStripMiningCandidate()) {
        candidate_loops.push_back(for_op);
      }
    });

    // Strip-mine each candidate loop with its own tiling.reserve_size op
    for (mlir::scf::ForOp for_op : candidate_loops) {
      stripMineLoopDynamic(for_op, analysis, rewriter);

      // Mark this loop as invalidated
      analysis.invalidateLoop(for_op);
    }

    // Preserve the analysis (except for entries that were transformed)
    markAnalysesPreserved<mlir::ktdf::LoopTilingAnalysis>();
  }

  // Strip-mine a loop in static mode with command-line size
  void stripMineLoopStatic(mlir::scf::ForOp for_op, int64_t strip_mine_factor,
                           mlir::IRRewriter& rewriter) {
    // Create tile size as a Value
    rewriter.setInsertionPoint(for_op);
    mlir::Location loc = for_op.getLoc();
    mlir::Value tile_size_val =
        mlir::arith::ConstantIndexOp::create(rewriter, loc, strip_mine_factor);

    // Use MLIR's tile utility to strip-mine this single loop
    llvm::SmallVector<mlir::scf::ForOp> loops_to_tile = {for_op};
    llvm::SmallVector<mlir::Value> tile_sizes = {tile_size_val};

    // The tile function performs strip-mining by creating an outer loop
    // that iterates over blocks and an inner loop that iterates within blocks
    auto inner_loops = mlir::tile(loops_to_tile, tile_sizes, for_op);

    // Propagate loop_type attribute from original loop to inner point loop
    if (!inner_loops.empty()) {
      propagateLoopTypeAttribute(loops_to_tile, inner_loops);
    }

    if (normalizeLoops && !inner_loops.empty()) {
      // Normalize the loops
      llvm::SmallVector<mlir::scf::ForOp> loops;
      loops.push_back(for_op);
      assert(inner_loops.size() == 1);
      loops.push_back(inner_loops.front());
      normalizeSCFLoops(inner_loops, rewriter);
    }
  }

  // Strip-mine a loop in dynamic mode with LoopTilingAnalysis
  void stripMineLoopDynamic(mlir::scf::ForOp for_op,
                            mlir::ktdf::LoopTilingAnalysis& analysis,
                            mlir::IRRewriter& rewriter) {
    const auto* metadata = analysis.getMetadataForLoop(for_op);

    // Loop should have metadata and be a strip-mining candidate
    // since we checked this before calling this function
    assert(metadata && metadata->isStripMiningCandidate() &&
           "Expected loop to be a strip-mining candidate");

    mlir::Location loc = for_op.getLoc();

    // Materialize tiling.reserve_size op above this loop
    rewriter.setInsertionPoint(for_op);

    // Create ktdf.tiling.reserve_size op with metadata attributes
    auto reserve_size_op = mlir::ktdf::TilingReserveSizeOp::create(
        rewriter, loc, rewriter.getIndexType(),
        rewriter.getIndexAttr(metadata->min_value),
        rewriter.getIndexAttr(metadata->divisibility));

    mlir::Value tile_size_val = reserve_size_op.getResult();

    // Use MLIR's tile utility to strip-mine this single loop
    llvm::SmallVector<mlir::scf::ForOp> loops_to_tile = {for_op};
    llvm::SmallVector<mlir::Value> tile_sizes = {tile_size_val};

    // The tile function performs strip-mining
    auto inner_loops = mlir::tile(loops_to_tile, tile_sizes, for_op);

    // Propagate loop_type attribute from original loop to inner point loop
    if (!inner_loops.empty()) {
      propagateLoopTypeAttribute(loops_to_tile, inner_loops);
    }

    if (normalizeLoops && !inner_loops.empty()) {
      // Normalize the loops
      llvm::SmallVector<mlir::scf::ForOp> loops;
      loops.push_back(for_op);
      assert(inner_loops.size() == 1);
      loops.push_back(inner_loops.front());
      normalizeSCFLoops(loops, rewriter);
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createStripMineSCFForLoopsPass() {
  return std::make_unique<StripMineSCFForLoopsPass>();
}

// Made with Bob
