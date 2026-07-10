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
// AffineMinCanonicalization: Canonicalize arith operations into affine.min
//
// This pass transforms the pattern:
//   %add = arith.addi %iv, %step
//   %min = arith.minsi %add, %ub
//   %result = affine.apply ()[%iv, %min] -> (-iv + min)
//
// Into the canonical form:
//   %result = affine.min (%iv)[%ub, %step] -> (ub - iv, step)
//
// Mathematical equivalence:
//   Original: min(iv + step, ub) - iv
//   Canonical: min(ub - iv, step)
//   When iv + step < ub: result = step
//   When iv + step >= ub: result = ub - iv
//
// Behavior with multiple uses:
//   - Only the affine.apply operation is replaced with affine.min
//   - If %add and %min have other uses, they remain in the IR
//   - If %add and %min become dead after replacement, the greedy pattern
//     rewriter automatically erases them via dead code elimination
//   - This ensures the transformation is always safe and beneficial
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "dataflow-scheduler/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define PASS_NAME "affine-min-canonicalization"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Affine Min Canonicalization pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_AFFINEMINCANONICALIZATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Check if an affine map matches the pattern ()[s0, s1] -> (-s0 + s1)
/// or equivalent forms like ()[s0, s1] -> (s1 - s0)
bool isSubtractionMap(mlir::AffineMap map) {
  // Must have 0 dimensions and 2 symbols
  if (map.getNumDims() != 0 || map.getNumSymbols() != 2) {
    return false;
  }

  // Must have exactly 1 result
  if (map.getNumResults() != 1) {
    return false;
  }

  mlir::AffineExpr result = map.getResult(0);

  // Try to match using MLIR's expression comparison
  // Create the expected patterns: both s1 - s0 and -s0 + s1
  mlir::MLIRContext* ctx = map.getContext();
  mlir::AffineExpr s0 = mlir::getAffineSymbolExpr(0, ctx);
  mlir::AffineExpr s1 = mlir::getAffineSymbolExpr(1, ctx);
  mlir::AffineExpr expected1 = s1 - s0;   // s1 - s0
  mlir::AffineExpr expected2 = -s0 + s1;  // -s0 + s1

  bool matches = (result == expected1) || (result == expected2);

  LDBG(1) << "  Comparing expressions:";
  LDBG(1) << "    Result: " << result;
  LDBG(1) << "    Expected1 (s1-s0): " << expected1;
  LDBG(1) << "    Expected2 (-s0+s1): " << expected2;
  LDBG(1) << "    Match: " << (matches ? "YES" : "NO");

  return matches;
}

/// Pattern to canonicalize: arith.addi -> arith.minsi -> affine.apply
/// into affine.min
struct CanonicalizeToAffineMin
    : public mlir::OpRewritePattern<mlir::affine::AffineApplyOp> {
  using OpRewritePattern<mlir::affine::AffineApplyOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::affine::AffineApplyOp applyOp,
      mlir::PatternRewriter& rewriter) const override {
    LDBG(1) << "Checking affine.apply: " << applyOp;
    LDBG(1) << "  Map: " << applyOp.getAffineMap();

    // Check if the map matches our pattern: ()[s0, s1] -> (-s0 + s1)
    if (!isSubtractionMap(applyOp.getAffineMap())) {
      LDBG(1) << "  Map does not match subtraction pattern";
      return mlir::failure();
    }

    // Get operands: should be [%iv, %min]
    if (applyOp.getMapOperands().size() != 2) {
      return mlir::failure();
    }

    mlir::Value iv = applyOp.getMapOperands()[0];
    mlir::Value minResult = applyOp.getMapOperands()[1];

    // Check if minResult comes from arith.minsi
    auto minOp = minResult.getDefiningOp<mlir::arith::MinSIOp>();
    if (!minOp) {
      return mlir::failure();
    }

    // Note: We allow multiple uses of minOp result because the transformation
    // is still beneficial even if the intermediate value is used elsewhere

    // Get minsi operands: should be [%add, %ub] or [%ub, %add]
    mlir::Value lhs = minOp.getLhs();
    mlir::Value rhs = minOp.getRhs();

    mlir::Value addResult;
    mlir::Value ub;

    // Try both orderings
    if (auto addOp = lhs.getDefiningOp<mlir::arith::AddIOp>()) {
      addResult = lhs;
      ub = rhs;
    } else if (auto addOp = rhs.getDefiningOp<mlir::arith::AddIOp>()) {
      addResult = rhs;
      ub = lhs;
    } else {
      return mlir::failure();
    }

    auto addOp = addResult.getDefiningOp<mlir::arith::AddIOp>();

    // Note: We allow multiple uses of addOp result because the transformation
    // is still beneficial even if the intermediate value is used elsewhere

    // Get addi operands: should be [%iv, %step] or [%step, %iv]
    mlir::Value addLhs = addOp.getLhs();
    mlir::Value addRhs = addOp.getRhs();

    mlir::Value step;

    // Check which operand matches the IV
    if (addLhs == iv) {
      step = addRhs;
    } else if (addRhs == iv) {
      step = addLhs;
    } else {
      return mlir::failure();
    }

    LDBG(1) << "Matched canonicalization pattern:";
    LDBG(1) << "  IV: " << iv;
    LDBG(1) << "  Step: " << step;
    LDBG(1) << "  UB: " << ub;
    LDBG(1) << "  Replacing:";
    LDBG(1) << "    " << *addOp;
    LDBG(1) << "    " << *minOp;
    LDBG(1) << "    " << *applyOp;

    // Create the affine.min operation
    mlir::MLIRContext* ctx = applyOp.getContext();

    // Create affine map: (d0)[s0, s1] -> (s0 - d0, s1)
    // This represents: min(ub - iv, step)
    mlir::AffineExpr d0 = mlir::getAffineDimExpr(0, ctx);
    mlir::AffineExpr s0 = mlir::getAffineSymbolExpr(0, ctx);
    mlir::AffineExpr s1 = mlir::getAffineSymbolExpr(1, ctx);

    mlir::AffineExpr expr1 = s0 - d0;  // ub - iv
    mlir::AffineExpr expr2 = s1;       // step

    mlir::AffineMap minMap = mlir::AffineMap::get(
        /*dimCount=*/1,
        /*symbolCount=*/2, {expr1, expr2}, ctx);

    // Create affine.min with operands: (%iv)[%ub, %step]
    llvm::SmallVector<mlir::Value> operands = {iv, ub, step};

    auto minMapOp = mlir::affine::AffineMinOp::create(
        rewriter, applyOp.getLoc(), minMap, operands);

    LDBG(1) << "Created affine.min:";
    LDBG(1) << "    " << *minMapOp;

    // Replace the affine.apply with affine.min
    rewriter.replaceOp(applyOp, minMapOp.getResult());

    // The pattern rewriter will automatically clean up dead ops (minOp and
    // addOp) if they have no more uses

    return mlir::success();
  }
};

struct AffineMinCanonicalizationPass
    : public impl::AffineMinCanonicalizationPassBase<
          AffineMinCanonicalizationPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    mlir::MLIRContext* ctx = &getContext();
    mlir::RewritePatternSet patterns(ctx);

    // Add our canonicalization pattern
    patterns.add<CanonicalizeToAffineMin>(ctx);

    // Apply patterns greedily
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    LDBG(1) << "Canonicalization complete";
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createAffineMinCanonicalizationPass() {
  return std::make_unique<AffineMinCanonicalizationPass>();
}

// Made with Bob