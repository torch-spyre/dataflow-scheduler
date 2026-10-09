//===-- GeneralizeLinalgOps.cpp ---------------------------------*- c++ -*-===//
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

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <utility>

#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"  // IWYU pragma: keep

#define PASS_NAME "generalize-linalg-ops"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Generalize Linalg Ops pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_GENERALIZELINALGOPSPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

/// The upstream generalization pattern, leaving 'linalg.fill' named.
struct GeneralizeAllButFill : mlir::linalg::LinalgGeneralizationPattern {
  using LinalgGeneralizationPattern::LinalgGeneralizationPattern;

  auto matchAndRewrite(mlir::linalg::LinalgOp op,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    if (mlir::isa<mlir::linalg::FillOp>(op.getOperation())) {
      return rewriter.notifyMatchFailure(op, "linalg.fill is kept named");
    }
    return LinalgGeneralizationPattern::matchAndRewrite(op, rewriter);
  }
};

struct GeneralizeLinalgOpsPass
    : public impl::GeneralizeLinalgOpsPassBase<GeneralizeLinalgOpsPass> {
  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<GeneralizeAllButFill>(patterns.getContext());
    if (failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
