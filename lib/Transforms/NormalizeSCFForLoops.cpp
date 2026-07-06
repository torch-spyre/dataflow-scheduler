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

#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/SCFTilingUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_NORMALIZESCFFORLOOPSPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

struct NormalizeSCFForLoopsPass
    : public impl::NormalizeSCFForLoopsPassBase<NormalizeSCFForLoopsPass> {
  using NormalizeSCFForLoopsPassBase<
      NormalizeSCFForLoopsPass>::NormalizeSCFForLoopsPassBase;

  void runOnOperation() override {
    mlir::Operation* op = getOperation();
    mlir::IRRewriter rewriter(op->getContext());

    llvm::SmallVector<mlir::scf::ForOp> loops;
    op->walk([&](mlir::scf::ForOp for_op) { loops.push_back(for_op); });

    normalizeSCFLoops(loops, rewriter);
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createNormalizeSCFForLoopsPass() {
  return std::make_unique<NormalizeSCFForLoopsPass>();
}

// Made with Bob
