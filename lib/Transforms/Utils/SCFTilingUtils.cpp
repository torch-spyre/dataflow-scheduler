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

#include "dataflow-scheduler/Transforms/Utils/SCFTilingUtils.h"

#include "dataflow-scheduler/Dialect/KTDF/KTDFAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"

using namespace scheduler;

void scheduler::sinkAffineApplyOpsToUses(mlir::scf::ForOp loop,
                                         mlir::IRRewriter& rewriter) {
  // Collect all affine.apply operations in the loop body
  llvm::SmallVector<mlir::affine::AffineApplyOp> affine_ops;
  loop.getBody()->walk([&](mlir::affine::AffineApplyOp affine_op) {
    affine_ops.push_back(affine_op);
  });

  // For each affine.apply, materialize it before each use
  for (mlir::affine::AffineApplyOp affine_op : affine_ops) {
    if (affine_op->use_empty()) continue;

    // Collect all uses first (before we start modifying)
    llvm::SmallVector<mlir::OpOperand*> uses;
    for (mlir::OpOperand& use : affine_op->getUses()) {
      uses.push_back(&use);
    }

    // For each use, clone the affine.apply right before the using operation
    for (mlir::OpOperand* use : uses) {
      mlir::Operation* user = use->getOwner();

      // Clone the affine.apply operation before the user
      rewriter.setInsertionPoint(user);
      mlir::affine::AffineApplyOp cloned_op =
          mlir::cast<mlir::affine::AffineApplyOp>(rewriter.clone(*affine_op));

      // Replace this specific use with the cloned operation
      use->set(cloned_op.getResult());
    }

    // Erase the original affine.apply since all uses have been replaced
    rewriter.eraseOp(affine_op);
  }
}

void scheduler::normalizeSCFLoops(llvm::ArrayRef<mlir::scf::ForOp> loops,
                                  mlir::IRRewriter& rewriter) {
  // For each loop, normalize it to start at 0 and denormalize the IV
  for (mlir::scf::ForOp loop : loops) {
    // Get the original lower bound and step before normalization
    mlir::Value orig_lb = loop.getLowerBound();
    mlir::Value orig_step = loop.getStep();

    // Skip loops that are already normalized (lb=0, step=1)
    if (mlir::getConstantIntValue(orig_lb) == 0 &&
        mlir::getConstantIntValue(orig_step) == 1)
      continue;
    mlir::Location loc = loop.getLoc();

    // Use MLIR's emitNormalizedLoopBounds to compute normalized bounds
    rewriter.setInsertionPoint(loop);
    mlir::Value ub = loop.getUpperBound();
    auto normalized_range =
        mlir::emitNormalizedLoopBounds(rewriter, loc, orig_lb, ub, orig_step);

    // Update the loop to use normalized bounds (0 to size, step 1)
    loop.setLowerBound(mlir::getValueOrCreateConstantIndexOp(
        rewriter, loc, normalized_range.offset));
    loop.setUpperBound(mlir::getValueOrCreateConstantIndexOp(
        rewriter, loc, normalized_range.size));
    loop.setStep(mlir::getValueOrCreateConstantIndexOp(
        rewriter, loc, normalized_range.stride));

    // Use MLIR's denormalizeInductionVariable to replace uses of the
    // normalized IV with: orig_lb + iv * orig_step
    rewriter.setInsertionPointToStart(loop.getBody());
    mlir::denormalizeInductionVariable(rewriter, loc, loop.getInductionVar(),
                                       orig_lb, orig_step);

    // Post-process: sink affine.apply operations closer to their uses
    // This moves the denormalization computation from the loop start to just
    // before the first use, which is especially beneficial when uses are
    // deeply nested (e.g., inside ktdf.pipeline blocks)
    sinkAffineApplyOpsToUses(loop, rewriter);
  }
}

void scheduler::propagateLoopTypeAttribute(
    llvm::ArrayRef<mlir::scf::ForOp> original_loops,
    llvm::ArrayRef<mlir::scf::ForOp> inner_loops) {
  // Check if we have matching number of loops
  if (original_loops.size() != inner_loops.size()) {
    return;
  }

  // For each pair of original and inner loop, propagate the loop_type attribute
  for (size_t i = 0; i < original_loops.size(); ++i) {
    mlir::scf::ForOp original_loop = original_loops[i];
    mlir::scf::ForOp inner_loop = inner_loops[i];

    // Check if the original loop has a loop_type attribute
    if (auto loop_type_attr =
            original_loop->getAttrOfType<mlir::ktdf::LoopTypeAttr>(
                "loop_type")) {
      // Propagate the attribute to the inner loop
      inner_loop->setAttr("loop_type", loop_type_attr);
    }
  }
}

// Made with Bob
