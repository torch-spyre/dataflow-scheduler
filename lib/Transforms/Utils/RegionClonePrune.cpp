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

#include "dataflow-scheduler/Transforms/Utils/RegionClonePrune.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define PRUNE_ANCHOR_MARKER "prune_anchor"

using namespace scheduler;

namespace {

/// What a cloned op is kept for.
///
/// `required` is the anchor and everything a kept op reads. `structural` is
/// what holds a required op in a region, and the terminator of such a region.
/// An op in neither is erased.
struct Marking {
  llvm::DenseSet<mlir::Operation*> required;
  llvm::DenseSet<mlir::Operation*> structural;

  bool keeps(mlir::Operation* op) const {
    return required.contains(op) || structural.contains(op);
  }
};

/// Marks what a clone rooted at \p prune_root keeps to hold \p anchor.
///
/// The two sets grow together: an op joins `required` through what reads it and
/// `structural` through what it holds, and each rule feeds the other -- a
/// surviving loop needs its bounds, and a surviving region needs whatever its
/// terminator hands back. So this runs to a fixpoint and nothing is decided
/// from a half-built set. Which loops the clone can then do without is
/// canDemote()'s question, asked once this is settled.
Marking markClone(mlir::Operation* anchor, mlir::Operation* prune_root) {
  Marking marks;
  llvm::SmallVector<mlir::Operation*> worklist;

  auto inClone = [&](mlir::Operation* op) {
    return op != nullptr && prune_root->isProperAncestor(op);
  };
  auto require = [&](mlir::Operation* op) {
    if (inClone(op) && marks.required.insert(op).second) worklist.push_back(op);
  };
  auto structure = [&](mlir::Operation* op) {
    if (inClone(op) && marks.structural.insert(op).second)
      worklist.push_back(op);
  };

  require(anchor);
  while (!worklist.empty()) {
    mlir::Operation* op = worklist.pop_back_val();

    // Whatever it reads: the anchor's arithmetic, a loop's bounds, the
    // condition of an scf.if.
    for (mlir::Value operand : op->getOperands()) {
      require(operand.getDefiningOp());
    }

    if (marks.required.contains(op)) {
      for (mlir::Operation* parent = op->getParentOp(); inClone(parent);
           parent = parent->getParentOp()) {
        structure(parent);
      }
    }

    // A region that stays keeps handing back what it did, so its terminator
    // stays and what the terminator reads is required.
    if (marks.structural.contains(op)) {
      for (mlir::Region& region : op->getRegions()) {
        for (mlir::Block& block : region) {
          if (!block.empty()) structure(block.getTerminator());
        }
      }
    }
  }

  return marks;
}

/// Whether the clone can do without \p loop, its body running once in place of
/// every iteration.
///
/// It can where the iterations are all the same one and nothing depends on
/// there having been several: nothing kept reads the variable, the loop carries
/// no value out, and nothing required inside it besides \p anchor touches
/// memory.
///
/// The anchor is the exception on purpose. Collapsing repeats of one transfer
/// is what hoisting an invariant transfer does in the first place.
bool canDemote(mlir::scf::ForOp loop, mlir::Operation* anchor,
               const Marking& marks) {
  if (loop->getNumResults() != 0) {
    return false;
  }

  // Kept rather than required: an inner loop that stays may take this variable
  // as a bound, and it is the loop rather than any result of it that reads it.
  for (mlir::Operation* user : loop.getInductionVar().getUsers()) {
    if (marks.keeps(user)) return false;
  }

  bool once = true;
  loop.getBody()->walk([&](mlir::Operation* op) {
    if (op == anchor || !marks.required.contains(op)) return;
    if (!mlir::isMemoryEffectFree(op)) once = false;
  });
  return once;
}

}  // namespace

mlir::Operation* scheduler::cloneRegionAndPruneToAnchor(
    mlir::Region& source_region, mlir::Block* target_block,
    mlir::Block::iterator ip, mlir::Operation* keep_anchor,
    mlir::IRMapping& value_map) {
  mlir::OpBuilder builder(target_block, ip);

  mlir::Operation* prune_root = target_block->getParentOp();
  assert(prune_root && "target_block must have a parent op");

  // Tag keep_anchor so its clone can be found by attribute after cloning,
  // avoiding a full origin_map traversal of the cloned subtree.
  keep_anchor->setAttr(PRUNE_ANCHOR_MARKER, builder.getI32IntegerAttr(1));

  // Clone top-level ops from source_region's entry block.
  mlir::Block& source_block = source_region.front();
  llvm::SmallVector<mlir::Operation*> cloned_top_level;
  for (mlir::Operation& op : source_block) {
    cloned_top_level.push_back(builder.clone(op, value_map));
  }

  // Remove the marker from the original now that cloning is done.
  keep_anchor->removeAttr(PRUNE_ANCHOR_MARKER);

  // Find the cloned anchor by locating the marker in the cloned tree.
  mlir::Operation* cloned_anchor = nullptr;
  for (mlir::Operation* top : cloned_top_level) {
    top->walk([&](mlir::Operation* op) {
      if (!op->hasAttr(PRUNE_ANCHOR_MARKER)) return mlir::WalkResult::advance();
      cloned_anchor = op;
      op->removeAttr(PRUNE_ANCHOR_MARKER);
      return mlir::WalkResult::interrupt();
    });
    if (cloned_anchor) break;
  }
  assert(cloned_anchor && "anchor not found in cloned region");

  const Marking marks = markClone(cloned_anchor, prune_root);

  // Collect erasure candidates in post-order (children before parents), so a
  // loop is erased after whatever was inside it.
  llvm::SmallVector<mlir::Operation*> erasable;
  for (mlir::Operation* top : cloned_top_level) {
    top->walk([&](mlir::Operation* op) {
      // A loop the clone can do without hands its body to the block around it
      // and goes; one that stays holds whatever the marking left in it.
      if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        if (marks.structural.contains(op) &&
            canDemote(loop, cloned_anchor, marks)) {
          mlir::Block* body = &op->getRegion(0).front();
          op->getBlock()->getOperations().splice(
              mlir::Block::iterator(op), body->getOperations(), body->begin(),
              body->getTerminator()->getIterator());
          erasable.push_back(op);
        }
        return;
      }

      if (marks.keeps(op)) return;
      if (op->hasTrait<mlir::OpTrait::IsTerminator>()) return;
      erasable.push_back(op);
    });
  }

  // Nothing the clone keeps may read what goes. The marking is what makes that
  // true; this says so if a rule is ever missed, rather than leaving the clone
  // holding a null operand.
  for (mlir::Operation* op : erasable) {
    for (mlir::Value result : op->getResults()) {
      for (mlir::Operation* user : result.getUsers()) {
        (void)user;
        assert(!marks.keeps(user) && "pruned op is still read by a kept op");
      }
    }
  }

  // The rest of the uses are among the ops going, in an order erasing them one
  // at a time does not respect.
  for (mlir::Operation* op : erasable) {
    op->dropAllUses();
  }
  for (mlir::Operation* op : erasable) {
    op->erase();
  }

  return cloned_anchor;
}
