//===-- Reuse.cpp -----------------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Analysis/Reuse.h"

#include "dataflow-scheduler/Analysis/InvarianceCheck.h"
#include "dataflow-scheduler/Analysis/WriteSetScan.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/PipelineScope.h"

using namespace mlir;
using namespace mlir::ktdf;

namespace {

/// True iff `v`'s defining op (if any) is located inside `region`
/// (transitively: at any depth, including nested regions).
bool definedInsideRegion(Value v, Region& region) {
  Operation* def = v.getDefiningOp();
  return def && region.isAncestor(def->getParentRegion());
}

/// Walk `scope.loops` innermost-to-outermost. For each loop, check all
/// five conditions. Return the index in `scope.loops` of the outermost
/// loop where every condition still holds, or -1 if even the innermost
/// loop fails. The index doubles as a depth metric: higher = farther
/// out from the pipeline.
int findOutermostLegalTargetDepth(::DataTransferOp transfer,
                                  const ::PipelineEnclosingScope& scope) {
  int best = -1;
  for (int i = 0; i < static_cast<int>(scope.loops.size()); ++i) {
    scf::ForOp loop = scope.loops[i];
    if (!scheduler::transferIsInvariantWrt(transfer, loop.getInductionVar())) {
      break;
    }
    if (scheduler::regionWritesTo(loop.getRegion(), transfer.getSource())) {
      break;
    }
    if (scheduler::regionWritesTo(loop.getRegion(), transfer.getDestination(),
                                  /*ignore=*/transfer.getOperation())) {
      break;
    }
    if (definedInsideRegion(transfer.getSource(), loop.getRegion())) {
      break;
    }
    best = i;
  }
  return best;
}

/// Collect the indirect-address-buffer memrefs addressed by the
/// `ind_data_transfer` ops in @p stage .
///
/// A `data_transfer` writing into one of them is the IAB fill for that
/// indirect transfer. The fill must stay in the same stage as the transfer it
/// feeds -- the buffer is local state of the hardware unit running the stage
/// and does not survive a stage boundary -- so it is never a hoist candidate.
/// Naming the buffer rather than the stage keeps unrelated transfers that
/// StageCoarsening merges into this stage hoistable.
llvm::SmallPtrSet<Value, 2> getIndAddrBufs(::StageOp stage) {
  llvm::SmallPtrSet<Value, 2> bufs;
  stage.walk([&](::IndDataTransferOp transfer) {
    if (Value src = transfer.getIndSrcMemref()) {
      bufs.insert(src);
    }
    if (Value dst = transfer.getIndDstMemref()) {
      bufs.insert(dst);
    }
  });
  return bufs;
}

}  // namespace

auto mlir::ktdf::reuse::findFirstCandidate(::PipelineOp pipeline)
    -> std::optional<Candidate> {
  PipelineEnclosingScope scope = getPipelineEnclosingScope(pipeline);
  if (scope.loops.empty()) {
    return std::nullopt;
  }

  // Scan every data_transfer in the pipeline, then pick the one whose
  // legal hoist target is deepest (i.e., farthest out from the pipeline).
  // Picking source-order-first would let an early shallow-target hoist
  // poison the pipeline-enclosing-scope chain for later transfers that
  // could have hoisted further. Picking the deepest target avoids that.
  std::optional<Candidate> best_candidate;
  int best_depth = -1;
  for (Operation& child : pipeline.getBody()->getOperations()) {
    auto stage = dyn_cast<::StageOp>(&child);
    if (!stage) {
      continue;
    }
    const llvm::SmallPtrSet<Value, 2> ind_addr_bufs = getIndAddrBufs(stage);
    stage.walk([&](::DataTransferOp transfer) {
      if (transfer.isDestFifo() ||
          ind_addr_bufs.contains(transfer.getDestination())) {
        return WalkResult::advance();
      }
      int depth = findOutermostLegalTargetDepth(transfer, scope);
      if (depth < 0) {
        return WalkResult::advance();
      }
      if (depth > best_depth) {
        best_depth = depth;
        best_candidate = Candidate{transfer, stage, scope.loops[depth]};
      }
      return WalkResult::advance();
    });
  }
  return best_candidate;
}
