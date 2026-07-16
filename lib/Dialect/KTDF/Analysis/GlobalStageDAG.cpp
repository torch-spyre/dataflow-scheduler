//===-- GlobalStageDAG.cpp --------------------------------------*- c++ -*-===//
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
// Implements the global stage DAG analysis.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Dialect/KTDF/Analysis/GlobalStageDAG.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/DebugLog.h>

#include <functional>
#include <map>
#include <queue>

#define DEBUG_TYPE "global-stage-dag"

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// analyzeStageDependencies
//===----------------------------------------------------------------------===//

auto mlir::ktdf::analyzeStageDependencies(ArrayRef<StageOp> stages,
                                          StageDependencyDAG& dag)
    -> LogicalResult {
  LDBG(1) << "Analyze stage dependencies";

  for (auto stage : stages) {
    dag.predecessors[stage.getOperation()] = {};
    dag.successors[stage.getOperation()] = {};
  }

  // Build a Value → producer map so each depends_in lookup is O(1).
  llvm::DenseMap<Value, Operation*> token_producer_map;
  for (auto stage : stages) {
    for (auto operand : stage.getDependsOut()) {
      token_producer_map[operand] = stage.getOperation();
    }
  }

  for (auto stage : stages) {
    for (auto input_token : stage.getDependsIn()) {
      auto it = token_producer_map.find(input_token);
      if (it != token_producer_map.end()) {
        Operation* producer = it->second;
        dag.predecessors[stage.getOperation()].push_back(producer);
        dag.successors[producer].push_back(stage.getOperation());
      }
    }
  }

  // Cycle detection via DFS.
  std::map<Operation*, int> color;  // 0=white, 1=gray, 2=black
  std::function<bool(Operation*)> hasCycle;
  hasCycle = [&](Operation* op) -> bool {
    color[op] = 1;
    for (auto* succ : dag.successors[op]) {
      if (color[succ] == 1) return true;
      if (color[succ] == 0 && hasCycle(succ)) return true;
    }
    color[op] = 2;
    return false;
  };

  for (auto stage : stages) {
    if (color[stage.getOperation()] == 0) {
      if (hasCycle(stage.getOperation())) {
        return stage.emitError(
            "Circular token dependency detected in pipeline");
      }
    }
  }

  LLVM_DEBUG({
    for (auto stage : stages) {
      llvm::dbgs() << "  Stage " << stage.getOperation() << ":\n"
                   << "    Predecessors: ";
      for (auto* pred : dag.predecessors[stage.getOperation()])
        llvm::dbgs() << pred << " ";
      llvm::dbgs() << "\n    Successors: ";
      for (auto* succ : dag.successors[stage.getOperation()])
        llvm::dbgs() << succ << " ";
      llvm::dbgs() << "\n";
    }
  });

  return success();
}

//===----------------------------------------------------------------------===//
// topologicalSortStages
//===----------------------------------------------------------------------===//

auto mlir::ktdf::topologicalSortStages(ArrayRef<StageOp> stages,
                                       const StageDependencyDAG& dag,
                                       SmallVectorImpl<StageOp>& sorted_stages)
    -> LogicalResult {
  LDBG(1) << "Topologically sort stages";

  if (stages.empty()) return success();

  std::map<Operation*, int> in_degree;
  std::queue<Operation*> queue;

  for (auto stage : stages) {
    Operation* op = stage.getOperation();
    auto it = dag.predecessors.find(op);
    in_degree[op] = (it != dag.predecessors.end()) ? it->second.size() : 0;
    if (in_degree[op] == 0) queue.push(op);
  }

  while (!queue.empty()) {
    Operation* current = queue.front();
    queue.pop();

    if (auto stage = dyn_cast<StageOp>(current)) sorted_stages.push_back(stage);

    auto succ_it = dag.successors.find(current);
    if (succ_it != dag.successors.end()) {
      for (auto* succ : succ_it->second) {
        if (--in_degree[succ] == 0) queue.push(succ);
      }
    }
  }

  if (sorted_stages.size() != stages.size()) return failure();

  LLVM_DEBUG({
    llvm::dbgs() << "Topological sort result:\n";
    for (size_t i = 0; i < sorted_stages.size(); ++i)
      llvm::dbgs() << "  [" << i << "] " << sorted_stages[i].getOperation()
                   << "\n";
  });

  return success();
}

//===----------------------------------------------------------------------===//
// getRootStages / getLeafStages (file-private helpers)
//===----------------------------------------------------------------------===//
//
// Example — given this nesting:
//
//   PipelineOp (outer)
//     StageOp A  ──token──►  StageOp B  ──token──►  StageOp C
//                             └─ PipelineOp (inner)
//                                  StageOp B1 ──token──► StageOp B2
//                                  StageOp B3  (no deps)
//
//   getRootStages(A) = {A}          (A has no nested pipeline)
//   getRootStages(B) = {B1, B3}     (B1 and B3 have no predecessors inside B)
//   getRootStages(C) = {C}
//
//   getLeafStages(A) = {A}
//   getLeafStages(B) = {B2, B3}     (B2 and B3 have no successors inside B)
//   getLeafStages(C) = {C}
//
// buildGlobalStageDAG uses these to stitch the outer edges:
//   A → B1, A → B3   (getLeafStages(A) × getRootStages(B))
//   B2 → C, B3 → C   (getLeafStages(B) × getRootStages(C))
//
//===----------------------------------------------------------------------===//

namespace {

/// Returns the recursive first-in-topo-order root StageOps reachable from
/// stage. If stage contains no nested PipelineOp, returns {stage}.
auto getRootStages(StageOp stage) -> SmallVector<StageOp, 4> {
  PipelineOp nested_pipeline;
  stage.walk([&](PipelineOp p) {
    nested_pipeline = p;
    return WalkResult::interrupt();
  });

  if (!nested_pipeline) return {stage};

  llvm::SmallVector<StageOp, 8> inner_stages;
  for (auto& op : nested_pipeline.getBodyRegion().front()) {
    if (auto s = dyn_cast<StageOp>(op)) inner_stages.push_back(s);
  }

  StageDependencyDAG inner_dag;
  if (failed(analyzeStageDependencies(inner_stages, inner_dag))) return {stage};

  llvm::SmallVector<StageOp> result;
  for (auto inner : inner_stages) {
    auto it = inner_dag.predecessors.find(inner.getOperation());
    if (it == inner_dag.predecessors.end() || it->second.empty()) {
      auto roots = getRootStages(inner);
      result.append(roots.begin(), roots.end());
    }
  }
  return result;
}

/// Returns the recursive last-in-topo-order leaf StageOps reachable from
/// stage. If stage contains no nested PipelineOp, returns {stage}.
auto getLeafStages(StageOp stage) -> llvm::SmallVector<StageOp, 4> {
  PipelineOp nested_pipeline;
  stage.walk([&](PipelineOp p) {
    nested_pipeline = p;
    return WalkResult::interrupt();
  });

  if (!nested_pipeline) return {stage};

  llvm::SmallVector<StageOp, 8> inner_stages;
  for (auto& op : nested_pipeline.getBodyRegion().front()) {
    if (auto s = dyn_cast<StageOp>(op)) inner_stages.push_back(s);
  }

  StageDependencyDAG inner_dag;
  if (failed(analyzeStageDependencies(inner_stages, inner_dag))) return {stage};

  llvm::SmallVector<StageOp, 4> result;
  for (auto inner : inner_stages) {
    auto it = inner_dag.successors.find(inner.getOperation());
    if (it == inner_dag.successors.end() || it->second.empty()) {
      auto leaves = getLeafStages(inner);
      result.append(leaves.begin(), leaves.end());
    }
  }
  return result;
}

}  // namespace

//===----------------------------------------------------------------------===//
// buildGlobalStageDAG
//===----------------------------------------------------------------------===//

auto mlir::ktdf::buildGlobalStageDAG(Operation* root,
                                     StageDependencyDAG& global_dag)
    -> LogicalResult {
  llvm::SmallVector<PipelineOp, 8> pipelines;
  root->walk([&](PipelineOp p) { pipelines.push_back(p); });

  for (auto pipeline : llvm::reverse(pipelines)) {
    auto stages_range = pipeline.getStages();
    llvm::SmallVector<StageOp, 8> direct_stages(stages_range.begin(),
                                                stages_range.end());
    if (direct_stages.empty()) continue;

    StageDependencyDAG local_dag;
    if (failed(analyzeStageDependencies(direct_stages, local_dag)))
      return failure();

    for (auto& [producer_op, successors] : local_dag.successors) {
      auto producer_stage = dyn_cast<StageOp>(producer_op);
      if (!producer_stage) continue;
      auto producer_leaves = getLeafStages(producer_stage);

      for (auto* consumer_op : successors) {
        auto consumer_stage = dyn_cast<StageOp>(consumer_op);
        if (!consumer_stage) continue;
        auto consumer_roots = getRootStages(consumer_stage);

        for (auto leaf : producer_leaves) {
          for (auto root : consumer_roots) {
            global_dag.successors[leaf.getOperation()].push_back(
                root.getOperation());
            global_dag.predecessors[root.getOperation()].push_back(
                leaf.getOperation());
          }
        }
      }
    }
  }

  return success();
}
