//===-- GlobalStageDAG.h ----------------------------------------*- c++ -*-===//
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
// Global Stage DAG Analysis
//
// Provides a flat directed acyclic graph over leaf StageOps spanning all
// nesting levels of a pipeline hierarchy. Edges are derived from token flow
// and stitched across pipeline boundaries by resolving root/leaf stages of
// each nested pipeline.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_GLOBALSTAGEDAG_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_GLOBALSTAGEDAG_H_

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf {

// TODO(kfaf): Turn this into a managed MLIR analysis.

/// Directed graph of stage dependencies keyed on Operation*.
/// In the per-pipeline use case, nodes are direct-child StageOps.
/// In the global use case, nodes are exclusively leaf StageOps
/// (stages containing no nested PipelineOp).
struct StageDependencyDAG {
  DenseMap<Operation*, SmallVector<Operation*, 4>> predecessors;
  DenseMap<Operation*, SmallVector<Operation*, 4>> successors;
};

/// Build a flat stage DAG spanning all nesting levels of all pipelines in
/// func. Nodes are leaf StageOps only. Edges are stitched across pipeline
/// boundaries by resolving the root/leaf stages of each nested pipeline.
auto buildGlobalStageDAG(Operation* root, StageDependencyDAG& global_dag)
    -> LogicalResult;

/// Build a stage dependency DAG from the token flow of a flat stage list.
/// Nodes are the provided stages; edges connect token producers to consumers.
auto analyzeStageDependencies(ArrayRef<StageOp> stages, StageDependencyDAG& dag)
    -> LogicalResult;

/// Topologically sort stages according to the dependency DAG.
/// Returns stages in execution order (sources first, sinks last).
auto topologicalSortStages(ArrayRef<StageOp> stages,
                           const StageDependencyDAG& dag,
                           SmallVectorImpl<StageOp>& sorted_stages)
    -> LogicalResult;

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_GLOBALSTAGEDAG_H_
