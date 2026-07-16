//===-- StageGrouping.h -----------------------------------------*- c++ -*-===//
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
// Stage Grouping Analysis
//
// This analysis determines which stages in a pipeline can be coarsened together
// based on Loop distribution legality (data dependencies).
// Scalar dependencies are loop-fission preventing if their corresponding memory
// buffers cannot be expanded to turn them into non-scalar dependencies.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_STAGEGROUPING_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_STAGEGROUPING_H_

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"

namespace mlir::ktdf {

/// Represents a group of stages that should be coarsened together
struct StageGroup {
  /// Add a stage to this group
  void addStage(StageOp stage) { stages_.push_back(stage); }

  /// Check if this is a single-stage group
  bool isSingleStage() const { return stages_.size() == 1; }

  /// Get the first stage in the group
  StageOp getFirstStage() const {
    return stages_.empty() ? nullptr : stages_.front();
  }

  /// Get the last stage in the group
  StageOp getLastStage() const {
    return stages_.empty() ? nullptr : stages_.back();
  }

  /// Get all stages in the group
  const SmallVector<StageOp>& getStages() const { return stages_; }

 private:
  /// Stages in this group (in program order)
  SmallVector<StageOp> stages_;
};

/// Stage Grouping Analysis
///
/// Analyzes a pipeline to determine which stages can be coarsened together.
/// This is based on loop distribution legality rules and resource sharing.
class StageGroupingAnalysis {
 public:
  // TODO(kfaf): Turn this into a managed MLIR analysis.

  /// Construct analysis for a given pipeline with memory tree
  explicit StageGroupingAnalysis(
      PipelineOp pipeline, const scheduler::arch_view::MemoryTree& memory_tree);

  /// Construct analysis for stages with memory tree
  explicit StageGroupingAnalysis(
      ArrayRef<StageOp> stages,
      const scheduler::arch_view::MemoryTree& memory_tree);

  /// Get all stage groups
  const SmallVector<StageGroup>& getGroups() const { return groups_; }

  /// Get the group that contains a given stage
  const StageGroup* getGroupForStage(StageOp stage) const;

  /// Check if two stages are in the same group
  bool areInSameGroup(StageOp stage1, StageOp stage2) const;

  /// Print the grouping results to the given output stream
  void print(raw_ostream& os) const;

  /// Print the grouping results for debugging (to llvm::outs())
  void dump() const;

 private:
  void init(ArrayRef<StageOp> stages);

  /// Determine if two stages should be grouped together
  bool shouldGroupTogether(StageOp stage1, StageOp stage2);

  SmallVector<StageGroup> groups_;
  DenseMap<Operation*, unsigned> stage_to_group_;
  const scheduler::arch_view::MemoryTree& memory_tree_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_STAGEGROUPING_H_
