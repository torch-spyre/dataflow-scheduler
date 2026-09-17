//===-- SliceAnalysis.h -----------------------------------------*- c++ -*-===//
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
// SSA slice analyses
//
// A slice through an SSA program contains all values that are reachable (either
// by going forward or backward) via def-use chains. In MLIR, there is also
// (structured) control flow, i.e., there are operations which may pass values
// along edges defined by other means. The slice analyses provide cached queries
// for the built-in MLIR interfaces that define these relationships.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_SLICEANALYSIS_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_SLICEANALYSIS_H_

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/Analysis/SliceWalk.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/LLVM.h>

#include <optional>

#include "Predecessors.h"
#include "dataflow-scheduler/Analysis/Predecessors.h"
namespace scheduler {

/// Base class for implementing a forward slice analysis.
///
/// This analysis will determine whether SSA values are reachable from forward
/// dataflow starting with an initial set of values.
class ForwardSlice {
 public:
  using key_type = mlir::Value;
  using mapped_type = MembershipResult;
  using map_type = llvm::DenseMap<key_type, mapped_type>;

  /// Initializes a ForwardSlice using @p preds and the initial @p values .
  explicit ForwardSlice(PredecessorInfo& preds, mlir::ValueRange values);

  /// Inserts additional @p values into the slice.
  ///
  /// @retval false   @p values were already contained.
  /// @retval true    New values were added, and the cache was invalidated.
  auto insert(mlir::ValueRange values) -> bool;

  /// Determines whether @p value is in the slice.
  auto contains(mlir::Value value) -> MembershipResult;

 private:
  PredecessorInfo& preds_;
  map_type cache_;
};

/// Implements a ForwardSlice based on the loop variables of an operation.
///
/// Loop variables are induction variables and inter-iteration dependencies
/// carried by region arguments, as advertised by the mlir::LoopLikeOpInterface.
/// If the operation does not implement this interface, the slice is empty.
class LoopSliceAnalysis : public ForwardSlice {
 public:
  explicit LoopSliceAnalysis(mlir::Operation* op, PredecessorInfo& preds);
  // Allow construction as an MLIR analysis.
  explicit LoopSliceAnalysis(mlir::Operation* op,
                             mlir::AnalysisManager& analyses);
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_SLICEANALYSIS_H_
