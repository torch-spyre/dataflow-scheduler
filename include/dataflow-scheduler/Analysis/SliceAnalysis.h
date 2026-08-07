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

namespace scheduler {

/// Implements a backward dataflow SSA slice analysis.
///
/// This analysis will (cache and) return all immediate (control flow) SSA value
/// predecessors for a given input.
class BackwardSliceAnalysis {
  using set_type = llvm::SmallPtrSet<mlir::Value, 2>;

 public:
  /// Indicates the set of predecessors of an SSA value.
  struct Predecessors {
    /// Initializes an open set (lower bound) of @p values .
    [[nodiscard]] static auto lowerBound(mlir::ValueRange values = {})
        -> Predecessors {
      return Predecessors(false, values);
    }
    /// Initializes a closed set (exhaustive) of @p values .
    [[nodiscard]] static auto exhaustive(mlir::ValueRange values = {})
        -> Predecessors {
      return Predecessors(true, values);
    }

    /// Initializes an empty lower bound.
    /*implicit*/ Predecessors() = default;

    /// Updates this set to include @p values that are @p is_exhaustive .
    void unite(mlir::ValueRange values, bool is_exhaustive = true) {
      is_exhaustive_ &= is_exhaustive;
      values_.insert_range(values);
    }
    /// Updates this set to include @p rhs .
    void unite(const Predecessors& rhs) {
      is_exhaustive_ &= rhs.is_exhaustive_;
      values_.insert_range(rhs.values_);
    }

    /// Determines whether the set of predecessor values is known to be closed.
    ///
    /// If `true`, then there are no predecessors besides those enumerated by
    /// this container. If `false`, then there were types of control flow that
    /// could not be followed, possibly due to unregistered operations.
    [[nodiscard]] auto isExhaustive() const -> bool { return is_exhaustive_; }

    /// Determines whether there are no known and unknown predecessors.
    [[nodiscard]] auto isKnownEmpty() const -> bool {
      return isExhaustive() && values_.empty();
    }

    /// Gets the known predecessor values.
    [[nodiscard]] auto getValues() const -> const set_type& { return values_; }

    //===------------------------------------------------------------------===//
    // Container interface
    //===------------------------------------------------------------------===//

    using value_type = set_type::value_type;
    using size_type = set_type::size_type;
    using iterator = set_type::const_iterator;

    [[nodiscard]] auto empty() const -> bool { return values_.empty(); }
    [[nodiscard]] auto size() const -> size_type { return values_.size(); }

    [[nodiscard]] auto begin() const -> iterator { return values_.begin(); }
    [[nodiscard]] auto end() const -> iterator { return values_.end(); }

   private:
    friend class BackwardSliceAnalysis;

    explicit Predecessors(bool is_exhaustive, mlir::ValueRange values)
        : is_exhaustive_(is_exhaustive), values_(llvm::from_range, values) {}

    bool is_exhaustive_ = false;
    set_type values_;
  };

  using key_type = mlir::Value;
  using mapped_type = Predecessors;
  using map_type = llvm::DenseMap<key_type, mapped_type>;

  // Allow construction as an MLIR analysis.
  explicit BackwardSliceAnalysis(mlir::Operation* /*op*/ = nullptr);

  /// Gets the immediate @p predecessors of @p value .
  ///
  /// @param            value         Value to query the predecessors of.
  /// @param  [in,out]  is_exhaustive Whether the result is exhaustive.
  /// @param  [out]     predecessors  Set of predecessors.
  void getPredecessors(mlir::Value value, bool& is_exhaustive,
                       llvm::SmallPtrSetImpl<mlir::Value>& predecessors);
  /// Gets the immediate @p predecessors of @p value .
  void getPredecessors(mlir::Value value, Predecessors& predecessors) {
    getPredecessors(value, predecessors.is_exhaustive_, predecessors.values_);
  }
  /// Gets the immediate predecessors of @p value .
  [[nodiscard]] auto getPredecessors(mlir::Value value) -> Predecessors {
    Predecessors result;
    getPredecessors(value, result);
    return result;
  }

  /// Gets the immediate control flow predecessors of @p value .
  [[nodiscard]] auto getControlFlowPredecessors(mlir::Value value)
      -> const Predecessors&;

 private:
  map_type control_flow_;
};

/// Base class for implementing a forward slice analysis.
///
/// This analysis will determine whether SSA values are reachable from forward
/// dataflow starting with an initial set of values.
class ForwardSlice {
 public:
  /// Result of a slice membership check.
  enum class Result : char {
    /// Value is not in the slice.
    NoContain = 0,
    /// Value might be in the slice (lower bound).
    MayContain = 0b01,
    /// Value must be in the slice (upper bound).
    MustContain = 0b11,
  };

  using key_type = mlir::Value;
  using mapped_type = Result;
  using map_type = llvm::DenseMap<key_type, mapped_type>;

  /// Initializes a ForwardSlice using @p backward and the initial @p values .
  explicit ForwardSlice(BackwardSliceAnalysis& backward,
                        mlir::ValueRange values);

  /// Inserts additional @p values into the slice.
  ///
  /// @retval false   @p values were already contained.
  /// @retval true    New values were added, and the cache was invalidated.
  auto insert(mlir::ValueRange values) -> bool;

  /// Determines whether @p value is in the slice.
  auto contains(mlir::Value value) -> Result;

 private:
  BackwardSliceAnalysis& backward_;
  map_type cache_;
};

/// Implements a ForwardSlice based on the loop variables of an operation.
///
/// Loop variables are induction variables and inter-iteration dependencies
/// carried by region arguments, as advertised by the mlir::LoopLikeOpInterface.
/// If the operation does not implement this interface, the slice is empty.
class LoopSliceAnalysis : public ForwardSlice {
 public:
  explicit LoopSliceAnalysis(mlir::Operation* op,
                             BackwardSliceAnalysis& backward);
  // Allow construction as an MLIR analysis.
  explicit LoopSliceAnalysis(mlir::Operation* op,
                             mlir::AnalysisManager& analyses);
};

/// Backport of llvm-project/pull/188758.
[[nodiscard]]
auto getControlFlowPredecessors(mlir::Value value)
    -> std::optional<llvm::SmallVector<mlir::Value>>;

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_SLICEANALYSIS_H_
