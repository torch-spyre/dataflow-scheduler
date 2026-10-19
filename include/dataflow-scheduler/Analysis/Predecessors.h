//===-- Predecessors.h ------------------------------------------*- c++ -*-===//
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
// SSA value predecessors
//
// Using the SSA def-use chain and the MLIR control flow interfaces, the
// possible sources of an SSA value can be traced back to other SSA values. The
// PredecessorInfo struct caches this information similar to DominanceInfo.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_PREDECESSORS_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_PREDECESSORS_H_

#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>

namespace scheduler {

//===----------------------------------------------------------------------===//
// MembershipResult
//===----------------------------------------------------------------------===//

/// The possible results of a set membership query.
class MembershipResult {
 public:
  enum Kind : unsigned char {
    /// The element is not in the set.
    NoContain = 0,
    /// Membership of the element couldn't be decided.
    MayContain = 0b01,
    /// The element is in the set.
    MustContain = 0b11,
  };

  /// Initializes a NoContain MembershipResult.
  /*implicit*/ MembershipResult() = default;
  /*implicit*/ MembershipResult(Kind kind) : kind_(kind) {}

  [[nodiscard]] auto operator==(const MembershipResult& rhs) const -> bool {
    return kind_ == rhs.kind_;
  }
  [[nodiscard]] auto operator!=(const MembershipResult& other) const -> bool {
    return !(*this == other);
  }

  /// Returns `true` if the element might be in the set.
  explicit operator bool() const { return kind_ != NoContain; }

  /// Performs a lattice join with @p rhs .
  [[nodiscard]] auto join(MembershipResult rhs) const -> MembershipResult {
    return static_cast<Kind>(kind_ | rhs.kind_);
  }

  /// Returns `true` if the element is not in the set.
  [[nodiscard]] bool isNo() const { return kind_ == NoContain; }
  /// Returns `true` if the element might be in the set, but is not known to.
  [[nodiscard]] bool isMay() const { return kind_ == MayContain; }
  /// Returns `true` if the element must be in the set.
  [[nodiscard]] bool isMust() const { return kind_ == MustContain; }

  /// Prints a textual representation of the MembershipResult to @p os .
  void print(llvm::raw_ostream& os) const;

  friend llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                       const MembershipResult& result) {
    result.print(os);
    return os;
  }

 private:
  Kind kind_;
};

//===----------------------------------------------------------------------===//
// ValuePredecessors
//===----------------------------------------------------------------------===//

/// Stores the predecessor values of an SSA value.
class ValuePredecessors {
  using set_type = llvm::SmallPtrSet<mlir::Value, 2>;

 public:
  /// Initializes an open set (lower bound) of @p values .
  [[nodiscard]] static auto lowerBound(mlir::ValueRange values = {})
      -> ValuePredecessors {
    return ValuePredecessors(false, values);
  }
  /// Initializes a closed set (exhaustive) of @p values .
  [[nodiscard]] static auto exhaustive(mlir::ValueRange values = {})
      -> ValuePredecessors {
    return ValuePredecessors(true, values);
  }

  /// Initializes an empty lower bound.
  /*implicit*/ ValuePredecessors() = default;

  /// Lattice joins  @p values that are @p is_exhaustive in-place.
  void join(mlir::ValueRange values, bool is_exhaustive = true) {
    is_exhaustive_ &= is_exhaustive;
    values_.insert_range(values);
  }
  /// Lattice joins @p rhs in-place.
  void join(const ValuePredecessors& rhs) {
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

  /// Determines whether @p value is in the set.
  [[nodiscard]] auto contains(mlir::Value value) const {
    if (getValues().contains(value)) {
      return MembershipResult::MustContain;
    }

    return isExhaustive() ? MembershipResult::NoContain
                          : MembershipResult::MayContain;
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
  friend class PredecessorInfo;

  explicit ValuePredecessors(bool is_exhaustive, mlir::ValueRange values)
      : is_exhaustive_(is_exhaustive), values_(llvm::from_range, values) {}

  bool is_exhaustive_ = false;
  set_type values_;
};

//===----------------------------------------------------------------------===//
// PredecessorInfo
//===----------------------------------------------------------------------===//

/// Caches information about SSA value predecessors.
class PredecessorInfo {
 public:
  using key_type = mlir::Value;
  using mapped_type = ValuePredecessors;
  using map_type = llvm::DenseMap<key_type, mapped_type>;

  // Allow construction as an MLIR analysis.
  explicit PredecessorInfo(mlir::Operation* /*op*/ = nullptr) {}

  /// Gets the immediate @p predecessors of @p value .
  ///
  /// @param            value         Value to query the predecessors of.
  /// @param  [in,out]  is_exhaustive Whether the result is exhaustive.
  /// @param  [out]     predecessors  Set of predecessors.
  void getPredecessors(mlir::Value value, bool& is_exhaustive,
                       llvm::SmallPtrSetImpl<mlir::Value>& predecessors);
  /// Gets the immediate @p predecessors of @p value .
  ///
  /// @param            value         Value to query the predecessors of.
  /// @param  [in,out]  predecessors  ValuePredecessors to update.
  void getPredecessors(mlir::Value value, ValuePredecessors& predecessors) {
    getPredecessors(value, predecessors.is_exhaustive_, predecessors.values_);
  }
  /// Gets the immediate predecessors of @p value .
  [[nodiscard]] auto getPredecessors(mlir::Value value) -> ValuePredecessors {
    ValuePredecessors result;
    getPredecessors(value, result);
    return result;
  }

  /// Gets the immediate control flow predecessors of @p value .
  ///
  /// This method only considers local control flow, i.e., it does not follow
  /// values across call boundaries.
  [[nodiscard]] auto getControlFlowPredecessors(mlir::Value value)
      -> const ValuePredecessors&;

  void transitiveClosure(llvm::SmallPtrSetImpl<mlir::Value>& values,
                         bool& is_exhaustive);
  void transitiveClosure(ValuePredecessors& predecessors) {
    transitiveClosure(predecessors.values_, predecessors.is_exhaustive_);
  }

 private:
  [[nodiscard]] auto getControlFlowPredecessorsImpl(mlir::BlockArgument arg)
      -> const ValuePredecessors&;
  [[nodiscard]] auto getControlFlowPredecessorsImpl(mlir::OpResult result)
      -> const ValuePredecessors&;

  map_type control_flow_;
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_PREDECESSORS_H_
