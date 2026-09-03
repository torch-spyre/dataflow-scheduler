//===-- SliceAnalysis.cpp ---------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Analysis/SliceAnalysis.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/IR/ValueRange.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Pass/AnalysisManager.h>

#include "dataflow-scheduler/Analysis/Predecessors.h"

using namespace scheduler;

#define DEBUG_TYPE "slice-analysis"

//===----------------------------------------------------------------------===//
// ForwardSliceAnalysis
//===----------------------------------------------------------------------===//

ForwardSlice::ForwardSlice(PredecessorInfo& preds, mlir::ValueRange values)
    : preds_(preds) {
  for (auto value : values) {
    cache_[value] = MembershipResult::MustContain;
  }
}

auto ForwardSlice::insert(mlir::ValueRange values) -> bool {
  if (llvm::all_of(values, [&](mlir::Value value) -> bool {
        return cache_[value] == MembershipResult::MustContain;
      })) {
    return false;
  }

  // This invalidates the cache apart from MustContain.
  map_type temp;
  using std::swap;
  swap(temp, cache_);

  for (auto value : values) {
    cache_[value] = MembershipResult::MustContain;
  }
  for (auto [key, value] : temp) {
    if (value == MembershipResult::MustContain) {
      cache_[key] = MembershipResult::MustContain;
    }
  }

  return true;
}

namespace {

[[nodiscard]] auto isVisible(mlir::Region& scope, mlir::Region& from) -> bool {
  for (auto* region = &from; region != nullptr;
       region = region->getParentRegion()) {
    if (region == &scope) {
      return true;
    }
    if (region->getParentOp()->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
      return false;
    }
  }

  return false;
}

}  // namespace

auto ForwardSlice::contains(mlir::Value value) -> MembershipResult {
  if (cache_.empty()) {
    // Short-circuit on the known empty slice.
    return MembershipResult::NoContain;
  }

  // Lookup cached result or initialize with NoContain.
  auto [it, invalid] = cache_.try_emplace(value, MembershipResult::NoContain);
  if (!invalid) {
    return it->second;
  }

  // Find all predecessors of the value.
  auto is_exhaustive = true;
  llvm::SmallPtrSet<mlir::Value, 8U> predecessors;
  preds_.getPredecessors(value, is_exhaustive, predecessors);

  if (!is_exhaustive) {
    // As we were unable to determine all potential predecessors of the value,
    // we need to fall back to a more complicated search strategy to get good
    // results. In particular, we first check if the value can be reached from
    // the slice instead of the other way around.
    if (!llvm::any_of(cache_, [&](map_type::value_type& entry) -> bool {
          if (entry.second == MembershipResult::NoContain) {
            return false;
          }

          // TODO: Use mlir::DominanceInfo to get a more accurate result.
          return isVisible(*entry.first.getParentRegion(),
                           *value.getParentRegion());
        })) {
      // No value in the slice reaches the definition of the given value,
      // therefore it must be independent.
      return MembershipResult::NoContain;
    }

    // We must assume that the value is in the slice, but we can still refine
    // the result to MustContain by inspecting the known lower bound.
    it->second = MembershipResult::MayContain;
  } else {
    // We may assume that the value is independent for now. If it is visited in
    // the recursive search (which can only happen within a graph region, or if
    // we follow block arguments), then it being part of its own cycle does not
    // make it a member of the slice.
  }

  // Save the result value because the iterator may be invalidated.
  auto result = it->second;
  for (auto predecessor : predecessors) {
    const auto pred_result = contains(predecessor);
    if (pred_result.isMust()) {
      return cache_[value] = MembershipResult::MustContain;
    }
    if (pred_result.isMay()) {
      result = cache_[value] = MembershipResult::MayContain;
    }
  }

  return result;
}

//===----------------------------------------------------------------------===//
// LoopSliceAnalysis
//===----------------------------------------------------------------------===//

namespace {

auto getLoopVariables(mlir::Operation* op) -> llvm::SmallVector<mlir::Value> {
  llvm::SmallVector<mlir::Value> result;

  auto iface = llvm::dyn_cast<mlir::LoopLikeOpInterface>(op);
  if (iface == nullptr) {
    return result;
  }

  if (const auto ivs = iface.getLoopInductionVars(); ivs) {
    llvm::append_range(result, ivs.value());
  }
  llvm::append_range(result, iface.getRegionIterArgs());
  return result;
}

}  // namespace

LoopSliceAnalysis::LoopSliceAnalysis(mlir::Operation* op,
                                     PredecessorInfo& preds)
    : ForwardSlice(preds, getLoopVariables(op)) {}

LoopSliceAnalysis::LoopSliceAnalysis(mlir::Operation* op,
                                     mlir::AnalysisManager& analyses)
    : LoopSliceAnalysis(op, analyses.getAnalysis<PredecessorInfo>()) {}
