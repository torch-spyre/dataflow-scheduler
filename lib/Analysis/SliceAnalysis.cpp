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

using namespace scheduler;

#define DEBUG_TYPE "slice-analysis"

namespace {

void visitControlFlow(mlir::SelectLikeOpInterface select,
                      BackwardSliceAnalysis::map_type& result) {
  if (select->getNumResults() != 1) {
    return;
  }

  result.emplace_or_assign(
      select->getResult(0),
      BackwardSliceAnalysis::Predecessors::exhaustive(
          {select.getTrueValue(), select.getFalseValue()}));
}

void visitControlFlow(mlir::RegionBranchOpInterface branch,
                      BackwardSliceAnalysis::map_type& result) {
  for (auto branch_point : branch.getAllRegionBranchPoints()) {
    llvm::SmallVector<mlir::RegionSuccessor> successors;
    branch.getSuccessorRegions(branch_point, successors);
    for (auto& successor : successors) {
      const auto succ = successor.getSuccessorInputs();
      const auto pred = branch.getSuccessorOperands(branch_point, successor);
      for (auto [dst, src] : llvm::zip_equal(succ, pred)) {
        auto& cached =
            result
                .try_emplace(dst,
                             BackwardSliceAnalysis::Predecessors::exhaustive())
                .first->getSecond();
        cached.unite(src);
      }

      if (successor.isParent()) {
        continue;
      }

      // The leading block arguments that aren't forwarded are considered
      // "produced" by the operation (such as IVs).
      auto args = successor.getSuccessor()->getArguments();
      if (!succ.empty()) {
        assert(mlir::cast<mlir::BlockArgument>(succ.back()).getArgNumber() ==
               args.size() - 1);
        args = args.take_front(
            mlir::cast<mlir::BlockArgument>(succ.front()).getArgNumber());
      }
      for (auto arg : args) {
        result.try_emplace(arg,
                           BackwardSliceAnalysis::Predecessors::lowerBound());
      }
    }
  }
}

void visitControlFlow(mlir::BranchOpInterface branch,
                      BackwardSliceAnalysis::map_type& result) {
  for (auto& successor : branch->getBlockOperands()) {
    const auto operands =
        branch.getSuccessorOperands(successor.getOperandNumber());

    for (auto argument : successor.get()->getArguments()) {
      auto& cached =
          result
              .try_emplace(argument,
                           BackwardSliceAnalysis::Predecessors::exhaustive())
              .first->getSecond();
      if (const auto forwarded = operands[argument.getArgNumber()]; forwarded) {
        cached.unite(forwarded);
      } else {
        cached.unite({}, false);
      }
    }
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// BackwardSliceAnalysis
//===----------------------------------------------------------------------===//

BackwardSliceAnalysis::BackwardSliceAnalysis(mlir::Operation* /*op*/) {}

void BackwardSliceAnalysis::getPredecessors(
    mlir::Value value, bool& is_exhaustive,
    SmallPtrSetImpl<mlir::Value>& predecessors) {
  if (const auto result = llvm::dyn_cast<mlir::OpResult>(value); result) {
    is_exhaustive &= result.getOwner()->isRegistered();
    predecessors.insert_range(result.getOwner()->getOperands());
  }

  const auto& control_flow = getControlFlowPredecessors(value);
  is_exhaustive &= control_flow.is_exhaustive_;
  predecessors.insert_range(control_flow.values_);

  LDBG_OS([&](llvm::raw_ostream& os) {
    os << "getPredecessors(";
    value.printAsOperand(os, mlir::OpPrintingFlags().skipRegions());
    os << "): " << (is_exhaustive ? "exhaustive(" : "lowerBound(");
    llvm::interleaveComma(predecessors, os);
    os << ")";
  });
}

auto BackwardSliceAnalysis::getControlFlowPredecessors(mlir::Value value)
    -> const Predecessors& {
  {
    const auto it = control_flow_.find(value);
    if (it != control_flow_.end()) {
      return it->second;
    }
  }

  if (const auto argument = llvm::dyn_cast<mlir::BlockArgument>(value);
      argument) {
    if (!argument.getOwner()->hasNoPredecessors()) {
      // Argument is fully determined by predecessors' branch operations.
      auto is_exact = argument.getOwner()->getParentOp()->isRegistered();
      for (auto* const pred : argument.getOwner()->getPredecessors()) {
        if (auto iface = llvm::dyn_cast<mlir::BranchOpInterface>(
                pred->getTerminator())) {
          visitControlFlow(iface, control_flow_);
        } else {
          // We don't understand that one.
          is_exact = false;
        }
      }

      auto& result = control_flow_[value];
      result.is_exhaustive_ = is_exact;
      return result;
    }

    if (auto iface = llvm::dyn_cast<mlir::FunctionOpInterface>(
            argument.getOwner()->getParentOp());
        iface) {
      // We do not perform any inter-procedural analyses.
      return control_flow_[value] = Predecessors::exhaustive();
    }

    if (auto iface = llvm::dyn_cast<mlir::RegionBranchOpInterface>(
            argument.getOwner()->getParentOp());
        iface) {
      // Argument is determined by region branch semantics.
      visitControlFlow(iface, control_flow_);
    }
  } else {
    const auto result = llvm::cast<mlir::OpResult>(value);

    if (auto iface =
            llvm::dyn_cast<mlir::SelectLikeOpInterface>(result.getOwner());
        iface) {
      // Result is determined by select semantics.
      visitControlFlow(iface, control_flow_);
    } else if (auto iface = llvm::dyn_cast<mlir::RegionBranchOpInterface>(
                   result.getOwner());
               iface) {
      // Result is determined by region branch semantics.
      visitControlFlow(iface, control_flow_);
    } else {
      return control_flow_[value] = Predecessors::exhaustive();
    }
  }

  return control_flow_[value];
}

//===----------------------------------------------------------------------===//
// ForwardSliceAnalysis
//===----------------------------------------------------------------------===//

ForwardSlice::ForwardSlice(BackwardSliceAnalysis& backward,
                           mlir::ValueRange values)
    : backward_(backward) {
  for (auto value : values) {
    cache_[value] = Result::MustContain;
  }
}

auto ForwardSlice::insert(mlir::ValueRange values) -> bool {
  if (llvm::all_of(values, [&](mlir::Value value) -> bool {
        return cache_[value] == Result::MustContain;
      })) {
    return false;
  }

  // This invalidates the cache apart from MustContain.
  map_type temp;
  using std::swap;
  swap(temp, cache_);

  for (auto value : values) {
    cache_[value] = Result::MustContain;
  }
  for (auto [key, value] : temp) {
    if (value == Result::MustContain) {
      cache_[key] = Result::MustContain;
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

auto ForwardSlice::contains(mlir::Value value) -> Result {
  if (cache_.empty()) {
    // Short-circuit on the known empty slice.
    return Result::NoContain;
  }

  // Lookup cached result or initialize with NoContain.
  auto [it, invalid] = cache_.try_emplace(value, Result::NoContain);
  if (!invalid) {
    return it->second;
  }

  // Find all predecessors of the value.
  auto is_exhaustive = true;
  llvm::SmallPtrSet<mlir::Value, 8U> predecessors;
  backward_.getPredecessors(value, is_exhaustive, predecessors);

  if (!is_exhaustive) {
    // As we were unable to determine all potential predecessors of the value,
    // we need to fall back to a more complicated search strategy to get good
    // results. In particular, we first check if the value can be reached from
    // the slice instead of the other way around.
    if (!llvm::any_of(cache_, [&](map_type::value_type& entry) -> bool {
          if (entry.second == Result::NoContain) {
            return false;
          }

          // TODO: Use mlir::DominanceInfo to get a more accurate result.
          return isVisible(*entry.first.getParentRegion(),
                           *value.getParentRegion());
        })) {
      // No value in the slice reaches the definition of the given value,
      // therefore it must be independent.
      return Result::NoContain;
    }

    // We must assume that the value is in the slice, but we can still refine
    // the result to MustContain by inspecting the known lower bound.
    it->second = Result::MayContain;
  } else {
    // We may assume that the value is independent for now. If it is visited in
    // the recursive search (which can only happen within a graph region, or if
    // we follow block arguments), then it being part of its own cycle does not
    // make it a member of the slice.
  }

  // Save the result value because the iterator may be invalidated.
  auto result = it->second;
  for (auto predecessor : predecessors) {
    switch (contains(predecessor)) {
      case Result::MustContain:
        return cache_[value] = Result::MustContain;
      case Result::MayContain:
        result = cache_[value] = Result::MayContain;
        continue;
      case Result::NoContain:
        continue;
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
                                     BackwardSliceAnalysis& backward)
    : ForwardSlice(backward, getLoopVariables(op)) {}

LoopSliceAnalysis::LoopSliceAnalysis(mlir::Operation* op,
                                     mlir::AnalysisManager& analyses)
    : LoopSliceAnalysis(op, analyses.getAnalysis<BackwardSliceAnalysis>()) {}
