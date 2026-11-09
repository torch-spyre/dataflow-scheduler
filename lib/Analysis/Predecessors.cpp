//===-- Predecessors.cpp ----------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Analysis/Predecessors.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>

using namespace scheduler;

#define DEBUG_TYPE "predecessor-info"

//===----------------------------------------------------------------------===//
// MembershipResult
//===----------------------------------------------------------------------===//

void MembershipResult::print(llvm::raw_ostream& os) const {
  switch (kind_) {
    case NoContain:
      os << "NoContain";
      return;
    case MayContain:
      os << "MayContain";
      return;
    case MustContain:
      os << "MustContain";
      return;
  }

  llvm_unreachable("unhandled MembershipResult::Kind");
}

//===----------------------------------------------------------------------===//
// PredecessorInfo
//===----------------------------------------------------------------------===//

namespace {

void visitControlFlow(mlir::SelectLikeOpInterface select,
                      PredecessorInfo::map_type& result) {
  if (select->getNumResults() != 1) {
    return;
  }

  result.emplace_or_assign(select->getResult(0), ValuePredecessors::exhaustive(
                                                     {select.getTrueValue(),
                                                      select.getFalseValue()}));
}

void visitControlFlow(mlir::RegionBranchOpInterface branch,
                      PredecessorInfo::map_type& result) {
  for (auto branch_point : branch.getAllRegionBranchPoints()) {
    llvm::SmallVector<mlir::RegionSuccessor> successors;
    branch.getSuccessorRegions(branch_point, successors);
    for (auto& successor : successors) {
      const auto succ = successor.getSuccessorInputs();
      const auto pred = branch.getSuccessorOperands(branch_point, successor);
      for (auto [dst, src] : llvm::zip_equal(succ, pred)) {
        auto& cached = result.try_emplace(dst, ValuePredecessors::exhaustive())
                           .first->getSecond();
        cached.join(src);
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
        result.try_emplace(arg, ValuePredecessors::lowerBound());
      }
    }
  }
}

void visitControlFlow(mlir::BranchOpInterface branch,
                      PredecessorInfo::map_type& result) {
  for (auto& successor : branch->getBlockOperands()) {
    const auto operands =
        branch.getSuccessorOperands(successor.getOperandNumber());

    for (auto argument : successor.get()->getArguments()) {
      auto& cached =
          result.try_emplace(argument, ValuePredecessors::exhaustive())
              .first->getSecond();
      if (const auto forwarded = operands[argument.getArgNumber()]; forwarded) {
        cached.join(forwarded);
      } else {
        cached.join({}, false);
      }
    }
  }
}

}  // namespace

void PredecessorInfo::getPredecessors(
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

auto PredecessorInfo::getControlFlowPredecessors(mlir::Value value)
    -> const ValuePredecessors& {
  {
    const auto it = control_flow_.find(value);
    if (it != control_flow_.end()) {
      return it->second;
    }
  }

  if (const auto arg = llvm::dyn_cast<mlir::BlockArgument>(value); arg) {
    return getControlFlowPredecessorsImpl(arg);
  }

  const auto result = llvm::cast<mlir::OpResult>(value);
  return getControlFlowPredecessorsImpl(result);
}

void PredecessorInfo::transitiveClosure(
    llvm::SmallPtrSetImpl<mlir::Value>& values, bool& is_exhaustive) {
  llvm::SmallVector<mlir::Value> work_list(values.begin(), values.end());

  llvm::SmallPtrSet<mlir::Value, 8> predecessors;
  while (!work_list.empty()) {
    getPredecessors(work_list.pop_back_val(), is_exhaustive, predecessors);
    for (auto value : predecessors) {
      if (!values.insert(value).second) {
        continue;
      }

      work_list.push_back(value);
    }

    predecessors.clear();
  }
}

auto PredecessorInfo::getControlFlowPredecessorsImpl(mlir::BlockArgument arg)
    -> const ValuePredecessors& {
  // Handle block arguments of blocks with predecessors.
  if (!arg.getOwner()->hasNoPredecessors()) {
    // Argument is fully determined by predecessors' branch operations.
    auto is_exact = arg.getOwner()->getParentOp()->isRegistered();
    for (auto* const pred : arg.getOwner()->getPredecessors()) {
      if (auto iface =
              llvm::dyn_cast<mlir::BranchOpInterface>(pred->getTerminator())) {
        visitControlFlow(iface, control_flow_);
      } else {
        // We don't understand that one.
        LDBG() << "unknown branch semantics for op " << pred->getTerminator();
        is_exact = false;
      }
    }

    auto& result = control_flow_[arg];
    result.is_exhaustive_ = is_exact;
    return result;
  }

  // Handle arguments of function entry blocks.
  if (auto iface = llvm::dyn_cast<mlir::FunctionOpInterface>(
          arg.getOwner()->getParentOp());
      iface) {
    // We do not perform any inter-procedural analyses.
    return control_flow_[arg] = ValuePredecessors::exhaustive();
  }

  // Handle arguments of region branch entry blocks.
  if (auto iface = llvm::dyn_cast<mlir::RegionBranchOpInterface>(
          arg.getOwner()->getParentOp());
      iface) {
    // Argument is determined by region branch semantics.
    visitControlFlow(iface, control_flow_);
    return control_flow_[arg];
  }

  // There are no other kinds of local control flow.
  return control_flow_[arg] = ValuePredecessors::exhaustive();
}

auto PredecessorInfo::getControlFlowPredecessorsImpl(mlir::OpResult result)
    -> const ValuePredecessors& {
  // Check if the producer has select semantics.
  if (auto iface =
          llvm::dyn_cast<mlir::SelectLikeOpInterface>(result.getOwner());
      iface) {
    // Result is determined by select semantics.
    visitControlFlow(iface, control_flow_);
    return control_flow_[result];
  }

  // Check if the producer has region branching semantics.
  if (auto iface =
          llvm::dyn_cast<mlir::RegionBranchOpInterface>(result.getOwner());
      iface) {
    // Result is determined by region branch semantics.
    visitControlFlow(iface, control_flow_);
    return control_flow_[result];
  }

  // There are no other kinds of local control flow.
  return control_flow_[result] = ValuePredecessors::exhaustive();
}
