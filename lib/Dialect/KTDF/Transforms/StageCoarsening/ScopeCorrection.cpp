//===-- ScopeCorrection.cpp -------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Transforms/StageCoarsening/ScopeCorrection.h"

#include "llvm/Support/DebugLog.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "stage-coarsening"

using namespace mlir;
using namespace mlir::ktdf;

void ScopeCorrection::run() {
  outer_private_ = outer_pipeline_.getPrivateOp();
  if (!outer_private_) {
    LDBG(1) << "  No outer private node found, skipping.";
    return;
  }

  auto inner_pipelines = findInnerPipelines();
  if (inner_pipelines.empty()) {
    LDBG(1) << "  No inner pipelines found, skipping.";
    return;
  }

  analyzePrivateResultUsage();
  if (pipeline_to_sc_info_.empty()) {
    LDBG(1) << "  No results need to be moved.";
    return;
  }

  // Move results to their respective inner pipelines
  for (auto& [inner_pipeline, sc_info] : pipeline_to_sc_info_) {
    movePrivateResultsToInnerPipeline(inner_pipeline, sc_info);
  }
}

llvm::SmallVector<PipelineOp> ScopeCorrection::findInnerPipelines() {
  llvm::SmallVector<PipelineOp> inner_pipelines;
  outer_pipeline_.walk([&](PipelineOp inner_pipeline) {
    if (inner_pipeline != outer_pipeline_) {
      inner_pipelines.push_back(inner_pipeline);
    }
    return WalkResult::advance();
  });
  return inner_pipelines;
}

void ScopeCorrection::analyzePrivateResultUsage() {
  auto yield_op =
      cast<PrivateYieldOp>(outer_private_.getBody()->getTerminator());

  SmallPtrSet<Operation*, 16> visited_moved_ops;
  for (auto [idx, result] : llvm::enumerate(outer_private_.getResults())) {
    // Skip tokens as they are handled by the pipeline tree materialization
    if (isa<TokenType>(result.getType())) continue;

    // Check all uses of this result
    bool only_used_in_inner = true;
    PipelineOp target_inner_pipeline = nullptr;

    for (OpOperand& use : result.getUses()) {
      // Find which pipeline (if any) contains this use
      Operation* user = use.getOwner();
      PipelineOp containing_pipeline = user->getParentOfType<PipelineOp>();
      assert(containing_pipeline &&
             "found a private result with uses outside of a pipeline");
      // If use is not in an inner pipeline, we can't move this result
      if (containing_pipeline == outer_pipeline_) {
        only_used_in_inner = false;
        break;
      }

      // Check if all uses are in the same inner pipeline
      if (!target_inner_pipeline) {
        target_inner_pipeline = containing_pipeline;
      } else if (target_inner_pipeline != containing_pipeline) {
        // Used in multiple inner pipelines, can't move
        only_used_in_inner = false;
        break;
      }
    }

    if (!only_used_in_inner || !target_inner_pipeline) continue;
    LDBG(1) << "  Result #" << idx << " only used in inner pipeline";

    auto& sc_info = pipeline_to_sc_info_[target_inner_pipeline];
    sc_info.target_pipeline = target_inner_pipeline;
    sc_info.result_indices.push_back(idx);
    sc_info.old_results.push_back(result);

    // Find the operation that produces this yielded value.
    // Avoid adding duplicate ops when there are ops with multiple results being
    // moved (eg. fifo.allocate).
    Value yielded_val = yield_op.getOperand(idx);
    Operation* def_op = yielded_val.getDefiningOp();
    if (def_op && visited_moved_ops.insert(def_op).second) {
      sc_info.ops_to_move.push_back(def_op);
    }
  }
}

void ScopeCorrection::movePrivateResultsToInnerPipeline(
    PipelineOp inner_pipeline, ScopeCorrectionInfo& sc_info) {
  LDBG(1) << "  Moving " << sc_info.result_indices.size()
          << " result(s) to inner pipeline";

  // Find or create the inner pipeline's private operation
  auto inner_private = inner_pipeline.getPrivateOp();
  if (inner_private) {
    // Inner pipeline already has a private node - extend it
    extendInnerPrivate(inner_private, sc_info);
  } else {
    // Create a new private node for the inner pipeline
    createInnerPrivate(inner_pipeline, sc_info);
  }

  // Update the outer private to remove the moved results
  updateOuterPrivate(sc_info.result_indices);
}

PrivateOp ScopeCorrection::recreatePrivateOp(
    PrivateOp old_private, llvm::ArrayRef<Type> new_result_types,
    llvm::ArrayRef<Operation*> ops_to_prepend,
    llvm::ArrayRef<Operation*> ops_to_exclude) {
  // Create new private op with new result types
  builder_.setInsertionPoint(old_private);
  auto new_private =
      PrivateOp::create(builder_, old_private.getLoc(), new_result_types);

  // Build the body
  Block& new_body = *new_private.getBody();
  IRMapping value_map;

  // Clone operations to prepend (e.g., moved operations)
  builder_.setInsertionPointToStart(&new_body);
  for (Operation* op : ops_to_prepend) {
    builder_.clone(*op, value_map);
  }

  // Build exclusion set for O(1) lookup
  llvm::DenseSet<Operation*> exclude_set(ops_to_exclude.begin(),
                                         ops_to_exclude.end());

  // Handle existing body operations (if old_private is non-null)
  if (old_private) {
    Block& old_body = *old_private.getBody();
    // Clone operations from old body (excluding terminator and excluded ops)
    for (Operation& op : old_body.without_terminator()) {
      if (!exclude_set.contains(&op)) {
        builder_.clone(op, value_map);
      } else {
        LDBG(1) << "      Skipping excluded operation: " << op;
      }
    }
  }

  // Build yield operands: prepended ops' results + old yield operands (if
  // old_private exists)
  llvm::SmallVector<Value> new_yield_operands;

  // Add results from prepended operations
  for (Operation* op : ops_to_prepend) {
    for (size_t i = 0; i < op->getNumResults(); i++) {
      new_yield_operands.push_back(value_map.lookup(op->getResult(i)));
    }
  }

  // Add mapped operands from old yield (if old_private exists)
  if (old_private) {
    Block& old_body = *old_private.getBody();
    auto old_yield = cast<PrivateYieldOp>(old_body.getTerminator());

    for (auto [idx, operand] : llvm::enumerate(old_yield.getOperands())) {
      // Skip operands produced by excluded operations
      if (Operation* def_op = operand.getDefiningOp()) {
        if (exclude_set.contains(def_op)) {
          continue;
        }
      }
      new_yield_operands.push_back(value_map.lookupOrDefault(operand));
    }
  }

  // Create new yield
  builder_.setInsertionPointToEnd(&new_body);
  PrivateYieldOp::create(builder_, old_private.getLoc(), new_yield_operands);

  return new_private;
}

void ScopeCorrection::extendInnerPrivate(PrivateOp inner_private,
                                         ScopeCorrectionInfo& sc_info) {
  LDBG(1) << "    Extending existing inner private";

  // Get the old yield to preserve existing operands
  auto old_yield =
      cast<PrivateYieldOp>(inner_private.getBody()->getTerminator());

  // Build new result types: moved results first, then existing results
  llvm::SmallVector<Type> new_result_types = sc_info.getMovingResultTypes();
  for (Value val : old_yield.getOperands()) {
    new_result_types.push_back(val.getType());
  }

  // Recreate the private op with moved operations prepended
  auto new_inner_private =
      recreatePrivateOp(inner_private, new_result_types,
                        sc_info.ops_to_move,  // ops to prepend
                        {});                  // no ops to exclude

  // Map moved results to new results (at the beginning)
  for (unsigned i = 0; i < sc_info.old_results.size(); ++i) {
    sc_info.old_results[i].replaceAllUsesWith(new_inner_private.getResult(i));
  }

  // Replace uses of old existing results with new results
  const unsigned moved_count = sc_info.old_results.size();
  for (unsigned i = 0; i < inner_private.getNumResults(); ++i) {
    inner_private.getResult(i).replaceAllUsesWith(
        new_inner_private.getResult(moved_count + i));
  }

  inner_private.erase();
}

void ScopeCorrection::createInnerPrivate(PipelineOp inner_pipeline,
                                         ScopeCorrectionInfo& sc_info) {
  LDBG(1) << "    Creating new inner private";

  // Determine result types from operations to move
  llvm::SmallVector<Type> result_types = sc_info.getMovingResultTypes();

  // Use helper to create the private op with moved operations (no old_private
  // to preserve)
  builder_.setInsertionPointToStart(inner_pipeline.getBody());
  auto new_private = recreatePrivateOp(nullptr, result_types,
                                       sc_info.ops_to_move,  // ops to prepend
                                       {});  // no ops to exclude

  // Replace all uses of old results with new results
  for (unsigned i = 0; i < sc_info.old_results.size(); ++i) {
    sc_info.old_results[i].replaceAllUsesWith(new_private.getResult(i));
  }
}

void ScopeCorrection::updateOuterPrivate(
    llvm::ArrayRef<unsigned> removed_indices) {
  LDBG(1) << "    Updating outer private";

  // Build set of indices to remove for O(1) lookup
  llvm::DenseSet<unsigned> to_remove(removed_indices.begin(),
                                     removed_indices.end());

  // Get the yield operation
  auto old_yield =
      cast<PrivateYieldOp>(outer_private_.getBody()->getTerminator());

  // Identify operations that define removed results (to be excluded from new
  // body)
  llvm::SmallVector<Operation*> ops_moved;
  for (unsigned idx : removed_indices) {
    Value yielded_val = old_yield.getOperand(idx);
    if (Operation* def_op = yielded_val.getDefiningOp()) {
      ops_moved.push_back(def_op);
    }
  }

  // Build new yield operands and result types (excluding removed ones)
  llvm::SmallVector<Type> new_result_types;

  for (auto [idx, operand] : llvm::enumerate(old_yield.getOperands())) {
    if (!to_remove.contains(idx)) {
      new_result_types.push_back(operand.getType());
    }
  }

  // Use helper to recreate the private op, excluding removed operations
  auto new_outer_private = recreatePrivateOp(outer_private_, new_result_types,
                                             {},          // no ops to prepend
                                             ops_moved);  // ops to exclude

  // Map old results to new results (skipping removed ones)
  unsigned new_idx = 0;
  for (unsigned old_idx = 0; old_idx < outer_private_.getNumResults();
       ++old_idx) {
    if (!to_remove.contains(old_idx)) {
      outer_private_.getResult(old_idx).replaceAllUsesWith(
          new_outer_private.getResult(new_idx));
      new_idx++;
    }
  }

  outer_private_.erase();
}

// Made with Bob
