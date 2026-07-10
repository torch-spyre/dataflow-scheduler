//===-- PipelineTreeLegalizer.cpp -----------------------------------------===//
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
// Pipeline Tree Legalizer Implementation
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Transforms/Utils/PipelineTreeLegalizer.h"

#include "dataflow-scheduler/Analysis/PipelineTree.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "pipeline-tree-legalizer"

using namespace scheduler;
using namespace scheduler::pipeline_tree;

//===----------------------------------------------------------------------===//
// Rule Implementations
//===----------------------------------------------------------------------===//

bool PipelineChildrenRule::isViolated(PipelineTreeNode* node,
                                      PipelineTree& tree) const {
  // Check if this is a pipeline with illegal children
  if (!node->isPipelineNode()) return false;

  OperationTreeNode* child = node->getFirstChild();
  while (child) {
    PipelineTreeNode* child_pnode = static_cast<PipelineTreeNode*>(child);
    if (!child_pnode->isPrivateNode() && !child_pnode->isStageNode()) {
      return true;
    }
    child = child->getNextSibling();
  }

  return false;
}

bool PipelineChildrenRule::fixViolation(PipelineTreeNode* violating_node,
                                        PipelineTree& tree,
                                        int& next_stage_id) {
  // violating_node is the pipeline that has illegal children
  // Group consecutive illegal children and wrap each group in a single stage

  assert(violating_node->isPipelineNode() && "Expected pipeline node");

  LDBG(1) << "      Wrapping illegal children of pipeline "
          << violating_node->getNodeName();

  // Group consecutive illegal children
  llvm::SmallVector<llvm::SmallVector<PipelineTreeNode*>> illegal_groups;
  llvm::SmallVector<PipelineTreeNode*> current_group;

  OperationTreeNode* child = violating_node->getFirstChild();
  while (child) {
    PipelineTreeNode* child_pnode = static_cast<PipelineTreeNode*>(child);
    if (!child_pnode->isPrivateNode() && !child_pnode->isStageNode()) {
      // Illegal child - add to current group
      current_group.push_back(child_pnode);
    } else {
      // Legal child - close current group if any
      if (!current_group.empty()) {
        illegal_groups.push_back(current_group);
        current_group.clear();
      }
    }
    child = child->getNextSibling();
  }

  // Don't forget the last group
  if (!current_group.empty()) {
    illegal_groups.push_back(current_group);
  }

  // Wrap each group in a new stage
  for (const auto& group : illegal_groups) {
    LDBG(1) << "        Creating coarsened stage wrapper for " << group.size()
            << " consecutive illegal child(ren)";

    // Create a new coarsened stage
    StageNode* new_stage = tree.createStageNode(nullptr, next_stage_id++);

    // Get the position of the first child in the group for insertion
    OperationTreeNode* insert_before = group[0]->getPrevSibling();

    // Unlink all children in the group and add them to the new stage
    for (PipelineTreeNode* illegal_child : group) {
      illegal_child->unlink();
      new_stage->insertChildNode(illegal_child);
    }

    // Insert coarsened stage as child of pipeline (at the position of the first
    // child)
    violating_node->insertChildNode(new_stage, insert_before);

    LDBG(1) << "        Created coarsened stage " << (next_stage_id - 1);
  }
  return true;
}

void PipelineChildrenRule::print(llvm::raw_ostream& os) const {
  os << "Immediate children of pipeline must be stage or private";
}

bool StageParentRule::isViolated(PipelineTreeNode* node,
                                 PipelineTree& tree) const {
  // Check if this is a stage with non-pipeline parent
  if (!node->isStageNode()) return false;

  OperationTreeNode* parent = node->getParentNode();
  assert(parent && "all nodes must have a parent except root node");

  auto parent_pnode = static_cast<PipelineTreeNode*>(parent);
  return !parent_pnode->isPipelineNode();
}

bool StageParentRule::fixViolation(PipelineTreeNode* violating_node,
                                   PipelineTree& tree,
                                   int& next_coarsened_stage_id) {
  LDBG(1) << "      Creating nested pipeline for stage "
          << violating_node->getNodeName();
  assert(violating_node->isStageNode() && "Expected stage node");

  // Get the non-pipeline parent
  OperationTreeNode* parent = violating_node->getParentNode();
  assert(parent && "Stage should have a parent");
  auto parent_pnode = static_cast<PipelineTreeNode*>(parent);
  assert(!parent_pnode->isPipelineNode() &&
         "Parent should not be a pipeline since expected violation");

  // Create a new nested pipeline node (unmaterialized)
  PipelineNode* nested_pipeline = tree.createPipelineNode(nullptr);

  // Get the template from the nearest ancestor pipeline
  OperationTreeNode* ancestor = parent_pnode->getParentNode();
  while (ancestor) {
    auto ancestor_pnode = static_cast<PipelineTreeNode*>(ancestor);
    if (ancestor_pnode->isPipelineNode()) {
      PipelineNode* ancestor_pipeline =
          static_cast<PipelineNode*>(ancestor_pnode);
      if (ancestor_pipeline->getOperation()) {
        nested_pipeline->setTemplateOp(ancestor_pipeline->getOperation());
      } else {
        assert(ancestor_pipeline->getTemplateOp());
        nested_pipeline->setTemplateOp(ancestor_pipeline->getTemplateOp());
      }
      LDBG(1) << "      Set template from ancestor pipeline";
      break;
    }
    ancestor = ancestor->getParentNode();
  }

  LDBG(1) << "      Created nested pipeline node";

  // Collect all stage siblings that should move to the nested pipeline
  // (all stages that are children of the same non-pipeline parent)
  llvm::SmallVector<StageNode*> stages_to_move;
  OperationTreeNode* child = parent_pnode->getFirstChild();
  while (child) {
    auto child_pnode = static_cast<PipelineTreeNode*>(child);
    if (child_pnode->isStageNode()) {
      stages_to_move.push_back(static_cast<StageNode*>(child_pnode));
    }
    child = child->getNextSibling();
  }

  LDBG(1) << "      Moving " << stages_to_move.size()
          << " stage(s) to nested pipeline";

  // Insert the nested pipeline as a child of the parent
  // Get the position before the first stage
  OperationTreeNode* insert_before = nullptr;
  assert(!stages_to_move.empty() &&
         "expected to find at least the original violating stage node");
  insert_before = stages_to_move[0]->getPrevSibling();
  parent_pnode->insertChildNode(nested_pipeline, insert_before);

  // Move all stages to be children of the nested pipeline
  for (StageNode* stage : stages_to_move) {
    stage->unlink();
    nested_pipeline->insertChildNode(stage);
  }

  LDBG(1) << "      Nested pipeline created successfully";
  return true;
}

void StageParentRule::print(llvm::raw_ostream& os) const {
  os << "Stage must have pipeline as immediate parent";
}

bool PrivateMustExistRule::isViolated(PipelineTreeNode* node,
                                      PipelineTree& tree) const {
  // Check if this is a pipeline node
  if (!node->isPipelineNode()) return false;

  // Check if the first child is a private node
  OperationTreeNode* first_child = node->getFirstChild();
  if (first_child &&
      static_cast<PipelineTreeNode*>(first_child)->isPrivateNode())
    return false;

  // Check if any immediate child stages have non-empty dependencies
  OperationTreeNode* child = first_child;
  while (child) {
    auto child_pnode = static_cast<PipelineTreeNode*>(child);
    if (!child_pnode->isStageNode()) continue;
    auto stage = static_cast<StageNode*>(child_pnode);
    if (!stage->getDependencies().empty()) {
      return true;
    }
    child = child->getNextSibling();
  }
  return false;
}

bool PrivateMustExistRule::fixViolation(PipelineTreeNode* violating_node,
                                        PipelineTree& tree,
                                        int& next_coarsened_stage_id) {
  assert(violating_node->isPipelineNode() && "Expected pipeline node");

  LDBG(1) << "      Creating private node as first child of pipeline "
          << violating_node->getNodeName();

  // Create a new private node (unmaterialized)
  PrivateNode* new_private = tree.createPrivateNode(nullptr);

  // Insert the private node as the first child of the pipeline
  violating_node->insertAsFirstChild(new_private);

  LDBG(1) << "      Created private node as first child";
  return true;
}

void PrivateMustExistRule::print(llvm::raw_ostream& os) const {
  os << "Pipeline with stages that have dependencies must have private node as "
        "first child";
}

//===----------------------------------------------------------------------===//
// Legalizer Implementation
//===----------------------------------------------------------------------===//

Legalizer::Legalizer() : next_coarsened_stage_id_(0) {
  // Initialize all rules
  rules_.push_back(std::make_unique<PipelineChildrenRule>());
  rules_.push_back(std::make_unique<StageParentRule>());
  rules_.push_back(std::make_unique<PrivateMustExistRule>());
}

int Legalizer::findAndFixViolations(PipelineTree& tree) {
  assert(tree.getRoot());
  // Initialize next_coarsened_stage_id_ to one past the maximum existing stage
  // ID
  next_coarsened_stage_id_ = 0;
  OperationTreeNode::walk<OperationTreeNode::kPreOrder>(
      tree.getRoot(), [this](OperationTreeNode* node) {
        auto pnode = static_cast<PipelineTreeNode*>(node);
        if (pnode->isStageNode()) {
          auto stage = static_cast<StageNode*>(pnode);
          if (stage->getStageId() >= next_coarsened_stage_id_) {
            next_coarsened_stage_id_ = stage->getStageId() + 1;
          }
        }
        return node;
      });

  LDBG(1) << "  Starting coarsened stage IDs at " << next_coarsened_stage_id_;

  // Collect initial violations
  violations_.clear();
  collectViolations(tree);

  if (violations_.empty()) {
    LDBG(1) << "  No structure violations found.";
    return 0;
  }

  // Iteratively fix violations until none remain
  int iteration = 0;
  while (!violations_.empty()) {
    ++iteration;

    LLVM_DEBUG({
      llvm::dbgs() << "  Iteration " << iteration << ": Found "
                   << violations_.size() << " violation(s)\n";

      // Print all violations found in this iteration
      for (const Violation& violation : violations_) {
        llvm::dbgs() << "    - ";
        violation.print(llvm::dbgs());
        llvm::dbgs() << "\n";
      }
    });

    // Fix the first violation
    Violation& violation = violations_.front();
    LLVM_DEBUG({
      llvm::dbgs() << "  Correcting: ";
      violation.print(llvm::dbgs());
      llvm::dbgs() << "\n";
    });

    violation.rule->fixViolation(violation.violating_node, tree,
                                 next_coarsened_stage_id_);

    LLVM_DEBUG({
      llvm::dbgs() << "tree after fixing violation:\n";
      tree.print(llvm::dbgs());
    });

    // Re-collect violations after the fix
    // (fixing one violation may resolve or create others)
    violations_.clear();
    collectViolations(tree);
  }

  LDBG(1) << "  All violations resolved after " << iteration << " iteration(s)";
  return iteration;
}

void Legalizer::collectViolations(PipelineTree& tree) {
  // NOTE: This collects ALL violations in the tree, but correctViolations()
  // only fixes the first one before re-collecting. This means we do redundant
  // work checking nodes after the first violation is found.
  //
  // TODO: Enhance OperationTree::walk() to support early termination when
  // the client signals (e.g., by returning a special value or throwing an
  // exception). This would allow us to stop traversal after finding the
  // first violation, improving efficiency.
  //
  // Current overhead is acceptable because:
  // - Trees are typically small (dozens of nodes, not thousands)
  // - Violations converge quickly (usually 1-2 iterations)
  // - Tree traversal is O(n) which is fast for small n

  OperationTreeNode::walk<OperationTreeNode::kPreOrder>(
      tree.getRoot(), [&](OperationTreeNode* node) {
        auto pnode = static_cast<PipelineTreeNode*>(node);

        // Check all rules against this node
        for (const auto& rule : rules_) {
          if (rule->isViolated(pnode, tree)) {
            // Clone the rule for this violation
            std::unique_ptr<Rule> rule_copy;
            if (llvm::isa<PipelineChildrenRule>(rule.get())) {
              rule_copy = std::make_unique<PipelineChildrenRule>();
            } else if (llvm::isa<StageParentRule>(rule.get())) {
              rule_copy = std::make_unique<StageParentRule>();
            } else if (llvm::isa<PrivateMustExistRule>(rule.get())) {
              rule_copy = std::make_unique<PrivateMustExistRule>();
            }

            violations_.emplace_back(std::move(rule_copy), pnode);
          }
        }

        return node;
      });
}

// Made with Bob
