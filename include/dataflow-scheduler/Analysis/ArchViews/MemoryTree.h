//===-- MemoryTree.h --------------------------------------------*- c++ -*-===//
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
// Memory Tree
//
// This file defines a memory hierarchy view that represents the memory
// hierarchy as a tree structure based on an input architecture graph.
// The tree is built from a ktdf_arch::DeviceOp and shows the hierarchical
// relationships between memory resources.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_MEMORYTREE_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_MEMORYTREE_H_

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/Pass/AnalysisManager.h>

#include <optional>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"

namespace scheduler {

// Type alias for resource representation - mlir::Attribute
using ResourceType = mlir::Attribute;

namespace arch_view {

class MemoryTree : public mlir::ktdf_arch::DeviceView {
 public:
  using NodeId = unsigned;

  struct MemoryNode {
    NodeId id;
    ResourceType memory_resource;        // Canonical resource
    std::optional<ResourceType> alias;   // Optional alias for this resource
    std::optional<NodeId> parent;        // Parent memory in the hierarchy
    llvm::SmallVector<NodeId> children;  // Child memories in the hierarchy
    std::optional<size_t> capacity_in_bytes;
    std::optional<size_t> reserved_in_bytes;
  };

  /// Construct a MemoryTree as an MLIR analysis child of @p declaration .
  explicit MemoryTree(mlir::ktdf_arch::DeviceOp declaration,
                      mlir::AnalysisManager& analyses);

  /// Get a memory node by its ID
  std::optional<MemoryNode> getNode(NodeId node_id) const;

  /// Get a memory node by its resource attribute
  std::optional<MemoryNode> getNode(ResourceType memory_resource) const;

  /// Get the node ID for a given memory resource
  std::optional<NodeId> getNodeIdForResource(
      ResourceType memory_resource) const;

  /// Get all root nodes (memories with no parent)
  llvm::SmallVector<NodeId> getRootNodes() const;

  /// Get all leaf nodes (memories with no children)
  llvm::SmallVector<NodeId> getLeafNodes() const;

  /// Get the path from a source memory to a destination memory
  /// Returns empty vector if no path exists
  llvm::SmallVector<NodeId> getPath(NodeId source, NodeId target) const;

  /// Get the path from a source memory to a destination memory (by resource)
  llvm::SmallVector<NodeId> getPath(ResourceType source,
                                    ResourceType target) const;

  /// Get all memory nodes at a specific depth in the tree
  /// Depth 0 = root nodes, depth 1 = children of roots, etc.
  llvm::SmallVector<NodeId> getNodesAtDepth(unsigned depth) const;

  /// Get the depth of a node in the tree (distance from root)
  /// Returns nullopt if node doesn't exist
  std::optional<unsigned> getNodeDepth(NodeId node_id) const;

  /// Get all node IDs in the tree
  llvm::SmallVector<NodeId> getAllNodeIds() const;

  /// Check if a memory is a descendant of another memory (child, grandchild,
  /// etc.) Returns true if descendant is below ancestor in the hierarchy
  bool isDescendantOf(NodeId descendant, NodeId ancestor) const;

  /// Check if a memory is a descendant of another memory (by resource)
  bool isDescendantOf(ResourceType descendant, ResourceType ancestor) const;

  /// Get the capacity of a memory resource
  /// @return Capacity in bytes, or nullopt if not available
  std::optional<size_t> getCapacity(ResourceType memory_resource) const;

  /// Get the reserved/system memory of a memory resource
  /// @return Reserved bytes, or nullopt if not available
  std::optional<size_t> getReserved(ResourceType memory_resource) const;

  /// Get the available capacity (capacity - reserved) of a memory resource
  /// @return Available bytes, or nullopt if capacity not available
  std::optional<size_t> getAvailable(ResourceType memory_resource) const;

  /// Get all memory resources that have capacity information
  llvm::SmallVector<ResourceType> getMemoryResourcesWithCapacity() const;

  /// Check if two resources are aliases of each other
  /// @return true if the resources are aliases (including if they are the same)
  bool areAliases(ResourceType resource1, ResourceType resource2) const;

  /// Get all aliases for a resource (including the resource itself)
  /// @return Vector of all known aliases, or just the resource if no aliases
  llvm::SmallVector<ResourceType> getAliases(ResourceType resource) const;

  /// Check if a resource is global memory (root of tree)
  /// Matches against the resource and all its aliases
  /// @return true if resource matches any root node or its aliases
  bool isGlobalMemory(ResourceType memory_resource) const;

  /// Check if a resource is per-core scratchpad memory
  /// (immediate child of global memory)
  /// Matches against the resource and all its aliases
  /// @return true if resource is at depth 1 in the tree
  bool isPerCoreScratchPadMemory(ResourceType memory_resource) const;

  /// Check if a resource is below scratchpad memory
  /// (descendant of scratchpad memory)
  /// @return true if resource is a descendant of any scratchpad memory
  bool isBelowScratchPad(ResourceType memory_resource) const;

  /// Print the memory tree structure
  void print(llvm::raw_ostream& os) const;
  void dump() const;

 private:
  NodeId next_node_id_ = 0;
  llvm::DenseMap<NodeId, MemoryNode> nodes_;
  llvm::DenseMap<ResourceType, NodeId> resource_to_node_;
  llvm::DenseMap<ResourceType, NodeId> alias_to_node_;  // Maps aliases to nodes

  /// Helper to add a memory node to the tree
  NodeId addNode(ResourceType memory_resource,
                 std::optional<size_t> capacity_in_bytes,
                 std::optional<size_t> reserved_in_bytes);

  /// Helper to establish parent-child relationship
  void setParent(NodeId child_id, NodeId parent_id);

  /// Helper to print a subtree recursively
  void printSubtree(llvm::raw_ostream& os, NodeId node_id,
                    unsigned indent_level) const;

  /// Build the memory tree from the device held by this view.
  void initialize();

  /// Populate aliases from mem_space_mapping attribute
  void populateAliases(mlir::Attribute mem_space_mapping);

  /// Helper to stringify a resource attribute
  static std::string stringifyResource(ResourceType resource);
};

}  // namespace arch_view
}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_MEMORYTREE_H_

// Made with Bob