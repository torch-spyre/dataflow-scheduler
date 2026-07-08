//===----------------------------------------------------------------------===//
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

#include "dataflow-scheduler/Analysis/ArchViews/RoutingGraph.h"

#include <cassert>
#include <queue>
#include <sstream>

#include "Ktdp/KtdpAttrs.hpp"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace scheduler::arch_view;

namespace {

// Helper struct to track state during device initialization
struct DeviceInitContext {
  RoutingGraph& graph;

  // Maps SSA values (from memory/exec_unit ops) to their corresponding NodeIds
  llvm::DenseMap<mlir::Value, RoutingGraph::NodeId> value_to_node_id;

  // Tracks which group kinds we've already processed (for flattening)
  llvm::DenseSet<mlir::Attribute> processed_group_kinds;

  // Collects memory node sizes to apply after traversal
  // Key: NodeId of memory node
  // Value: Size in bytes
  llvm::DenseMap<RoutingGraph::NodeId, size_t> node_sizes;

  // Tracks load/store unit SSA values (exec_units with load_store attribute)
  // These are NOT created as nodes, but used as transfer units on edges
  // Contains: result of load/store exec_unit ops
  llvm::DenseSet<mlir::Value> load_store_units;

  // For load/store unit datapath patterns like: A -> LS -> B
  // We see two datapaths: "A to LS" and "LS to B"
  // This map tracks ALL sources from the first datapaths (can be multiple)
  // Key: the LS unit
  // Value: vector of sources (A1, A2, ...) from "A to LS" datapaths
  // When we see the second datapath "LS to B", we create edges: A1 -> B, A2 ->
  // B, ... with LS as transfer unit
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value>> ls_unit_sources;

  explicit DeviceInitContext(RoutingGraph& g) : graph(g) {}
};

// Process a memory operation
void processMemoryOp(mlir::ktdf_arch::MemoryOp mem_op, DeviceInitContext& ctx) {
  auto kind = mem_op.getKind();
  assert(kind && "Memory operation must have a kind attribute");

  auto node_id =
      ctx.graph.addNode(kind, RoutingGraph::ResourceNode::ResourceKind::Memory);
  ctx.value_to_node_id[mem_op.getResult()] = node_id;

  // Extract size property if present - will be set in initializeFromDeviceOp
  if (auto size_attr = mem_op.getSizeAttr()) {
    if (auto int_attr = mlir::dyn_cast<mlir::IntegerAttr>(size_attr)) {
      ctx.node_sizes[node_id] = int_attr.getValue().getSExtValue();
    }
  }
}

// Process an execution unit operation
void processExecutionUnitOp(mlir::ktdf_arch::ExecutionUnitOp exec_op,
                            DeviceInitContext& ctx) {
  // Skip load/store units - they are modeled as transfer units on edges
  if (exec_op.getLoadStoreAttr() && exec_op.getLoadStore()) {
    // Don't create a node, just track the value for datapath processing
    ctx.load_store_units.insert(exec_op.getResult());
    return;
  }

  auto kind = exec_op.getKind();
  // Use a default kind if not specified
  if (!kind) {
    kind = mlir::StringAttr::get(exec_op.getContext(), "exec_unit");
  }

  auto node_id = ctx.graph.addNode(
      kind, RoutingGraph::ResourceNode::ResourceKind::Compute);
  ctx.value_to_node_id[exec_op.getResult()] = node_id;
}

// Forward declaration for mutual recursion
void processRegion(mlir::Region& region, DeviceInitContext& ctx);

// Process a group operation
void processGroupOp(mlir::ktdf_arch::GroupOp group_op, DeviceInitContext& ctx) {
  auto group_kind = group_op.getKind();

  // Flatten groups: only process first instance of each kind
  if (group_kind && ctx.processed_group_kinds.contains(group_kind)) {
    return;  // Skip this group, already processed
  }

  if (group_kind) {
    ctx.processed_group_kinds.insert(group_kind);
  }

  // Map shared memory block arguments to their operands
  // This allows datapaths inside the group to reference shared resources
  auto& block = group_op.getRegion().front();
  for (auto [operand, block_arg] :
       llvm::zip(group_op.getSharedMemory(), block.getArguments())) {
    auto it = ctx.value_to_node_id.find(operand);
    assert(it != ctx.value_to_node_id.end() &&
           "Shared memory operand must be defined before group");
    ctx.value_to_node_id[block_arg] = it->second;
  }

  // Recursively process the group's body
  processRegion(group_op.getRegion(), ctx);

  // Map yielded values to their definitions inside the group
  if (auto yield_op = mlir::dyn_cast<mlir::ktdf_arch::YieldOp>(
          group_op.getRegion().front().getTerminator())) {
    for (auto [result, operand] :
         llvm::zip(group_op.getResults(), yield_op.getOperands())) {
      // The yielded operand should already be mapped
      auto it = ctx.value_to_node_id.find(operand);
      assert(it != ctx.value_to_node_id.end() &&
             "Yielded value must be defined in group body");
      ctx.value_to_node_id[result] = it->second;
    }
  }
}

// Process a datapath operation
//
// Load/store units (exec_units with load_store attribute) are modeled as
// transfer units on edges, not as separate nodes. When we encounter a pattern:
//   A -> LS_unit -> B
// We create a single edge: A -> B with LS_unit's kind as the transfer unit.
//
void processDatapathOp(mlir::ktdf_arch::DatapathOp datapath_op,
                       DeviceInitContext& ctx) {
  auto source_val = datapath_op.getSource();
  auto target_val = datapath_op.getTarget();

  // Check if source or target is a load/store unit
  bool source_is_ls = ctx.load_store_units.contains(source_val);
  bool target_is_ls = ctx.load_store_units.contains(target_val);

  // Case 1: Source is a load/store unit (second half of pattern: LS -> B)
  if (source_is_ls) {
    // Find all original sources from the first half (A1 -> LS, A2 -> LS, ...)
    auto orig_sources_it = ctx.ls_unit_sources.find(source_val);
    assert(orig_sources_it != ctx.ls_unit_sources.end() &&
           "Load/store unit must have sources from previous datapaths");

    // Get the load/store unit op to access its kind
    auto ls_unit_op =
        source_val.getDefiningOp<mlir::ktdf_arch::ExecutionUnitOp>();
    assert(ls_unit_op && "Load/store value must be defined by ExecutionUnitOp");

    auto target_it = ctx.value_to_node_id.find(target_val);
    assert(target_it != ctx.value_to_node_id.end() &&
           "Datapath target must be a defined resource");
    auto target_id = target_it->second;

    // Use the load/store unit's kind as the transfer unit
    auto transfer_unit = ls_unit_op.getKind();

    // Create edges from all sources to the target: A1 -> B, A2 -> B, ...
    for (auto orig_source_val : orig_sources_it->second) {
      auto source_it = ctx.value_to_node_id.find(orig_source_val);
      assert(source_it != ctx.value_to_node_id.end() &&
             "Original datapath source must be a defined resource");

      auto source_id = source_it->second;

      // Create edge: Ai -> B with LS_unit as transfer unit
      ctx.graph.addEdge(source_id, target_id, transfer_unit, 1);
    }
    return;
  }

  // Case 2: Target is a load/store unit (first half of pattern: A -> LS)
  if (target_is_ls) {
    // Store the source for when we process the second half (LS -> B)
    // Multiple sources can feed into the same LS unit
    ctx.ls_unit_sources[target_val].push_back(source_val);
    return;
  }

  // Case 3: Neither is a load/store unit - direct connection
  auto source_it = ctx.value_to_node_id.find(source_val);
  auto target_it = ctx.value_to_node_id.find(target_val);

  assert(source_it != ctx.value_to_node_id.end() &&
         "Datapath source must be a defined resource");
  assert(target_it != ctx.value_to_node_id.end() &&
         "Datapath target must be a defined resource");

  auto source_id = source_it->second;
  auto target_id = target_it->second;

  // Check for self-loops
  assert(source_id != target_id &&
         "Datapath cannot connect a resource to itself");

  // Extract transfer unit from kind attribute (if any)
  mlir::Attribute transfer_unit = datapath_op.getKind();

  // Add the edge
  ctx.graph.addEdge(source_id, target_id, transfer_unit, 1);
}

// Process a region recursively
void processRegion(mlir::Region& region, DeviceInitContext& ctx) {
  for (mlir::Operation& op : region.front()) {
    if (auto mem_op = mlir::dyn_cast<mlir::ktdf_arch::MemoryOp>(&op)) {
      processMemoryOp(mem_op, ctx);
    } else if (auto exec_op =
                   mlir::dyn_cast<mlir::ktdf_arch::ExecutionUnitOp>(&op)) {
      processExecutionUnitOp(exec_op, ctx);
    } else if (auto group_op = mlir::dyn_cast<mlir::ktdf_arch::GroupOp>(&op)) {
      processGroupOp(group_op, ctx);
    } else if (auto datapath_op =
                   mlir::dyn_cast<mlir::ktdf_arch::DatapathOp>(&op)) {
      processDatapathOp(datapath_op, ctx);
    }
    // Ignore other operations (like YieldOp, which is handled in
    // processGroupOp)
  }
}

}  // namespace

RoutingGraph::RoutingGraph(mlir::ktdf_arch::DeviceOp declaration,
                           mlir::AnalysisManager& analyses)
    : DeviceView(declaration, analyses) {
  initializeFromDevice(getDevice());
}

// Initialize a RoutingGraph from a ktdf_arch::DeviceOp.
//
// This function walks the operations in a DeviceOp and constructs an equivalent
// RoutingGraph representation. The graph must be empty before calling this
// function.
//
// Group operations are flattened by processing only the first instance of each
// group kind. This creates a representative view of the hierarchical structure
// where symmetric groups are deduplicated.
//
// Properties currently extracted:
//   - Memory nodes: kind (memory space), size (if specified)
//   - Execution unit nodes: kind (compute unit type)
//   - Datapath edges: kind (transfer unit), source, target
//
// Properties currently skipped:
//   - features, bandwidth, overlaps (not yet modeled in RoutingGraph)
//
// Example usage:
//   auto device = ...; // Get ktdf_arch::DeviceOp
//   RoutingGraph graph(device);
//   graph.print(llvm::outs());
//
void RoutingGraph::initializeFromDevice(mlir::ktdf_arch::Device& device) {
  // Precondition: graph must be empty
  assert(nodes_.empty() && "Graph must be empty before initialization");
  assert(adjacency_.empty() && "Graph must be empty before initialization");

  // Create context for tracking state during initialization
  DeviceInitContext ctx(*this);

  // Process the device body
  processRegion(device.getBodyRegion(), ctx);

  // Apply node sizes collected during traversal
  for (const auto& [node_id, size] : ctx.node_sizes) {
    nodes_[node_id].total_capacity_in_bytes = size;
    nodes_[node_id].total_reserved_in_bytes = 0;
  }
}

RoutingGraph::NodeId RoutingGraph::addNode(ResourceType resource,
                                           ResourceNode::ResourceKind kind) {
  NodeId node_id = next_node_id_++;

  nodes_.insert(
      {node_id, {node_id, resource, kind, std::nullopt, std::nullopt}});
  return node_id;
}

void RoutingGraph::addEdge(NodeId source, NodeId target,
                           ResourceType transfer_unit, unsigned cost) {
  assert(hasNode(source) && "source node must exist before adding edge");
  assert(hasNode(target) && "target node must exist before adding edge");

  adjacency_[source].push_back({source, target, transfer_unit, cost});
}

bool RoutingGraph::hasNode(NodeId node_id) const {
  return nodes_.count(node_id) != 0;
}

std::optional<RoutingGraph::ResourceNode> RoutingGraph::getNode(
    NodeId node_id) const {
  auto it = nodes_.find(node_id);
  if (it == nodes_.end()) return std::nullopt;
  return it->second;
}

RoutingGraph::NeighborList RoutingGraph::getNeighbors(NodeId node_id) const {
  NeighborList neighbors;
  auto it = adjacency_.find(node_id);
  if (it == adjacency_.end()) return neighbors;

  for (const EdgeInfo& edge : it->second) {
    neighbors.push_back(edge.target);
  }
  return neighbors;
}

std::optional<RoutingGraph::EdgeInfo> RoutingGraph::getEdgeInfo(
    NodeId source, NodeId target) const {
  auto it = adjacency_.find(source);
  if (it == adjacency_.end()) return std::nullopt;

  for (const EdgeInfo& edge : it->second) {
    if (edge.target == target) return edge;
  }
  return std::nullopt;
}

std::optional<RoutingGraph::Path> RoutingGraph::findShortestPath(
    NodeId source, NodeId target) const {
  if (!hasNode(source) || !hasNode(target)) return std::nullopt;

  std::queue<NodeId> worklist;
  llvm::DenseMap<NodeId, NodeId> predecessor;
  llvm::DenseMap<NodeId, bool> visited;

  worklist.push(source);
  visited[source] = true;

  while (!worklist.empty()) {
    NodeId current = worklist.front();
    worklist.pop();

    if (current == target) break;

    auto neighbors = getNeighbors(current);
    for (NodeId neighbor : neighbors) {
      if (visited.lookup(neighbor)) continue;
      visited[neighbor] = true;
      predecessor[neighbor] = current;
      worklist.push(neighbor);
    }
  }

  if (!visited.lookup(target)) return std::nullopt;

  Path reversed_path;
  NodeId current = target;
  reversed_path.push_back(current);

  while (current != source) {
    auto pred_it = predecessor.find(current);
    if (pred_it == predecessor.end()) return std::nullopt;
    current = pred_it->second;
    reversed_path.push_back(current);
  }

  Path path;
  path.reserve(reversed_path.size());
  for (auto it = reversed_path.rbegin(); it != reversed_path.rend(); ++it) {
    path.push_back(*it);
  }
  return path;
}

RoutingGraph::NodeId RoutingGraph::getNodeIdForResource(
    ResourceType resource) const {
  // Search through nodes to find the one with matching resource
  // Attributes are uniqued, so we can use direct equality comparison
  for (const auto& [node_id, node] : nodes_) {
    if (node.resource == resource) {
      return node_id;
    }
  }
  llvm_unreachable("resource not found in graph");
}

std::optional<RoutingGraph::EdgeInfo> RoutingGraph::getEdgeInfoForResources(
    ResourceType source, ResourceType dest) const {
  NodeId source_node = getNodeIdForResource(source);
  NodeId dest_node = getNodeIdForResource(dest);
  return getEdgeInfo(source_node, dest_node);
}

llvm::SmallVector<RoutingGraph::NodeId> RoutingGraph::getAllNodeIds() const {
  llvm::SmallVector<NodeId> node_ids;
  node_ids.reserve(nodes_.size());
  for (const auto& [node_id, _] : nodes_) {
    node_ids.push_back(node_id);
  }
  return node_ids;
}

void RoutingGraph::dump() const { print(llvm::dbgs()); }

void RoutingGraph::print(llvm::raw_ostream& os) const {
  os << "RoutingGraph {\n";
  os << "  Nodes:\n";
  for (const auto& [node_id, node] : nodes_) {
    os << "    - " << node_id << ": ";
    if (node.resource) {
      node.resource.print(os);
    } else {
      os << "<null>";
    }
    os << " [" << stringifyResourceKind(node.kind) << "]";

    // Print capacity and reserved info for Memory nodes
    if (node.kind == ResourceNode::ResourceKind::Memory) {
      bool has_info = false;
      if (node.total_capacity_in_bytes || node.total_reserved_in_bytes) {
        os << " (";
        if (node.total_capacity_in_bytes) {
          os << "capacity=" << *node.total_capacity_in_bytes << " bytes";
          has_info = true;
        }
        if (node.total_reserved_in_bytes) {
          if (has_info) os << ", ";
          os << "reserved=" << *node.total_reserved_in_bytes << " bytes";
        }
        os << ")";
      }
    }
    os << "\n";
  }

  os << "  Edges:\n";
  for (const auto& [node_id, node] : nodes_) {
    auto it = adjacency_.find(node_id);
    if (it == adjacency_.end()) continue;
    for (const EdgeInfo& edge : it->second) {
      auto source_node = getNode(edge.source);
      auto target_node = getNode(edge.target);
      assert(source_node && target_node &&
             "edge endpoint must reference a node");

      os << "    - " << edge.source << ": ";
      if (source_node->resource) {
        source_node->resource.print(os);
      } else {
        os << "<null>";
      }
      os << " -> " << edge.target << ": ";
      if (target_node->resource) {
        target_node->resource.print(os);
      } else {
        os << "<null>";
      }
      os << " [";
      if (edge.transfer_unit) {
        os << "transfer_unit=";
        edge.transfer_unit.print(os);
        os << ", ";
      }
      os << "cost=" << edge.cost << "]\n";
    }
  }
  os << "}\n";
}

RoutingGraph::ResourceNode::ResourceKind RoutingGraph::inferResourceKind(
    ResourceType resource) {
  if (!resource) {
    llvm_unreachable("cannot infer kind from null attribute");
  }

  // KtdpMemorySpaceAttr indicates memory resource
  if (mlir::isa<mlir::ktdp::KtdpMemorySpaceAttr>(resource)) {
    return ResourceNode::ResourceKind::Memory;
  }

  // StringAttr indicates compute resource (by convention)
  if (mlir::isa<mlir::StringAttr>(resource)) {
    return ResourceNode::ResourceKind::Compute;
  }

  llvm_unreachable("unknown resource attribute type");
}

llvm::StringRef RoutingGraph::stringifyResourceKind(
    ResourceNode::ResourceKind kind) {
  switch (kind) {
    case ResourceNode::ResourceKind::Memory:
      return "Memory";
    case ResourceNode::ResourceKind::Compute:
      return "Compute";
  }
  llvm_unreachable("unknown RoutingGraph::ResourceNode::ResourceKind");
}

std::optional<RoutingGraph::ResourceNode> RoutingGraph::getNode(
    ResourceType resource) const {
  NodeId node_id = getNodeIdForResource(resource);
  return getNode(node_id);
}

std::optional<size_t> RoutingGraph::getResourceCapacity(
    ResourceType resource) const {
  auto node = getNode(resource);
  if (!node) return std::nullopt;
  return node->total_capacity_in_bytes;
}

std::optional<size_t> RoutingGraph::getResourceReserved(
    ResourceType resource) const {
  auto node = getNode(resource);
  if (!node) return std::nullopt;
  return node->total_reserved_in_bytes;
}

std::optional<size_t> RoutingGraph::getResourceAvailable(
    ResourceType resource) const {
  auto capacity = getResourceCapacity(resource);
  auto reserved = getResourceReserved(resource);

  if (!capacity || !reserved) return std::nullopt;

  return *capacity - *reserved;
}

// Made with Bob