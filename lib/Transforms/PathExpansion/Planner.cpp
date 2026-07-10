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

#include "dataflow-scheduler/Transforms/PathExpansion/Planner.h"

#include "Ktdp/KtdpAttrs.hpp"
#include "dataflow-scheduler/Analysis/ArchViews/RoutingGraph.h"
#include "dataflow-scheduler/Analysis/PipelineTree.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "path-expansion-planner"

using namespace scheduler;

namespace scheduler {

namespace {

/// Helper to create an identity affine map for intermediate buffer accesses
/// This is used when we need an affine map for a memref buffer but don't have
/// one from a neighboring transfer (e.g., when the neighbor uses a FIFO)
static mlir::AffineMap createAffineMapForIntermediateBuffer(
    const PrivateResourceSpec* buffer, mlir::MLIRContext* context) {
  return mlir::AffineMap::getMultiDimIdentityMap(buffer->shape.size(), context);
}

/// Helper to validate FIFO slot index assignment
/// Checks that the assigned slot index matches the original result index
/// from the template transfer operation
void validateFifoSlotIndex(mlir::Value orig_fifo_value,
                           size_t assigned_slot_idx,
                           const PrivateResourceSpec* fifo_spec) {
  if (auto orig_private_result =
          mlir::dyn_cast<mlir::OpResult>(orig_fifo_value)) {
    // Find which result index this was in the original private op
    unsigned orig_result_idx = orig_private_result.getResultNumber();
    auto priv_op = orig_fifo_value.getDefiningOp<mlir::ktdf::PrivateOp>();
    assert(priv_op);
    mlir::ktdf::PrivateYieldOp yield = priv_op.getYieldOp();
    assert(orig_result_idx < yield.getOperands().size());
    mlir::Value fifo_slot_value = yield.getOperands()[orig_result_idx];
    mlir::OpResult fifo_alloc_result =
        mlir::dyn_cast<mlir::OpResult>(fifo_slot_value);
    assert(fifo_alloc_result);

    // The slot index we assign should match the original result index
    // This ensures we maintain the same slot ordering as the original
    assert(assigned_slot_idx == fifo_alloc_result.getResultNumber() &&
           "expected result index to match our calculated fifo slot index");
  }
  // Sanity check: slot index should be within allocated slots
  assert(assigned_slot_idx < fifo_spec->elements_per_slot.size() &&
         "Slot index should be within the allocated slots for this FIFO type");
}

/// Unified resource inference helper
/// Extracts resource attributes directly from types and specs
class ResourceInferrer {
 public:
  /// Infer resource attribute from a PrivateResourceSpec
  static ResourceType fromPrivateSpec(
      const PrivateResourceSpec* spec, bool is_producer_side,
      const scheduler::arch_view::RoutingGraph& arch_graph) {
    if (!spec) return nullptr;

    if (spec->kind == PrivateResourceSpec::Kind::kMemoryBuffer) {
      return spec->memory_resource;
    } else if (spec->kind == PrivateResourceSpec::Kind::kFifo) {
      // For FIFO: producer is on src side, consumer is on dest side
      ResourceType transfer_unit =
          is_producer_side ? spec->fifo_src : spec->fifo_dest;
      return findMemoryResourceForTransferUnit(transfer_unit, arch_graph);
    }

    return nullptr;
  }

  /// Helper to find transfer unit for a memory resource in a specific direction
  /// @param maybe_memory_resource The memory resource to find transfer unit for
  /// @param is_load_direction True for load (memory->compute), false for store
  /// @param arch_graph The architecture graph to search
  /// @return The transfer unit attribute, or memory_resource if not found
  /// FIXME: This is temporary code to find the memory resource by walking
  /// the architecture graph edges. This should be revisited in the context
  /// of evaluating representing load/store units in the routing view.
  static ResourceType findTransferUnitForMemory(
      ResourceType maybe_memory_resource, bool is_load_direction,
      const scheduler::arch_view::RoutingGraph& arch_graph) {
    // Check if this is a memory resource
    auto memory_node_opt = arch_graph.getNode(maybe_memory_resource);
    if (!memory_node_opt || memory_node_opt->kind !=
                                scheduler::arch_view::RoutingGraph::
                                    ResourceNode::ResourceKind::Memory) {
      // Not a memory resource, return as-is
      return maybe_memory_resource;
    }

    scheduler::arch_view::RoutingGraph::NodeId memory_node_id =
        memory_node_opt->id;

    // Walk edges from/to this memory node to find the appropriate transfer unit
    // For load direction: memory -> compute (outgoing edges from memory)
    // For store direction: compute -> memory (incoming edges to memory)
    // Important: We skip edges that have memory resources on either side
    // (e.g., L1->SFU not L1->DDR)
    if (is_load_direction) {
      // Check outgoing edges from memory node to non-memory resources
      for (auto neighbor_id : arch_graph.getNeighbors(memory_node_id)) {
        auto neighbor_node = arch_graph.getNode(neighbor_id);
        // Only consider edges to non-memory resources
        if (neighbor_node && neighbor_node->kind !=
                                 scheduler::arch_view::RoutingGraph::
                                     ResourceNode::ResourceKind::Memory) {
          auto edge_info = arch_graph.getEdgeInfo(memory_node_id, neighbor_id);
          if (edge_info && edge_info->transfer_unit) {
            return edge_info->transfer_unit;
          }
        }
      }
    } else {
      // Check incoming edges to memory node from non-memory resources
      for (auto node_id : arch_graph.getAllNodeIds()) {
        auto src_node = arch_graph.getNode(node_id);
        // Only consider edges from non-memory resources
        if (src_node && src_node->kind !=
                            scheduler::arch_view::RoutingGraph::ResourceNode::
                                ResourceKind::Memory) {
          for (auto neighbor_id : arch_graph.getNeighbors(node_id)) {
            if (neighbor_id != memory_node_id) continue;
            auto edge_info = arch_graph.getEdgeInfo(node_id, memory_node_id);
            if (edge_info && edge_info->transfer_unit) {
              return edge_info->transfer_unit;
            }
          }
        }
      }
    }

    // No transfer unit found, return the memory resource as-is
    return maybe_memory_resource;
  }

  /// Helper to find memory resource from a transfer unit by walking the
  /// architecture graph edges
  /// FIXME: This is temporary code to find the memory resource by walking
  /// the architecture graph edges. This should be revisited in the context
  /// of evaluating representing load/store units in the routing view.
  static ResourceType findMemoryResourceForTransferUnit(
      ResourceType maybe_transfer_unit,
      const scheduler::arch_view::RoutingGraph& arch_graph) {
    // Try to find an edge with this transfer unit and extract the memory node
    for (auto node_id : arch_graph.getAllNodeIds()) {
      auto node = arch_graph.getNode(node_id);
      if (!node) continue;

      // Use getNeighbors for O(N) search instead of O(N^2)
      for (auto neighbor_id : arch_graph.getNeighbors(node_id)) {
        auto edge_info = arch_graph.getEdgeInfo(node_id, neighbor_id);
        if (!edge_info || edge_info->transfer_unit != maybe_transfer_unit)
          continue;

        // Found an edge with matching transfer unit
        // Check which endpoint is a memory node
        auto src_node = arch_graph.getNode(edge_info->source);
        auto dest_node = arch_graph.getNode(edge_info->target);

        bool src_is_memory =
            src_node && src_node->kind ==
                            scheduler::arch_view::RoutingGraph::ResourceNode::
                                ResourceKind::Memory;
        bool dest_is_memory =
            dest_node && dest_node->kind ==
                             scheduler::arch_view::RoutingGraph::ResourceNode::
                                 ResourceKind::Memory;

        assert(!(src_is_memory && dest_is_memory) &&
               "this mechanism does not work when both source and destination "
               "are memory nodes");

        if (src_is_memory) {
          return src_node->resource;
        } else if (dest_is_memory) {
          return dest_node->resource;
        }
      }
    }

    // if no transfer unit edge matches, it is NOT a transfer unit, so just
    // return it as the resource.
    return maybe_transfer_unit;
  }

  /// Infer resource attribute from a stage by inspecting its first data
  /// transfer
  static ResourceType fromStage(
      mlir::ktdf::StageOp stage_op,
      const scheduler::arch_view::RoutingGraph& arch_graph) {
    if (!stage_op) return nullptr;

    // Find the first data transfer operation in the stage
    mlir::ktdf::DataTransferOp first_transfer = nullptr;
    stage_op.walk([&](mlir::ktdf::DataTransferOp transfer_op) {
      if (!first_transfer) {
        first_transfer = transfer_op;
        return mlir::WalkResult::interrupt();
      }
      return mlir::WalkResult::advance();
    });

    if (!first_transfer) return nullptr;

    // Check if source is a FIFO - if so, this stage is consuming from that FIFO
    if (auto source_type = first_transfer.getSource().getType()) {
      if (auto fifo_type =
              mlir::dyn_cast<mlir::ktdf::FifoSlotType>(source_type)) {
        // Stage is consuming from FIFO, so it's on the consumer (dest) side
        // The dest attribute might be a transfer unit like "L1LU", but we need
        // the memory resource like "L1""
        return findMemoryResourceForTransferUnit(fifo_type.getDest(),
                                                 arch_graph);
      }
    }

    // Check if destination is a FIFO - if so, this stage is producing to that
    // FIFO
    if (auto dest_type = first_transfer.getDestination().getType()) {
      if (auto fifo_type =
              mlir::dyn_cast<mlir::ktdf::FifoSlotType>(dest_type)) {
        // Stage is producing to FIFO, so it's on the producer (src) side
        // The src attribute might be a transfer unit like "L1LU", but we need
        // the memory resource like "L1"
        return findMemoryResourceForTransferUnit(fifo_type.getSrc(),
                                                 arch_graph);
      }
    }

    return nullptr;
  }

  /// Infer resource attribute from a Type (MemRefType or FifoSlotType)
  static ResourceType fromType(
      mlir::Type type, bool is_producer_side,
      const scheduler::arch_view::RoutingGraph& arch_graph) {
    if (auto memref_type = mlir::dyn_cast<mlir::MemRefType>(type)) {
      return memref_type.getMemorySpace();
    } else if (auto fifo_type =
                   mlir::dyn_cast<mlir::ktdf::FifoSlotType>(type)) {
      // For FIFO: producer is on src side, consumer is on dest side
      ResourceType maybe_transfer_unit =
          is_producer_side ? fifo_type.getSrc() : fifo_type.getDest();
      return findMemoryResourceForTransferUnit(maybe_transfer_unit, arch_graph);
    }
    return nullptr;
  }
};

/// Helper to check if a stage is an intermediate (synthetic) stage
inline bool isIntermediateStage(const StageNode* stage) {
  return stage->getOperation() == nullptr;
}

/// Helper to determine a stage's resource from its summary and position
ResourceType determineStageResource(
    StageNode* stage, size_t stage_index, size_t total_stages,
    const StageSummary& summary,
    const scheduler::arch_view::RoutingGraph& arch_graph) {
  // First, check if stage has an anchor resource (compute stages)
  if (summary.anchor_resource) {
    return summary.anchor_resource;
  }

  // For first stage, use input endpoint
  if (stage_index == 0 && !summary.input_endpoints.empty()) {
    return summary.input_endpoints[0];
  }

  // For last stage, use output endpoint
  if (stage_index == total_stages - 1 && !summary.output_endpoints.empty()) {
    return summary.output_endpoints[0];
  }

  // For intermediate stages without anchor resources, try to infer from
  // the stage's data transfer operations
  auto stage_op =
      mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(stage->getOperation());
  if (stage_op) {
    return ResourceInferrer::fromStage(stage_op, arch_graph);
  }

  return nullptr;
}

//===----------------------------------------------------------------------===//
// Debug Output Helpers
//===----------------------------------------------------------------------===//

/// Print stage list with classifications
static void debugPrintStageList(llvm::raw_ostream& os,
                                llvm::ArrayRef<StageNode*> stages,
                                const PathExpansionPlan* plan,
                                llvm::StringRef step_name) {
  os << "\n=== " << step_name << " ===\n";
  os << "Total stages: " << stages.size() << "\n";
  for (StageNode* stage : stages) {
    os << "  Stage " << stage->getStageId() << ": ";

    if (isIntermediateStage(stage))
      os << "(synthetic, unmaterialized)";
    else
      os << "(original)";

    // Print materialization info if available
    if (plan && plan->stage_info.count(stage)) {
      const StageMaterializationInfo& info = plan->stage_info.at(stage);
      info.print(os);
    }
    os << "\n";
  }
}

/// Print resource specs summary
static void debugPrintResourceSpecs(llvm::raw_ostream& os,
                                    const PrivateResourceFactory& factory) {
  os << "Private resources:\n";
  size_t mem_count = 0, fifo_count = 0;
  for (const auto& spec_ptr : factory.getSpecs()) {
    if (spec_ptr->kind == PrivateResourceSpec::Kind::kMemoryBuffer) {
      mem_count++;
    } else if (spec_ptr->kind == PrivateResourceSpec::Kind::kFifo) {
      fifo_count++;
    }
  }
  os << "  Memory buffers: " << mem_count << "\n";
  os << "  FIFOs: " << fifo_count << "\n";
}

}  // anonymous namespace

//===----------------------------------------------------------------------===//
// Print Functions for Data Structures
//===----------------------------------------------------------------------===//

void PrivateResourceSpec::print(llvm::raw_ostream& os) const {
  switch (kind) {
    case Kind::kMemoryBuffer:
      os << "MemoryBuffer(resource=" << [&]() {
        std::string s;
        llvm::raw_string_ostream os(s);
        (memory_resource).print(os);
        return os.str();
      }() << ", shape=[";
      for (size_t i = 0; i < shape.size(); ++i) {
        os << shape[i];
        if (i + 1 < shape.size()) os << "x";
      }
      os << "], element_type=" << element_type << ")";
      break;
    case Kind::kFifo:
      os << "Fifo(src=" << fifo_src << ", dest=" << fifo_dest << ", slots=[";
      for (size_t i = 0; i < elements_per_slot.size(); ++i) {
        os << elements_per_slot[i];
        if (i + 1 < elements_per_slot.size()) os << ", ";
      }
      os << "], element_type=" << element_type << ")";
      break;
    case Kind::kUnknown:
      os << "Unknown";
      break;
  }
}

void PrivateResourceSpec::dump() const {
  print(llvm::dbgs());
  llvm::dbgs() << "\n";
}

void TransferMaterializationInfo::print(llvm::raw_ostream& os) const {
  os << "Transfer(" <<
      [&]() {
        std::string s;
        llvm::raw_string_ostream os(s);
        (source_resource).print(os);
        return os.str();
      }()
     << " -> " << [&]() {
          std::string s;
          llvm::raw_string_ostream os(s);
          (dest_resource).print(os);
          return os.str();
        }();

  if (source_private_resource) {
    os << ", priv_source=";
    source_private_resource->print(os);
  }

  if (dest_private_resource) {
    os << ", priv_dest=";
    dest_private_resource->print(os);
  }

  if (template_op) {
    os << ", template=";
    printLocation(os, template_op);
  } else {
    os << ", synthetic";
  }

  os << ")";
}

void TransferMaterializationInfo::dump() const {
  print(llvm::dbgs());
  llvm::dbgs() << "\n";
}

void TransferMaterializationInfo::printDebug(llvm::raw_ostream& os) const {
  os <<
      [&]() {
        std::string s;
        llvm::raw_string_ostream os(s);
        (source_resource).print(os);
        return os.str();
      }()
     << " -> " << [&]() {
          std::string s;
          llvm::raw_string_ostream os(s);
          (dest_resource).print(os);
          return os.str();
        }();

  if (source_private_resource) {
    os << " (source: ";
    if (source_private_resource->kind == PrivateResourceSpec::Kind::kFifo) {
      os << "FIFO slot " << source_slot_index;
    } else {
      os << "buffer";
    }
    os << ")";
  }

  if (dest_private_resource) {
    os << " (dest: ";
    if (dest_private_resource->kind == PrivateResourceSpec::Kind::kFifo) {
      os << "FIFO slot " << dest_slot_index;
    } else {
      os << "buffer";
    }
    os << ")";
  }
}

void StageMaterializationInfo::print(llvm::raw_ostream& os) const {
  os << " kind=";
  switch (kind) {
    case StageMaterializationInfo::Kind::kPreserveOriginal:
      os << "PreserveOriginal";
      break;
    case StageMaterializationInfo::Kind::kAdaptFifoKinds:
      os << "AdaptFifoKinds";
      break;
    case StageMaterializationInfo::Kind::kAdaptTransfer:
      os << "AdaptTransfer";
      break;
    case StageMaterializationInfo::Kind::kSyntheticTransfer:
      os << "SyntheticTransfer";
      break;
  }
  os << ", applicable_unit:";
  if (applicable_unit.has_value()) {
    if (auto str_attr = mlir::dyn_cast<mlir::StringAttr>(*applicable_unit)) {
      os << str_attr.getValue();
    } else {
      os << "<unknown>";
    }
  } else {
    os << "none";
  }
  os << ", transfers=" << transfers.size();

#if 1
  os << "\nTransfers:\n";
  for (const TransferMaterializationInfo* ti : transfers) {
    ti->print(os);
    os << "\n";
  }
#endif
}
void StageMaterializationInfo::dump() const {
  print(llvm::dbgs());
  llvm::dbgs() << "\n";
}

/// Analyze a single stage and create its summary
/// Extract anchor resource attribute from stage's applicable units
static ResourceType extractAnchorResource(mlir::ktdf::StageOp stage_op) {
  auto applicable_units = stage_op.getApplicableUnits();
  if (!applicable_units) {
    return nullptr;
  }

  assert(applicable_units->size() == 1 &&
         "path expansion currently does not handle nested pipelines with "
         "multi-unit stages");

  return applicable_units->getValue()[0];
}

/// Analyze transfer operation endpoints and add to summary
static void analyzeTransferEndpoints(
    mlir::ktdf::DataTransferOp transfer_op, StageSummary& summary,
    const scheduler::arch_view::RoutingGraph& arch_graph) {
  // Analyze source endpoint
  if (auto source_type = transfer_op.getSource().getType()) {
    if (auto resource =
            ResourceInferrer::fromType(source_type, true, arch_graph)) {
      summary.input_endpoints.push_back(resource);
    }
  }

  // Analyze destination endpoint
  if (auto dest_type = transfer_op.getDestination().getType()) {
    if (auto resource =
            ResourceInferrer::fromType(dest_type, false, arch_graph)) {
      summary.output_endpoints.push_back(resource);
    }
  }
}

/// Analyze stage body to determine operation types
static void analyzeStageOperations(
    mlir::ktdf::StageOp stage_op, StageSummary& summary, bool& has_compute,
    bool& has_transfer, const scheduler::arch_view::RoutingGraph& arch_graph) {
  stage_op.walk([&](mlir::Operation* op) {
    if (auto transfer_op = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(op)) {
      has_transfer = true;
      analyzeTransferEndpoints(transfer_op, summary, arch_graph);
      return mlir::WalkResult::advance();
    }

    if (mlir::isa<mlir::linalg::LinalgOp>(op)) {
      has_compute = true;
    }
    return mlir::WalkResult::advance();
  });
}

llvm::FailureOr<StageSummary> analyzeStage(
    StageNode* stage_node,
    const scheduler::arch_view::RoutingGraph& arch_graph) {
  StageSummary summary;
  summary.node = stage_node;

  auto stage_op =
      mlir::dyn_cast<mlir::ktdf::StageOp>(stage_node->getOperation());
  if (!stage_op) {
    return mlir::failure();
  }

  // Extract anchor resource from applicable units
  summary.anchor_resource = extractAnchorResource(stage_op);

  // Analyze stage body for compute and transfer operations
  bool has_compute = false;
  bool has_transfer = false;
  analyzeStageOperations(stage_op, summary, has_compute, has_transfer,
                         arch_graph);

  summary.is_transfer_only = has_transfer && !has_compute;
  summary.is_compute_containing = has_compute;

  return summary;
}

//===----------------------------------------------------------------------===//
// Stage DAG Validation Functions
//===----------------------------------------------------------------------===//

mlir::LogicalResult validateLinearChain(
    llvm::ArrayRef<StageNode*> sorted_stages) {
  if (sorted_stages.empty()) {
    return mlir::success();  // Empty is trivially linear
  }

  // For a linear chain, each stage (except the last) should have exactly one
  // dependency and each stage (except the first) should be depended upon by
  // exactly one other stage

  // Count how many stages depend on each stage
  llvm::DenseMap<StageNode*, int> dependent_count;
  for (StageNode* stage : sorted_stages) {
    dependent_count[stage] = 0;
  }

  for (StageNode* stage : sorted_stages) {
    const llvm::SmallVector<StageNode*>& deps = stage->getDependencies();

    // Each stage should have at most one outgoing dependency
    if (deps.size() > 1) {
      LDBG(1) << "Stage " << stage->getStageId()
              << " has multiple outgoing dependencies (branching)";
      return mlir::failure();
    }

    // Count dependents
    for (StageNode* dep : deps) {
      dependent_count[dep]++;
    }
  }

  // Check that each stage (except source) has exactly one incoming dependency
  int source_count = 0;
  int sink_count = 0;

  for (StageNode* stage : sorted_stages) {
    int incoming = dependent_count[stage];
    const llvm::SmallVector<StageNode*>& outgoing = stage->getDependencies();

    if (incoming == 0) {
      source_count++;
    }
    if (outgoing.empty()) {
      sink_count++;
    }

    // In a linear chain, each non-source stage should have exactly 1 incoming
    // edge
    if (incoming > 1) {
      LDBG(1) << "Stage " << stage->getStageId()
              << " has multiple incoming dependencies (merging)";
      return mlir::failure();
    }
  }

  // Should have exactly one source and one sink
  if (source_count != 1 || sink_count != 1) {
    LDBG(1) << "Expected 1 source and 1 sink, got " << source_count
            << " sources and " << sink_count << " sinks";
    return mlir::failure();
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Stage Analysis Functions
//===----------------------------------------------------------------------===//

/// Overload that takes sorted stages to ensure path inference order matches DAG
/// order
llvm::FailureOr<llvm::SmallVector<StageSummary>> analyzeStages(
    llvm::ArrayRef<StageNode*> sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph) {
  llvm::SmallVector<StageSummary> summaries;

  for (StageNode* stage : sorted_stages) {
    llvm::FailureOr<StageSummary> summary_or = analyzeStage(stage, arch_graph);
    if (mlir::failed(summary_or)) {
      return mlir::failure();
    }
    summaries.push_back(*summary_or);
  }

  return summaries;
}

llvm::FailureOr<llvm::SmallVector<ResourceType>> inferOriginalPath(
    llvm::ArrayRef<StageSummary> summaries) {
  llvm::SmallVector<ResourceType> path;

  if (summaries.empty()) return mlir::failure();

  // Add anchor resources from compute stages
  for (const auto& summary : summaries) {
    if (summary.anchor_resource) {
      // Only add if different from last resource in path
      if (path.empty() || path.back() != summary.anchor_resource) {
        path.push_back(summary.anchor_resource);
      }
    } else if (!(summary.input_endpoints.empty() &&
                 summary.output_endpoints.empty())) {
      assert(summary.input_endpoints.size() == summary.output_endpoints.size());
      for (size_t i = 0; i < summary.input_endpoints.size(); i++) {
        if (path.empty() || path.back() != summary.input_endpoints[i])
          path.push_back(summary.input_endpoints[i]);
      }
      for (size_t i = 0; i < summary.output_endpoints.size(); i++) {
        if (path.empty() || path.back() != summary.output_endpoints[i])
          path.push_back(summary.output_endpoints[i]);
      }
    }
  }

  return path;
}

//===----------------------------------------------------------------------===//
// Step 1: Connection Legalization
//===----------------------------------------------------------------------===//

/// Insert intermediate stages for an illegal hop between two stages
static void insertIntermediateStages(
    PipelineTree& tree, PipelineNode* pipeline, ResourceType source_resource,
    ResourceType dest_resource,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan, int& next_stage_id,
    StageNode*& last_inserted_stage) {
  // Find shortest path between resources
  scheduler::arch_view::RoutingGraph::NodeId source_node =
      arch_graph.getNodeIdForResource(source_resource);
  scheduler::arch_view::RoutingGraph::NodeId dest_node =
      arch_graph.getNodeIdForResource(dest_resource);

  std::optional<llvm::SmallVector<scheduler::arch_view::RoutingGraph::NodeId>>
      path = arch_graph.findShortestPath(source_node, dest_node);

  assert(path && "No path found between stages");
  assert(path->size() >= 3 && "Expected at least one intermediate node");

  LLVM_DEBUG(
      llvm::dbgs() << "  Illegal hop from " <<
      [&]() {
        std::string s;
        llvm::raw_string_ostream os(s);
        (source_resource).print(os);
        return os.str();
      }() << " to " <<
      [&]() {
        std::string s;
        llvm::raw_string_ostream os(s);
        (dest_resource).print(os);
        return os.str();
      }() << ", inserting "
                   << (path->size() - 2) << " intermediate stage(s)\n");

  // Insert intermediate stages for each intermediate node in the path
  // Skip first (source) and last (dest) nodes
  for (size_t i = 1; i < path->size() - 1; ++i) {
    scheduler::arch_view::RoutingGraph::NodeId intermediate_node = (*path)[i];
    std::optional<scheduler::arch_view::RoutingGraph::ResourceNode>
        intermediate_res_node = arch_graph.getNode(intermediate_node);

    assert(intermediate_res_node && "Intermediate node not found in graph");

    ResourceType intermediate_resource = intermediate_res_node->resource;

    // Create new intermediate stage with null operation (unmaterialized)
    StageNode* intermediate_stage =
        tree.createStageNode(nullptr, next_stage_id++);

    LLVM_DEBUG(llvm::dbgs() << "    Creating intermediate stage "
                            << intermediate_stage->getStageId() << " at " <<
               [&]() {
                 std::string s;
                 llvm::raw_string_ostream os(s);
                 (intermediate_resource).print(os);
                 return os.str();
               }() << "\n");

    // Create minimal materialization info for the intermediate stage
    // Kind will be set to kSyntheticTransfer, transfers filled in Step 4
    StageMaterializationInfo intermediate_info;
    intermediate_info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;
    intermediate_info.stage_resource = intermediate_resource;

    plan->stage_info[intermediate_stage] = intermediate_info;

    // Insert the intermediate stage after the last inserted stage
    pipeline->insertChildNode(intermediate_stage, last_inserted_stage);
    last_inserted_stage = intermediate_stage;
  }
}

/// Step 1: Identify illegal connections and insert intermediate stages
static llvm::FailureOr<llvm::SmallVector<StageNode*>> legalizeStageConnections(
    PipelineTree& tree, PipelineNode* pipeline,
    llvm::ArrayRef<StageNode*> original_sorted_stages,
    llvm::ArrayRef<StageSummary> summaries,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan, int& next_stage_id) {
  StageNode* last_inserted_stage = nullptr;

  LDBG(1) << "Original stages: " << original_sorted_stages.size();

  for (size_t i = 0; i < original_sorted_stages.size(); ++i) {
    StageNode* current_stage = original_sorted_stages[i];
    const StageSummary& current_summary = summaries[i];

    // Determine the resource for this stage
    ResourceType current_resource =
        determineStageResource(current_stage, i, original_sorted_stages.size(),
                               current_summary, arch_graph);

    assert(current_resource && "Unable to determine resource for stage");

    // Update last_inserted_stage to current
    last_inserted_stage = current_stage;

    // Create minimal materialization info for original stage
    // Kind will be determined in later steps based on what needs adaptation
    StageMaterializationInfo info;
    info.kind =
        StageMaterializationInfo::Kind::kPreserveOriginal;  // Default, may be
                                                            // updated
    info.stage_resource = current_resource;  // Store resource for later steps

    // Set applicable_unit for compute stages
    if (current_summary.anchor_resource) {
      auto node = arch_graph.getNode(current_summary.anchor_resource);
      if (node && node->kind == scheduler::arch_view::RoutingGraph::
                                    ResourceNode::ResourceKind::Compute) {
        if (mlir::isa<mlir::StringAttr>(current_summary.anchor_resource)) {
          info.applicable_unit = current_summary.anchor_resource;
        }
      }
    }

    plan->stage_info[current_stage] = info;

    LLVM_DEBUG(llvm::dbgs() << "Stage " << current_stage->getStageId() << " at "
                            <<
               [&]() {
                 std::string s;
                 llvm::raw_string_ostream os(s);
                 (current_resource).print(os);
                 return os.str();
               }() << "\n");

    // Check if we need to expand the path to the next stage
    if (i + 1 < original_sorted_stages.size()) {
      const StageSummary& next_summary = summaries[i + 1];

      // Determine the resource for the next stage
      StageNode* next_stage = original_sorted_stages[i + 1];
      ResourceType next_resource = determineStageResource(
          next_stage, i + 1, original_sorted_stages.size(), next_summary,
          arch_graph);

      assert(next_resource && "Unable to determine resource for next stage");

      // Check if the hop is legal (direct edge exists)
      std::optional<scheduler::arch_view::RoutingGraph::EdgeInfo> direct_edge =
          arch_graph.getEdgeInfoForResources(current_resource, next_resource);

      if (!direct_edge) {
        // Hop is not legal - insert intermediate stages
        insertIntermediateStages(tree, pipeline, current_resource,
                                 next_resource, arch_graph, plan, next_stage_id,
                                 last_inserted_stage);
      } else {
        LDBG(1) << "  Legal hop to next stage";
      }
    }
  }

  // Rebuild dependencies: create linear chain through all stages
  llvm::SmallVector<StageNode*> all_stages = pipeline->getStages();

  // Clear existing dependencies
  for (StageNode* stage : all_stages) {
    stage->nullifyAllDependencies();
    stage->removeNullifiedDependencies();
  }

  // Get new topological sort to establish order
  llvm::FailureOr<llvm::SmallVector<StageNode*>> new_sorted_or =
      tree.topologicalSortStages(pipeline);
  if (mlir::failed(new_sorted_or)) {
    // If sort fails, manually create dependencies based on insertion order
    // This shouldn't happen but provides a fallback
    for (size_t i = 0; i < all_stages.size() - 1; ++i) {
      all_stages[i]->addDependency(all_stages[i + 1]);
    }
    new_sorted_or = tree.topologicalSortStages(pipeline);
    if (mlir::failed(new_sorted_or)) {
      LDBG(1) << "Failed to topologically sort after expansion";
      return mlir::failure();
    }
  }

  llvm::SmallVector<StageNode*> sorted = *new_sorted_or;

  // Create linear chain dependencies
  for (size_t i = 0; i < sorted.size() - 1; ++i) {
    sorted[i]->addDependency(sorted[i + 1]);
    LLVM_DEBUG(llvm::dbgs()
               << "  Added dependency: Stage " << sorted[i]->getStageId()
               << " -> Stage " << sorted[i + 1]->getStageId() << "\n");
  }

  return sorted;
}

//===----------------------------------------------------------------------===//
// Step 2: Analyze Original Stage Transfers
//===----------------------------------------------------------------------===//

/// Step 2 (single stage): Analyze data transfers in a stage that faces
/// intermediate stages.
/// Determine the intermediate resource from adjacent stages
static ResourceType getIntermediateResource(StageNode* prev_stage,
                                            StageNode* next_stage,
                                            PathExpansionPlan* plan) {
  bool prev_is_intermediate = prev_stage && isIntermediateStage(prev_stage);
  bool next_is_intermediate = next_stage && isIntermediateStage(next_stage);

  if (prev_is_intermediate) {
    assert(plan->stage_info.count(prev_stage) &&
           "Intermediate stage should have info");
    return plan->stage_info[prev_stage].stage_resource;
  } else if (next_is_intermediate) {
    assert(plan->stage_info.count(next_stage) &&
           "Intermediate stage should have info");
    return plan->stage_info[next_stage].stage_resource;
  } else {
    llvm_unreachable("Expected either prev or next stage to be intermediate");
  }
}

/// Check if transfer involves an intermediate stage (one side is memref)
static bool isIntermediateTransfer(mlir::Type src_type, mlir::Type dest_type) {
  bool src_is_memref = mlir::isa<mlir::MemRefType>(src_type);
  bool dest_is_memref = mlir::isa<mlir::MemRefType>(dest_type);
  // Transfer involves intermediate if exactly one side is memref
  return src_is_memref != dest_is_memref;
}

/// Create buffer specification and transfer info for intermediate stage
static TransferMaterializationInfo* createIntermediateTransferInfo(
    mlir::ktdf::DataTransferOp transfer, mlir::Type src_type,
    mlir::Type dest_type, ResourceType current_resource,
    ResourceType intermediate_resource, bool intermediate_is_source,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan) {
  bool src_is_memref = mlir::isa<mlir::MemRefType>(src_type);
  mlir::MemRefType memref_type = src_is_memref
                                     ? mlir::cast<mlir::MemRefType>(src_type)
                                     : mlir::cast<mlir::MemRefType>(dest_type);

  // The buffer shape is the transfer tile size, not the full tensor shape.
  // Use source sizes when the memref is the source, dest sizes otherwise.
  // We expect the sizes to be constants because the intermediate new buffers
  // that we are going to create are 1 stick size.
  llvm::SmallVector<mlir::OpFoldResult> tile_sizes =
      src_is_memref ? transfer.getMixedSourceSizes()
                    : transfer.getMixedDestSizes();
  llvm::SmallVector<int64_t> buffer_shape;
  for (mlir::OpFoldResult ofr : tile_sizes) {
    auto int_attr =
        mlir::dyn_cast<mlir::IntegerAttr>(ofr.dyn_cast<mlir::Attribute>());
    assert(int_attr && "Expected static tile sizes for intermediate buffer");
    buffer_shape.push_back(int_attr.getInt());
  }

  // Create memory buffer spec for the intermediate resource
  PrivateResourceSpec* buffer_spec = plan->resource_factory.createMemoryBuffer(
      intermediate_resource, buffer_shape, memref_type.getElementType());

  // Get architecture edge info
  scheduler::arch_view::RoutingGraph::NodeId intermediate_node =
      arch_graph.getNodeIdForResource(intermediate_resource);
  scheduler::arch_view::RoutingGraph::NodeId current_node =
      arch_graph.getNodeIdForResource(current_resource);

  scheduler::arch_view::RoutingGraph::NodeId src_node =
      intermediate_is_source ? intermediate_node : current_node;
  scheduler::arch_view::RoutingGraph::NodeId dest_node =
      intermediate_is_source ? current_node : intermediate_node;

  std::optional<scheduler::arch_view::RoutingGraph::EdgeInfo> edge =
      arch_graph.getEdgeInfo(src_node, dest_node);
  assert(edge && "Edge should exist for legal hop");

  // Create transfer materialization info using factory
  mlir::OpBuilder builder(transfer->getContext());
  return plan->transfer_factory.createFromTemplateWithBuffer(
      transfer, *edge, intermediate_resource, current_resource,
      intermediate_is_source, buffer_spec, builder);
}

static mlir::LogicalResult analyzeStageTransfers(
    StageNode* current_stage, StageNode* prev_stage, StageNode* next_stage,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    StageMaterializationInfo& stage_info, PathExpansionPlan* plan) {
  bool prev_is_intermediate = prev_stage && isIntermediateStage(prev_stage);
  bool next_is_intermediate = next_stage && isIntermediateStage(next_stage);

  // If this stage doesn't face any intermediate stages, skip it
  if (!prev_is_intermediate && !next_is_intermediate) {
    return mlir::success();
  }

  // Walk through data transfer operations
  mlir::ktdf::StageOp stage_op =
      mlir::cast<mlir::ktdf::StageOp>(current_stage->getOperation());

  stage_op.walk([&](mlir::ktdf::DataTransferOp transfer) {
    mlir::Type src_type = transfer.getSource().getType();
    mlir::Type dest_type = transfer.getDestination().getType();

    // Skip if transfer doesn't involve an intermediate stage
    if (!isIntermediateTransfer(src_type, dest_type)) {
      return mlir::WalkResult::advance();
    }

    // Determine resources and direction
    ResourceType current_resource = stage_info.stage_resource;
    bool intermediate_is_source = prev_is_intermediate;
    ResourceType intermediate_resource =
        getIntermediateResource(prev_stage, next_stage, plan);

    // Create transfer info with buffer spec
    TransferMaterializationInfo* transfer_info = createIntermediateTransferInfo(
        transfer, src_type, dest_type, current_resource, intermediate_resource,
        intermediate_is_source, arch_graph, plan);

    stage_info.transfers.push_back(transfer_info);

    // Update stage kind to kAdaptTransfer since we're modifying transfers
    if (stage_info.kind == StageMaterializationInfo::Kind::kPreserveOriginal) {
      stage_info.kind = StageMaterializationInfo::Kind::kAdaptTransfer;
    }

    LLVM_DEBUG({
      const PrivateResourceSpec* buffer_spec =
          transfer_info->source_private_resource;
      if (!intermediate_is_source) {
        buffer_spec = transfer_info->dest_private_resource;
      }
      llvm::dbgs() << "  Stage " << current_stage->getStageId()
                   << ": Created buffer spec for " <<
          [&]() {
            std::string s;
            llvm::raw_string_ostream os(s);
            (intermediate_resource).print(os);
            return os.str();
          }() << " shape=[";
      for (size_t j = 0; j < buffer_spec->shape.size(); ++j) {
        llvm::dbgs() << buffer_spec->shape[j];
        if (j + 1 < buffer_spec->shape.size()) llvm::dbgs() << "x";
      }
      mlir::MemRefType memref_type =
          mlir::isa<mlir::MemRefType>(src_type)
              ? mlir::cast<mlir::MemRefType>(src_type)
              : mlir::cast<mlir::MemRefType>(dest_type);
      llvm::dbgs() << "], element=" << memref_type.getElementType() << "\n";
    });

    return mlir::WalkResult::advance();
  });

  return mlir::success();
}

/// Step 2: Analyze data transfers in original stages that are
/// intermediate-stage facing, and for each data transfer create a
/// TransferMaterializationInfo in the StageMaterializationInfo::transfers list
/// with PrivateResourceSpec memory resource (according to the resource type of
/// the intermediate stage, eg L1). Assign this to source_private_resource or
/// dest_private_resource depending on whether the intermediate resource is at
/// the souce or sink of this stage.
static mlir::LogicalResult analyzeOriginalStageTransfers(
    llvm::ArrayRef<StageNode*> sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan) {
  LDBG(1) << "Analyzing " << sorted_stages.size() << " stages";

  for (size_t i = 0; i < sorted_stages.size(); ++i) {
    StageNode* current_stage = sorted_stages[i];

    // Skip intermediate (synthetic) stages
    if (isIntermediateStage(current_stage)) continue;

    // Get the stage's materialization info
    assert(plan->stage_info.count(current_stage) &&
           "Stage should have materialization info from Step 1");
    StageMaterializationInfo& stage_info = plan->stage_info[current_stage];

    // Get adjacent stages
    StageNode* prev_stage = (i > 0) ? sorted_stages[i - 1] : nullptr;
    StageNode* next_stage =
        (i + 1 < sorted_stages.size()) ? sorted_stages[i + 1] : nullptr;

    if (mlir::failed(analyzeStageTransfers(current_stage, prev_stage,
                                           next_stage, arch_graph, stage_info,
                                           plan))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Step 3: Analyze FIFO Operations
//===----------------------------------------------------------------------===//

/// Step 3: Look for fifo_read/fifo_write operations in stages that are facing
/// intermediate stages. For each such operation, query the architecture graph
/// to determine the correct FIFO type based on the resources involved, create a
/// corresponding PrivateResourceSpec for the FIFO, and link it to a new
/// TransferMaterializationInfo object in the
/// StageMaterializationInfo::transfers for that stage.
static mlir::LogicalResult analyzeFifoOperations(
    llvm::ArrayRef<StageNode*> sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan) {
  LDBG(1) << "Analyzing FIFO operations in " << sorted_stages.size()
          << " stages";

  // Track FIFO allocations by (src, dest) attribute pair to group slots
  // Map from attribute pair to the PrivateResourceSpec* created by factory
  using FifoKey = std::pair<mlir::Attribute, mlir::Attribute>;
  llvm::DenseMap<FifoKey, PrivateResourceSpec*> fifo_specs;
  // Track next available slot index for each FIFO type
  llvm::DenseMap<FifoKey, size_t> fifo_next_slot;

  for (size_t i = 0; i < sorted_stages.size(); ++i) {
    StageNode* current_stage = sorted_stages[i];

    // Skip intermediate (synthetic) stages
    if (isIntermediateStage(current_stage)) continue;

    // Get the stage's materialization info
    assert(plan->stage_info.count(current_stage) &&
           "Stage should have materialization info from Step 1");
    StageMaterializationInfo& stage_info = plan->stage_info[current_stage];

    // Get adjacent stages
    StageNode* prev_stage = (i > 0) ? sorted_stages[i - 1] : nullptr;
    StageNode* next_stage =
        (i + 1 < sorted_stages.size()) ? sorted_stages[i + 1] : nullptr;

    bool prev_is_intermediate = prev_stage && isIntermediateStage(prev_stage);
    bool next_is_intermediate = next_stage && isIntermediateStage(next_stage);

    // If this stage doesn't face any intermediate stages, skip it
    if (!prev_is_intermediate && !next_is_intermediate) {
      continue;
    }

    // Walk through FIFO read/write operations
    mlir::ktdf::StageOp stage_op =
        mlir::cast<mlir::ktdf::StageOp>(current_stage->getOperation());

    stage_op.walk([&](mlir::Operation* op) {
      // Check if this is a read_from_fifo or write_to_fifo operation
      auto read_op = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(op);
      auto write_op = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(op);

      if (!read_op && !write_op) {
        return mlir::WalkResult::advance();
      }

      // Determine if this is a read or write, and get the fifo slot
      bool is_read = (read_op != nullptr);
      mlir::Value fifo_slot =
          is_read ? read_op.getFifoSlot() : write_op.getFifoSlot();
      auto fifo_slot_type =
          mlir::cast<mlir::ktdf::FifoSlotType>(fifo_slot.getType());

      // Determine which adjacent stage is involved
      // Reads consume from previous stage, writes produce to next stage
      StageNode* adjacent_stage = is_read ? prev_stage : next_stage;
      bool adjacent_is_intermediate =
          is_read ? prev_is_intermediate : next_is_intermediate;

      if (!adjacent_is_intermediate) {
        return mlir::WalkResult::advance();
      }

      // Get resources (already as attributes)
      ResourceType current_resource = stage_info.stage_resource;
      ResourceType adjacent_resource =
          plan->stage_info[adjacent_stage].stage_resource;

      // Determine source and destination based on operation type
      ResourceType source_resource =
          is_read ? adjacent_resource : current_resource;
      ResourceType dest_resource =
          is_read ? current_resource : adjacent_resource;

      // Get element type and number of elements
      mlir::Type element_type = fifo_slot_type.getElementType();
      assert(!fifo_slot_type.isDynamicNumElements() &&
             "Dynamic FIFO sizes not supported in path expansion");
      int64_t num_elements = fifo_slot_type.getStaticNumElements();

      // Helper to convert resource attribute to FIFO endpoint attribute
      // Infers the transfer unit from the architecture graph.
      auto resourceToFifoEndpoint = [&](ResourceType resource,
                                        bool is_source) -> mlir::Attribute {
        // For FIFO endpoints:
        // - source (producer) side uses load units (memory -> compute)
        // - dest (consumer) side uses store units (compute -> memory)
        return ResourceInferrer::findTransferUnitForMemory(resource, is_source,
                                                           arch_graph);
      };

      mlir::Attribute fifo_src = resourceToFifoEndpoint(source_resource, true);
      mlir::Attribute fifo_dest = resourceToFifoEndpoint(dest_resource, false);
      FifoKey fifo_key = {fifo_src, fifo_dest};

      // Group FIFO allocations by (src, dest) attribute pair
      if (fifo_specs.find(fifo_key) == fifo_specs.end()) {
        // Create new FIFO spec via factory with first slot
        PrivateResourceSpec* spec = plan->resource_factory.createFifo(
            fifo_src, fifo_dest, {num_elements},  // First slot
            element_type);
        fifo_specs[fifo_key] = spec;
        fifo_next_slot[fifo_key] = 0;  // Initialize slot counter
      } else {
        // Add another slot to existing FIFO spec
        PrivateResourceSpec* spec = fifo_specs[fifo_key];
        spec->elements_per_slot.push_back(num_elements);
      }

      // Assign and increment slot index for this FIFO type
      size_t slot_idx = fifo_next_slot[fifo_key]++;

      // Validate slot index assignment
      validateFifoSlotIndex(fifo_slot, slot_idx, fifo_specs[fifo_key]);

      // Get edge info from architecture graph (needed for transfer creation)
      std::optional<scheduler::arch_view::RoutingGraph::EdgeInfo> edge =
          arch_graph.getEdgeInfoForResources(source_resource, dest_resource);
      assert(edge && "Edge should exist for legal hop");

      // Create transfer materialization info using factory
      TransferMaterializationInfo* transfer_info =
          plan->transfer_factory.createFromFifoOp(
              op, *edge, source_resource, dest_resource, fifo_specs[fifo_key],
              slot_idx, is_read);

      stage_info.transfers.push_back(transfer_info);

      // Update stage kind to kAdaptFifoKinds since we're adapting FIFO
      // operations
      if (stage_info.kind ==
          StageMaterializationInfo::Kind::kPreserveOriginal) {
        stage_info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;
      }

      LLVM_DEBUG({
        llvm::dbgs() << "  Stage " << current_stage->getStageId() << ": "
                     << (is_read ? "read_from_fifo" : "write_to_fifo")
                     << " - FIFO src=" << fifo_src << ", dest=" << fifo_dest
                     << ", slot " << slot_idx << ", elements=" << num_elements
                     << "\n";
      });

      return mlir::WalkResult::advance();
    });
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Step 4: Populate Intermediate Stage Transfers
//===----------------------------------------------------------------------===//

/// Step 4: Walk through intermediate stages and for each one, look at the
/// surrounding stages' transfers. For each data transfer in a neighboring
/// stage, create a corresponding TransferMaterializationInfo for the
/// intermediate stage, reusing the source/dest private resources and slot
/// indexes from the corresponding transfer in the neighboring stage.
/// Note: For a typical A -> B -> C chain where B is the intermediate stage,
/// this assumes that for every data transfer in A there is a corresponding
/// data/fifo transfer in C, and we are just adding a corresponding data
/// transfer in B.
static mlir::LogicalResult populateIntermediateStageTransfers(
    llvm::ArrayRef<StageNode*> sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan) {
  LDBG(1) << "Populating intermediate stage transfers for "
          << sorted_stages.size() << " stages";

  for (size_t i = 0; i < sorted_stages.size(); ++i) {
    StageNode* current_stage = sorted_stages[i];

    // Only process intermediate (synthetic) stages
    if (!isIntermediateStage(current_stage)) continue;

    // Get the stage's materialization info
    assert(plan->stage_info.count(current_stage) &&
           "Intermediate stage should have materialization info from Step 1");
    StageMaterializationInfo& stage_info = plan->stage_info[current_stage];

    // Get adjacent stages
    StageNode* prev_stage = (i > 0) ? sorted_stages[i - 1] : nullptr;
    StageNode* next_stage =
        (i + 1 < sorted_stages.size()) ? sorted_stages[i + 1] : nullptr;

    // Intermediate stages should have both prev and next stages
    assert(prev_stage && next_stage &&
           "Intermediate stage should have both neighbors");

    // Get the intermediate resource
    ResourceType intermediate_resource = stage_info.stage_resource;

    LLVM_DEBUG(llvm::dbgs() << "  Processing intermediate stage "
                            << current_stage->getStageId() << " (resource: " <<
               [&]() {
                 std::string s;
                 llvm::raw_string_ostream os(s);
                 (intermediate_resource).print(os);
                 return os.str();
               }() << ")\n");

    // Update stage kind to kSyntheticTransfer
    stage_info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;

    // Look at previous stage's transfers that have this intermediate stage as
    // destination
    if (plan->stage_info.count(prev_stage) == 0) continue;
    StageMaterializationInfo& prev_info = plan->stage_info[prev_stage];

    llvm::SmallPtrSet<const TransferMaterializationInfo*, 4>
        visited_next_transfer;
    for (const TransferMaterializationInfo* prev_transfer :
         prev_info.transfers) {
      // Check if this transfer's destination is the intermediate resource
      if (prev_transfer->dest_resource != intermediate_resource) continue;
      // Create corresponding transfer for intermediate stage
      // This transfer goes FROM the intermediate resource TO the next stage

      // Reuse the private resource from prev_transfer's destination as our
      // source
      assert(prev_transfer->dest_private_resource);
      const PrivateResourceSpec* source_resource_spec =
          prev_transfer->dest_private_resource;

      // Look for matching transfer in next stage to get destination resource
      // Find a transfer in next_stage that has intermediate_resource as source
      const PrivateResourceSpec* dest_resource_spec = nullptr;
      size_t dest_slot_idx = 0;
      llvm::SmallVector<mlir::Value> dest_indices_from_next;
      llvm::SmallVector<mlir::OpFoldResult> dest_sizes_from_next;
      mlir::AffineMap dest_map_from_next;
      assert(plan->stage_info.count(next_stage));
      StageMaterializationInfo& next_info = plan->stage_info[next_stage];
      for (const TransferMaterializationInfo* next_transfer :
           next_info.transfers) {
        if (visited_next_transfer.contains(next_transfer)) continue;
        if (next_transfer->source_resource == intermediate_resource) {
          // Found matching transfer - use its source and indices/sizes/map
          dest_resource_spec = next_transfer->source_private_resource;
          dest_slot_idx = next_transfer->source_slot_index;
          dest_indices_from_next = next_transfer->source_indices;
          dest_sizes_from_next = next_transfer->source_sizes;
          dest_map_from_next = next_transfer->source_map;
          visited_next_transfer.insert(next_transfer);
          break;
        }
      }
      assert(dest_resource_spec);

      // Infer actual source and destination resources from the private resource
      // specs
      std::optional<ResourceType> inferred_source_resource =
          ResourceInferrer::fromPrivateSpec(source_resource_spec,
                                            true,  // producer side
                                            arch_graph);
      std::optional<ResourceType> inferred_dest_resource =
          ResourceInferrer::fromPrivateSpec(dest_resource_spec,
                                            false,  // consumer side
                                            arch_graph);

      assert(inferred_source_resource && inferred_dest_resource &&
             "Should be able to infer resources from private specs");

      // Get edge info for the actual hop
      std::optional<scheduler::arch_view::RoutingGraph::EdgeInfo> edge =
          arch_graph.getEdgeInfoForResources(*inferred_source_resource,
                                             *inferred_dest_resource);

      assert(edge && "Edge should exist for legal hop");

      // Create transfer info for intermediate stage using factory
      // Pass affine maps from neighboring transfers
      // If a map is null but the side is a memref buffer, create an identity
      // map
      mlir::MLIRContext* ctx = prev_stage->getOperation()->getContext();
      mlir::AffineMap source_map_for_intermediate = prev_transfer->dest_map;
      if (!source_map_for_intermediate && source_resource_spec &&
          source_resource_spec->kind ==
              PrivateResourceSpec::Kind::kMemoryBuffer) {
        source_map_for_intermediate =
            createAffineMapForIntermediateBuffer(source_resource_spec, ctx);
      }

      mlir::AffineMap dest_map_for_intermediate = dest_map_from_next;
      if (!dest_map_for_intermediate && dest_resource_spec &&
          dest_resource_spec->kind ==
              PrivateResourceSpec::Kind::kMemoryBuffer) {
        dest_map_for_intermediate =
            createAffineMapForIntermediateBuffer(dest_resource_spec, ctx);
      }

      TransferMaterializationInfo* transfer_info =
          plan->transfer_factory.createSynthetic(
              *edge, *inferred_source_resource, *inferred_dest_resource,
              source_resource_spec, prev_transfer->dest_slot_index,
              prev_transfer->dest_indices, prev_transfer->dest_sizes,
              source_map_for_intermediate, dest_resource_spec, dest_slot_idx,
              dest_indices_from_next, dest_sizes_from_next,
              dest_map_for_intermediate, ctx);

      stage_info.transfers.push_back(transfer_info);

      LLVM_DEBUG({
        llvm::dbgs() << "    Created transfer: " <<
            [&]() {
              std::string s;
              llvm::raw_string_ostream os(s);
              (*inferred_source_resource).print(os);
              return os.str();
            }() << " -> "
                     << [&]() {
                          std::string s;
                          llvm::raw_string_ostream os(s);
                          (*inferred_dest_resource).print(os);
                          return os.str();
                        }();
        if (transfer_info->source_private_resource) {
          llvm::dbgs() << " (source: ";
          if (transfer_info->source_private_resource->kind ==
              PrivateResourceSpec::Kind::kFifo) {
            llvm::dbgs() << "FIFO slot " << transfer_info->source_slot_index;
          } else {
            llvm::dbgs() << "buffer";
          }
          llvm::dbgs() << ")";
        }
        if (transfer_info->dest_private_resource) {
          llvm::dbgs() << " (dest: ";
          if (transfer_info->dest_private_resource->kind ==
              PrivateResourceSpec::Kind::kFifo) {
            llvm::dbgs() << "FIFO slot " << transfer_info->dest_slot_index;
          } else {
            llvm::dbgs() << "buffer";
          }
          llvm::dbgs() << ")";
        }
        llvm::dbgs() << "\n";
      });
    }
  }

  return mlir::success();
}

/// Helper to get transfer unit from architecture graph edge
/// Returns the transfer unit for the edge between source and dest resources
static std::optional<ResourceType> getTransferUnitFromEdge(
    const scheduler::arch_view::RoutingGraph& arch_graph,
    const TransferMaterializationInfo& transfer) {
  std::optional<scheduler::arch_view::RoutingGraph::EdgeInfo> edge =
      arch_graph.getEdgeInfoForResources(transfer.source_resource,
                                         transfer.dest_resource);

  if (!edge) {
    LLVM_DEBUG(
        llvm::dbgs() << "  Warning: No edge found for " <<
        [&]() {
          std::string s;
          llvm::raw_string_ostream os(s);
          (transfer.source_resource).print(os);
          return os.str();
        }() << " -> " <<
        [&]() {
          std::string s;
          llvm::raw_string_ostream os(s);
          (transfer.dest_resource).print(os);
          return os.str();
        }() << "\n");
    return std::nullopt;
  }

  // Return the transfer_unit attribute directly as ResourceType
  if (edge->transfer_unit) return edge->transfer_unit;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Step 5: Assign Applicable Units
//===----------------------------------------------------------------------===//

/// Step 5 (single stage): Assign applicable unit to a stage based on its
/// transfer source/dest resources. For transfer stages, query the architecture
/// graph to get the transfer unit (component) that handles the transfer between
/// the source and destination resources. For example: L1 -> SFU uses L1LU. For
/// non-transfer stages, use the anchor resource if available, otherwise use the
/// first transfer.

/// get applicable unit for transfer stages
static std::optional<ResourceType> getTransferStageUnit(
    StageNode* stage, const scheduler::arch_view::RoutingGraph& arch_graph,
    const StageMaterializationInfo& stage_info) {
  if (stage_info.transfers.empty()) {
    return std::nullopt;
  }

  const TransferMaterializationInfo* transfer = stage_info.transfers[0];
  std::optional<ResourceType> transfer_unit =
      getTransferUnitFromEdge(arch_graph, *transfer);

  if (transfer_unit) {
    LLVM_DEBUG(
        llvm::dbgs() << "  Stage " << stage->getStageId()
                     << ": Assigned applicable unit " <<
        [&]() {
          std::string s;
          llvm::raw_string_ostream os(s);
          (*transfer_unit).print(os);
          return os.str();
        }() << " for "
                     <<
        [&]() {
          std::string s;
          llvm::raw_string_ostream os(s);
          (transfer->source_resource).print(os);
          return os.str();
        }() << " -> " <<
        [&]() {
          std::string s;
          llvm::raw_string_ostream os(s);
          (transfer->dest_resource).print(os);
          return os.str();
        }() << "\n");
  }
  return transfer_unit;
}

/// get applicable unit for compute stages based on resource
static std::optional<ResourceType> getComputeStageUnit(
    StageNode* stage, const StageMaterializationInfo& stage_info) {
  if (!stage_info.stage_resource) {
    return std::nullopt;
  }

  ResourceType resource = stage_info.stage_resource;

  // Return the resource directly as the applicable unit
  LLVM_DEBUG(llvm::dbgs() << "  Stage " << stage->getStageId()
                          << ": Assigned applicable unit " <<
             [&]() {
               std::string s;
               llvm::raw_string_ostream os(s);
               (resource).print(os);
               return os.str();
             }() << " from stage resource\n");
  return resource;
}

static mlir::LogicalResult assignApplicableUnit(
    StageNode* stage, const scheduler::arch_view::RoutingGraph& arch_graph,
    StageMaterializationInfo& stage_info) {
  // Skip if already has an applicable unit assigned
  if (stage_info.applicable_unit.has_value()) {
    return mlir::success();
  }

  // Determine the applicable unit based on stage kind
  if (stage_info.kind == StageMaterializationInfo::Kind::kAdaptTransfer ||
      stage_info.kind == StageMaterializationInfo::Kind::kSyntheticTransfer) {
    // For transfer stages, use the transfer unit from the architecture graph
    stage_info.applicable_unit =
        getTransferStageUnit(stage, arch_graph, stage_info);
    return mlir::success();
  }

  // For non-transfer stages (kPreserveOriginal, kAdaptFifoKinds)
  // Try to use the stage resource first
  stage_info.applicable_unit = getComputeStageUnit(stage, stage_info);
  if (stage_info.applicable_unit) {
    return mlir::success();
  }

  // Fallback: use the first transfer's edge to determine the unit
  stage_info.applicable_unit =
      getTransferStageUnit(stage, arch_graph, stage_info);
  if (stage_info.applicable_unit) {
    LDBG(1) << "  Stage " << stage->getStageId()
            << ": Assigned applicable unit from first transfer (fallback)";
  }

  return mlir::success();
}

/// Step 5: Assign applicable units to all stages.
static mlir::LogicalResult assignApplicableUnits(
    llvm::ArrayRef<StageNode*> sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan) {
  LDBG(1) << "Assigning applicable units for " << sorted_stages.size()
          << " stages";

  for (StageNode* stage : sorted_stages) {
    // Skip stages that don't have materialization info
    if (plan->stage_info.count(stage) == 0) continue;
    StageMaterializationInfo& stage_info = plan->stage_info[stage];
    if (mlir::failed(assignApplicableUnit(stage, arch_graph, stage_info))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Main Planning Function
//===----------------------------------------------------------------------===//

/// Prepare and validate pipeline for path expansion
static llvm::FailureOr<llvm::SmallVector<StageNode*>> preparePipelineStages(
    PipelineTree& tree, PipelineNode* pipeline) {
  // Perform topological sort
  llvm::FailureOr<llvm::SmallVector<StageNode*>> sorted_stages_or =
      tree.topologicalSortStages(pipeline);
  if (mlir::failed(sorted_stages_or)) {
    LDBG(1) << "Failed to perform topological sort (cycle detected)";
    return mlir::failure();
  }

  llvm::SmallVector<StageNode*> sorted_stages = *sorted_stages_or;

  // Validate linear chain topology
  if (mlir::failed(validateLinearChain(sorted_stages))) {
    LDBG(1) << "Stage topology is not a linear chain";
    return mlir::failure();
  }

  return sorted_stages;
}

/// Build full shortest path from original path segments
static std::optional<llvm::SmallVector<ResourceType>> buildFullShortestPath(
    const llvm::SmallVector<ResourceType>& original_path,
    const scheduler::arch_view::RoutingGraph& arch_graph) {
  llvm::SmallVector<ResourceType> full_shortest_path;

  for (size_t i = 0; i < original_path.size() - 1; ++i) {
    ResourceType seg_source = original_path[i];
    ResourceType seg_dest = original_path[i + 1];

    scheduler::arch_view::RoutingGraph::NodeId source_node =
        arch_graph.getNodeIdForResource(seg_source);
    scheduler::arch_view::RoutingGraph::NodeId dest_node =
        arch_graph.getNodeIdForResource(seg_dest);

    std::optional<llvm::SmallVector<scheduler::arch_view::RoutingGraph::NodeId>>
        segment_path_or = arch_graph.findShortestPath(source_node, dest_node);
    if (!segment_path_or) {
      LLVM_DEBUG(
          llvm::dbgs() << "No legal path found from " <<
          [&]() {
            std::string s;
            llvm::raw_string_ostream os(s);
            (seg_source).print(os);
            return os.str();
          }() << " to " <<
          [&]() {
            std::string s;
            llvm::raw_string_ostream os(s);
            (seg_dest).print(os);
            return os.str();
          }() << "\n");
      return std::nullopt;
    }

    // Convert node IDs to resources and add to full path
    for (size_t j = 0; j < segment_path_or->size(); ++j) {
      scheduler::arch_view::RoutingGraph::NodeId node_id =
          (*segment_path_or)[j];
      std::optional<scheduler::arch_view::RoutingGraph::ResourceNode> node_or =
          arch_graph.getNode(node_id);
      if (!node_or) return std::nullopt;

      // Avoid duplicating connection points between segments
      if (j == 0 && !full_shortest_path.empty() &&
          full_shortest_path.back() == node_or->resource) {
        continue;
      }
      full_shortest_path.push_back(node_or->resource);
    }
  }

  return full_shortest_path;
}

/// Execute the 5-step path expansion algorithm
static mlir::LogicalResult executeExpansionSteps(
    llvm::SmallVector<StageNode*>& expanded_stages, PipelineTree& tree,
    PipelineNode* pipeline, const llvm::SmallVector<StageNode*>& sorted_stages,
    const llvm::SmallVector<StageSummary>& summaries,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan, int& next_stage_id) {
  // STEP 1: Legalize connections and insert intermediate stages
  LLVM_DEBUG(llvm::dbgs() << "\n=== Step 1: Connection Legalization ===\n");
  auto expanded_stages_or =
      legalizeStageConnections(tree, pipeline, sorted_stages, summaries,
                               arch_graph, plan, next_stage_id);
  if (mlir::failed(expanded_stages_or)) {
    return mlir::failure();
  }
  expanded_stages = *expanded_stages_or;
  LLVM_DEBUG(
      debugPrintStageList(llvm::dbgs(), expanded_stages, plan, "After Step 1"));

  // STEP 2: Analyze original stage transfers
  LLVM_DEBUG(
      llvm::dbgs() << "\n=== Step 2: Original Stage Transfer Analysis ===\n");
  if (mlir::failed(
          analyzeOriginalStageTransfers(expanded_stages, arch_graph, plan))) {
    return mlir::failure();
  }
  LLVM_DEBUG(
      debugPrintStageList(llvm::dbgs(), expanded_stages, plan, "After Step 2"));

  // STEP 3: Analyze FIFO operations
  LLVM_DEBUG(llvm::dbgs() << "\n=== Step 3: FIFO Operation Analysis ===\n");
  if (mlir::failed(analyzeFifoOperations(expanded_stages, arch_graph, plan))) {
    return mlir::failure();
  }
  LLVM_DEBUG(
      debugPrintStageList(llvm::dbgs(), expanded_stages, plan, "After Step 3"));

  // STEP 4: Populate intermediate stage transfers
  LLVM_DEBUG(llvm::dbgs()
             << "\n=== Step 4: Intermediate Stage Transfer Population ===\n");
  if (mlir::failed(populateIntermediateStageTransfers(expanded_stages,
                                                      arch_graph, plan))) {
    return mlir::failure();
  }
  LLVM_DEBUG(
      debugPrintStageList(llvm::dbgs(), expanded_stages, plan, "After Step 4"));

  // STEP 5: Assign applicable units
  LLVM_DEBUG(llvm::dbgs() << "\n=== Step 5: Assign Applicable Units ===\n");
  if (mlir::failed(assignApplicableUnits(expanded_stages, arch_graph, plan))) {
    return mlir::failure();
  }
  LLVM_DEBUG(
      debugPrintStageList(llvm::dbgs(), expanded_stages, plan, "After Step 5"));

  return mlir::success();
}

std::unique_ptr<PathExpansionPlan> planPathExpansion(
    PipelineTree& tree, PipelineNode* pipeline,
    const scheduler::arch_view::RoutingGraph& arch_graph) {
  auto plan = std::make_unique<PathExpansionPlan>();
  plan->changed = false;

  // Prep Step 1: Prepare and validate pipeline stages
  llvm::FailureOr<llvm::SmallVector<StageNode*>> sorted_stages_or =
      preparePipelineStages(tree, pipeline);
  if (mlir::failed(sorted_stages_or)) {
    return nullptr;
  }
  llvm::SmallVector<StageNode*> sorted_stages = *sorted_stages_or;

  // Prep Step 2: Analyze stages in topological order
  llvm::FailureOr<llvm::SmallVector<StageSummary>> summaries_or =
      analyzeStages(sorted_stages, arch_graph);
  if (mlir::failed(summaries_or)) {
    return nullptr;
  }
  llvm::SmallVector<StageSummary> summaries = *summaries_or;

  // Prep Step 3: Infer original path
  llvm::FailureOr<llvm::SmallVector<ResourceType>> original_path_or =
      inferOriginalPath(summaries);
  if (mlir::failed(original_path_or)) {
    return nullptr;
  }
  llvm::SmallVector<ResourceType> original_path = *original_path_or;

  if (original_path.size() < 2) {
    return nullptr;  // Need at least source and destination
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Original PipelineTree before expansion:\n";
    pipeline->print(llvm::dbgs());
    llvm::dbgs() << "\nOriginal topological order: ";
    for (StageNode* stage : sorted_stages) {
      llvm::dbgs() << "Stage " << stage->getStageId() << " ";
    }
    llvm::dbgs() << "\n\n";
  });

  // Prep Step 4: Build full shortest path
  std::optional<llvm::SmallVector<ResourceType>> full_shortest_path_or =
      buildFullShortestPath(original_path, arch_graph);
  if (!full_shortest_path_or) {
    return nullptr;
  }
  llvm::SmallVector<ResourceType> full_shortest_path = *full_shortest_path_or;

  LLVM_DEBUG({
    llvm::dbgs() << "Original path: ";
    for (auto res : original_path) {
      llvm::dbgs() << [&]() {
        std::string s;
        llvm::raw_string_ostream os(s);
        (res).print(os);
        return os.str();
      }() << " -> ";
    }
    llvm::dbgs() << "\nFull shortest path: ";
    for (auto res : full_shortest_path) {
      llvm::dbgs() << [&]() {
        std::string s;
        llvm::raw_string_ostream os(s);
        (res).print(os);
        return os.str();
      }() << " -> ";
    }
    llvm::dbgs() << "\n";
  });

  // Check if expansion is needed
  if (original_path == full_shortest_path) {
    plan->changed = false;
    LDBG(1) << "Pipeline already legal";
    return plan;
  }

  plan->changed = true;
  plan->modified_tree = &tree;

  // Execute the 5-step expansion algorithm
  int next_stage_id = sorted_stages.size();
  llvm::SmallVector<StageNode*> expanded_stages;

  if (mlir::failed(executeExpansionSteps(expanded_stages, tree, pipeline,
                                         sorted_stages, summaries, arch_graph,
                                         plan.get(), next_stage_id))) {
    return nullptr;
  }

  // Final summary
  LLVM_DEBUG({
    llvm::dbgs() << "\n=== Planning Complete ===\n";
    llvm::dbgs() << "Total stages: " << expanded_stages.size() << "\n";
    llvm::dbgs() << "Total private resources: "
                 << plan->resource_factory.getSpecs().size() << "\n";
    debugPrintResourceSpecs(llvm::dbgs(), plan->resource_factory);
  });

  return plan;
}

}  // namespace scheduler

// Made with Bob
