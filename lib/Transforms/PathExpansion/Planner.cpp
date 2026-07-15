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
#include "llvm/ADT/SmallPtrSet.h"
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
static mlir::AffineMap createAffineMapForIntermediateBuffer(
    const PrivateResourceSpec* buffer, mlir::MLIRContext* context) {
  return mlir::AffineMap::getMultiDimIdentityMap(buffer->shape.size(), context);
}

/// Helper to validate FIFO slot index assignment
void validateFifoSlotIndex(mlir::Value orig_fifo_value,
                           size_t assigned_slot_idx,
                           const PrivateResourceSpec* fifo_spec) {
  if (auto orig_private_result =
          mlir::dyn_cast<mlir::OpResult>(orig_fifo_value)) {
    unsigned orig_result_idx = orig_private_result.getResultNumber();
    auto priv_op = orig_fifo_value.getDefiningOp<mlir::ktdf::PrivateOp>();
    assert(priv_op);
    mlir::ktdf::PrivateYieldOp yield = priv_op.getYieldOp();
    assert(orig_result_idx < yield.getOperands().size());
    mlir::Value fifo_slot_value = yield.getOperands()[orig_result_idx];
    mlir::OpResult fifo_alloc_result =
        mlir::dyn_cast<mlir::OpResult>(fifo_slot_value);
    assert(fifo_alloc_result);

    assert(assigned_slot_idx == fifo_alloc_result.getResultNumber() &&
           "expected result index to match our calculated fifo slot index");
  }
  assert(assigned_slot_idx < fifo_spec->elements_per_slot.size() &&
         "Slot index should be within the allocated slots for this FIFO type");
}

/// Helper to check if a stage is an intermediate (synthetic) stage
inline bool isIntermediateStage(const StageNode* stage) {
  return stage->getOperation() == nullptr;
}

//===----------------------------------------------------------------------===//
// Debug Output Helpers
//===----------------------------------------------------------------------===//

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

    if (plan && plan->stage_info.count(stage)) {
      const StageMaterializationInfo& info = plan->stage_info.at(stage);
      info.print(os);
    }
    os << "\n";
  }
}

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

//===----------------------------------------------------------------------===//
// Stage DAG Validation Functions
//===----------------------------------------------------------------------===//
mlir::LogicalResult validateLinearChain(
    llvm::ArrayRef<StageNode*> sorted_stages) {
  if (sorted_stages.empty()) {
    return mlir::success();
  }

  llvm::DenseMap<StageNode*, int> dependent_count;
  for (StageNode* stage : sorted_stages) {
    dependent_count[stage] = 0;
  }

  for (StageNode* stage : sorted_stages) {
    const llvm::SmallVector<StageNode*>& deps = stage->getDependencies();

    if (deps.size() > 1) {
      LDBG(1) << "Stage " << stage->getStageId()
              << " has multiple outgoing dependencies (branching)";
      return mlir::failure();
    }

    for (StageNode* dep : deps) {
      dependent_count[dep]++;
    }
  }

  int source_count = 0;
  int sink_count = 0;

  for (StageNode* stage : sorted_stages) {
    int incoming = dependent_count[stage];
    const llvm::SmallVector<StageNode*>& outgoing = stage->getDependencies();

    if (incoming == 0) source_count++;
    if (outgoing.empty()) sink_count++;

    if (incoming > 1) {
      LDBG(1) << "Stage " << stage->getStageId()
              << " has multiple incoming dependencies (merging)";
      return mlir::failure();
    }
  }

  if (source_count != 1 || sink_count != 1) {
    LDBG(1) << "Expected 1 source and 1 sink, got " << source_count
            << " sources and " << sink_count << " sinks";
    return mlir::failure();
  }

  return mlir::success();
}
/// Helper to determine a stage's resource (for original stages only).
/// For compute stages (anchor_resource present): uses anchor_resource.
/// For transfer stages: uses the MemRefType memory space from the first
/// data_transfer op.
/// Return the memory-space attribute from val's type if it is a MemRefType
/// with a memory space, otherwise return nullptr.
static ResourceType memrefMemorySpace(mlir::Value val) {
  if (auto mt = mlir::dyn_cast<mlir::MemRefType>(val.getType()))
    return mt.getMemorySpace();
  return nullptr;
}

/// Return the memory-space of the first data_transfer's source memref in
/// stage_op, or nullptr if none found.
static ResourceType firstTransferSourceMemSpace(mlir::ktdf::StageOp stage_op) {
  ResourceType result;
  stage_op.walk([&](mlir::ktdf::DataTransferOp transfer_op) {
    if (result) return mlir::WalkResult::interrupt();
    result = memrefMemorySpace(transfer_op.getSource());
    return result ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
  });
  return result;
}

/// Return the memory-space of the first data_transfer's destination memref in
/// stage_op, or nullptr if none found.
static ResourceType firstTransferDestMemSpace(mlir::ktdf::StageOp stage_op) {
  ResourceType result;
  stage_op.walk([&](mlir::ktdf::DataTransferOp transfer_op) {
    if (result) return mlir::WalkResult::interrupt();
    result = memrefMemorySpace(transfer_op.getDestination());
    return result ? mlir::WalkResult::interrupt() : mlir::WalkResult::advance();
  });
  return result;
}

//===----------------------------------------------------------------------===//
// Main Planning Functions
//===----------------------------------------------------------------------===//

/// Prepare and validate pipeline for path expansion
static llvm::FailureOr<llvm::SmallVector<StageNode*>> preparePipelineStages(
    PipelineTree& tree, PipelineNode* pipeline) {
  llvm::FailureOr<llvm::SmallVector<StageNode*>> sorted_stages_or =
      tree.topologicalSortStages(pipeline);
  if (mlir::failed(sorted_stages_or)) {
    LDBG(1) << "Failed to perform topological sort (cycle detected)";
    return mlir::failure();
  }

  llvm::SmallVector<StageNode*> sorted_stages = *sorted_stages_or;

  if (mlir::failed(validateLinearChain(sorted_stages))) {
    LDBG(1) << "Stage topology is not a linear chain";
    return mlir::failure();
  }

  return sorted_stages;
}

/// Build full shortest path by walking consecutive pairs of original-stage
/// resources and concatenating the BFS result for each segment.
/// This correctly handles round-trip pipelines (e.g. DDR→SFU→DDR) where
/// start == end, which a single findShortestPath(DDR, DDR) call cannot resolve.
static std::optional<scheduler::arch_view::RoutingGraph::Path>
buildFullShortestPath(llvm::ArrayRef<ResourceType> original_resource_path,
                      const scheduler::arch_view::RoutingGraph& arch_graph) {
  scheduler::arch_view::RoutingGraph::Path full_path;

  for (size_t i = 0; i + 1 < original_resource_path.size(); ++i) {
    scheduler::arch_view::RoutingGraph::NodeId src =
        arch_graph.getNodeIdForResource(original_resource_path[i]);
    scheduler::arch_view::RoutingGraph::NodeId dst =
        arch_graph.getNodeIdForResource(original_resource_path[i + 1]);

    std::optional<scheduler::arch_view::RoutingGraph::Path> seg =
        arch_graph.findShortestPath(src, dst);
    if (!seg) return std::nullopt;

    for (size_t j = 0; j < seg->size(); ++j) {
      // Skip the junction node already present as the tail of the previous seg.
      if (j == 0 && !full_path.empty() && full_path.back() == (*seg)[j])
        continue;
      full_path.push_back((*seg)[j]);
    }
  }

  return full_path;
}

/// Assign stage_resource to every original stage using a context-aware
/// two-pass algorithm that accounts for neighboring stages.
///
/// The stage_resource is the memory endpoint adjacent to the stage's LS unit
/// in the routing graph.  It cannot always be determined from a single stage
/// in isolation when a transfer has two memref operands; the neighbor context
/// disambiguates which side is the external endpoint.
///
/// Rules (applied in order, first match wins):
///  1. Anchor stage (applicable_unit present in input IR): stage_resource ==
///     that applicable-unit resource.
///  2. First stage in chain (no left neighbor): use source memref memory-space.
///  3. Last stage in chain (no right neighbor): use dest memref memory-space.
///  4. Intermediate stage: examine already-resolved neighbor resources —
///       a. If left neighbor's stage_resource matches this stage's transfer
///          source memref → use this stage's transfer dest memref memory-space.
///       b. Else if right neighbor's stage_resource matches this stage's
///          transfer dest memref → use this stage's transfer source memref
///          memory-space.
///       c. Otherwise assert (unresolvable without further passes).
///
/// Pass 1 resolves rules 1–3.  Pass 2 resolves rule 4 for any remaining
/// stages, iterating until stable (handles chains of intermediate stages).
static void assignOriginalStageResources(
    llvm::ArrayRef<StageNode*> sorted_stages, PathExpansionPlan* plan) {
  size_t n = sorted_stages.size();

  // Initialise all entries with kPreserveOriginal and null stage_resource.
  for (size_t i = 0; i < n; ++i) {
    StageMaterializationInfo info;
    info.kind = StageMaterializationInfo::Kind::kPreserveOriginal;
    plan->stage_info[sorted_stages[i]] = info;
  }

  // Pass 1: resolve rules 1, 2, 3.
  for (size_t i = 0; i < n; ++i) {
    StageNode* stage = sorted_stages[i];
    StageMaterializationInfo& info = plan->stage_info[stage];

    auto stage_op =
        mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(stage->getOperation());
    if (!stage_op) continue;

    // Rule 1: anchor stage — applicable_unit already known from input IR.
    if (auto units = stage_op.getApplicableUnits()) {
      assert(units->size() == 1 &&
             "path expansion currently does not handle nested pipelines with "
             "multi-unit stages");
      info.stage_resource = units->getValue()[0];
      continue;
    }

    // Rule 2: first stage — use source memref.
    if (i == 0) {
      ResourceType r = firstTransferSourceMemSpace(stage_op);
      assert(r && "First stage must have a memref source on its data_transfer");
      info.stage_resource = r;
      continue;
    }

    // Rule 3: last stage — use dest memref.
    if (i == n - 1) {
      ResourceType r = firstTransferDestMemSpace(stage_op);
      assert(r && "Last stage must have a memref dest on its data_transfer");
      info.stage_resource = r;
      continue;
    }

    // Rule 3.5: exactly one side of the transfer is a memref (the other side
    // is a FIFO or similar non-memref).  The memref side is unambiguously the
    // stage resource — no neighbor context needed.
    {
      ResourceType src_ms = firstTransferSourceMemSpace(stage_op);
      ResourceType dst_ms = firstTransferDestMemSpace(stage_op);
      if (src_ms && !dst_ms) {
        info.stage_resource = src_ms;
        continue;
      }
      if (dst_ms && !src_ms) {
        info.stage_resource = dst_ms;
        continue;
      }
    }
    // Rule 4 will be handled in pass 2.
  }

  // Pass 2: resolve rule 4 iteratively until stable.
  // Each iteration resolves at least one previously-unresolved intermediate
  // stage (using a neighbor resolved in the previous iteration).
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 1; i + 1 < n; ++i) {
      StageNode* stage = sorted_stages[i];
      StageMaterializationInfo& info = plan->stage_info[stage];
      if (info.stage_resource) continue;  // already resolved

      auto stage_op =
          mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(stage->getOperation());
      if (!stage_op) continue;

      ResourceType src_ms = firstTransferSourceMemSpace(stage_op);
      ResourceType dst_ms = firstTransferDestMemSpace(stage_op);

      ResourceType left_resource =
          plan->stage_info[sorted_stages[i - 1]].stage_resource;
      ResourceType right_resource =
          plan->stage_info[sorted_stages[i + 1]].stage_resource;

      // Rule 4a: left neighbor's resource matches our transfer source
      //          → our endpoint is the destination side.
      if (left_resource && src_ms && left_resource == src_ms) {
        assert(dst_ms &&
               "Intermediate stage transfer must have a dest memref "
               "when resolved via left neighbor");
        info.stage_resource = dst_ms;
        changed = true;
        continue;
      }

      // Rule 4b: right neighbor's resource matches our transfer destination
      //          → our endpoint is the source side.
      if (right_resource && dst_ms && right_resource == dst_ms) {
        assert(src_ms &&
               "Intermediate stage transfer must have a source memref "
               "when resolved via right neighbor");
        info.stage_resource = src_ms;
        changed = true;
        continue;
      }
    }
  }

  // Verify all stages were resolved.
  for (size_t i = 0; i < n; ++i) {
    assert(plan->stage_info[sorted_stages[i]].stage_resource &&
           "Unable to determine stage_resource for original stage");
  }
}

/// Find the first unvisited original stage whose stage_resource == resource,
/// searching only within the given subrange of sorted_stages.
/// \p subrange is a slice of sorted_stages delimited by [begin_idx, end_idx).
static StageNode* findUnvisitedStageByResource(
    llvm::ArrayRef<StageNode*> sorted_stages,
    const llvm::SmallPtrSet<StageNode*, 8>& visited, ResourceType resource,
    const PathExpansionPlan* plan, size_t begin_idx, size_t end_idx) {
  for (size_t idx = begin_idx; idx < end_idx; ++idx) {
    StageNode* s = sorted_stages[idx];
    if (visited.count(s)) continue;
    auto it = plan->stage_info.find(s);
    if (it == plan->stage_info.end()) continue;
    if (it->second.stage_resource == resource) return s;
  }
  return nullptr;
}

/// Step 1: Build the expanded stage list by walking the routing-graph path.
/// Sets stage_resource and applicable_unit for every stage (original and
/// synthetic). Inserts synthetic stages into the pipeline and rebuilds the
/// linear dependency chain.
static llvm::FailureOr<llvm::SmallVector<StageNode*>> buildExpandedStageList(
    PipelineTree& tree, PipelineNode* pipeline,
    const scheduler::arch_view::RoutingGraph::Path& path,
    llvm::ArrayRef<StageNode*> sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan, int& next_stage_id) {
  using RK = scheduler::arch_view::RoutingGraph::ResourceNode::ResourceKind;

  // Ordered list of stages in the expanded pipeline (dependency order)
  llvm::SmallVector<StageNode*> ordered_stages;
  // Tracks which original stages have already been assigned
  llvm::SmallPtrSet<StageNode*, 8> visited;
  // Tracks the last stage added, so synthetic stages are inserted after it
  // in the pipeline's sibling list (which controls materializer output order).
  StageNode* last_inserted = nullptr;

  // Pre-walk: build an ordered list of sorted-stage indices for every Compute
  // node encountered on the path, in path order.  This drives the
  // direction-aware lookup: between two consecutive compute nodes at sorted
  // indices A and B, Memory/LS lookups are restricted to sorted_stages[A+1, B).
  // Before the first compute node lookups are restricted to [0, first_anchor).
  // After the last compute node lookups are restricted to (last_anchor, end].
  llvm::SmallVector<size_t> anchor_sorted_indices;
  for (size_t pi = 0; pi < path.size(); ++pi) {
    auto pnode_opt = arch_graph.getNode(path[pi]);
    if (!pnode_opt || pnode_opt->kind != RK::Compute) continue;
    // Find the sorted-stage index for this compute resource.
    for (size_t k = 0; k < sorted_stages.size(); ++k) {
      auto it = plan->stage_info.find(sorted_stages[k]);
      if (it == plan->stage_info.end()) continue;
      if (it->second.stage_resource == pnode_opt->resource) {
        anchor_sorted_indices.push_back(k);
        break;
      }
    }
  }

  // Direction-aware lookup bounds [search_begin, search_end).
  // search_end is initialised to the first anchor's sorted index (or the end of
  // sorted_stages if there are no compute nodes) so that load-side lookups
  // never reach store-side stages.  Both bounds advance each time a Compute
  // node is processed (Case C).
  size_t next_anchor_cursor = 0;  // index into anchor_sorted_indices
  size_t search_begin = 0;
  size_t search_end = anchor_sorted_indices.empty() ? sorted_stages.size()
                                                    : anchor_sorted_indices[0];

  // Helper: append a stage to ordered_stages and update last_inserted.
  auto insertStage = [&](StageNode* stage) {
    ordered_stages.push_back(stage);
    last_inserted = stage;
  };

  // Helper: direction-aware findUnvisited wrapper.
  auto findStage = [&](ResourceType resource) -> StageNode* {
    return findUnvisitedStageByResource(sorted_stages, visited, resource, plan,
                                        search_begin, search_end);
  };

  // Walk the path
  size_t n = path.size();
  size_t i = 0;
  while (i < n) {
    auto node_opt = arch_graph.getNode(path[i]);
    assert(node_opt && "node in path must exist in graph");
    const auto& node = *node_opt;

    if (node.kind == RK::Memory) {
      // --- Case A: Memory node ---
      StageNode* S = findStage(node.resource);
      if (S) {
        // An original stage claims this memory node
        visited.insert(S);

        // Peek right: if an LS-unit node follows, set applicable_unit and skip
        if (i + 1 < n) {
          auto right_opt = arch_graph.getNode(path[i + 1]);
          if (right_opt && right_opt->kind == RK::LoadStoreUnit) {
            plan->stage_info[S].applicable_unit = right_opt->resource;
            i += 2;  // skip memory node + LS node
            insertStage(S);
            continue;
          }
        }
        // No LS node to the right; applicable_unit stays unset for now
        i += 1;
        insertStage(S);
        continue;
      } else {
        // No original stage for this memory node — peek right for LS unit
        if (i + 1 < n) {
          auto right_opt = arch_graph.getNode(path[i + 1]);
          if (right_opt && right_opt->kind == RK::LoadStoreUnit) {
            // Create synthetic stage covering this LS unit
            StageNode* synth = tree.createStageNode(nullptr, next_stage_id++);
            StageMaterializationInfo info;
            info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;
            info.stage_resource = node.resource;
            info.applicable_unit = right_opt->resource;
            plan->stage_info[synth] = info;
            pipeline->insertChildNode(synth, last_inserted);
            i += 2;  // skip memory node + LS node
            insertStage(synth);
            continue;
          }
        }
        // Memory node with no LS right neighbor and no original stage — skip
        i += 1;
        continue;
      }
    } else if (node.kind == RK::LoadStoreUnit) {
      // --- Case B: LoadStoreUnit node ---
      // Reached only when the preceding memory node did NOT consume this node.
      // Pattern: LS-unit → Memory (store side)
      StageNode* stage_to_use = nullptr;
      bool advanced_extra = false;

      if (i + 1 < n) {
        auto right_opt = arch_graph.getNode(path[i + 1]);
        if (right_opt && right_opt->kind == RK::Memory) {
          // Look for original stage at the right memory node
          StageNode* S = findStage(right_opt->resource);
          if (S) {
            visited.insert(S);
            plan->stage_info[S].applicable_unit = node.resource;
            stage_to_use = S;
          } else {
            // Synthetic store-side stage
            StageNode* synth = tree.createStageNode(nullptr, next_stage_id++);
            StageMaterializationInfo info;
            info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;
            info.stage_resource = right_opt->resource;
            info.applicable_unit = node.resource;
            plan->stage_info[synth] = info;
            pipeline->insertChildNode(synth, last_inserted);
            stage_to_use = synth;
          }
          advanced_extra = true;
        }
      }

      if (!stage_to_use) {
        // No memory neighbor — unusual topology; create synthetic anyway
        StageNode* synth = tree.createStageNode(nullptr, next_stage_id++);
        StageMaterializationInfo info;
        info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;
        info.applicable_unit = node.resource;
        plan->stage_info[synth] = info;
        pipeline->insertChildNode(synth, last_inserted);
        stage_to_use = synth;
      }

      insertStage(stage_to_use);
      i += advanced_extra ? 2 : 1;
      continue;
    } else {
      // --- Case C: Compute node ---
      // The current anchor is anchor_sorted_indices[next_anchor_cursor].
      // Search exactly that one slot so we pick up the right compute stage.
      assert(next_anchor_cursor < anchor_sorted_indices.size() &&
             "Compute node in path must have a corresponding original stage");
      size_t anchor_idx = anchor_sorted_indices[next_anchor_cursor];
      ++next_anchor_cursor;
      StageNode* S = sorted_stages[anchor_idx];
      assert(!visited.count(S) && "Compute stage should not have been visited");
      visited.insert(S);
      plan->stage_info[S].applicable_unit = node.resource;
      insertStage(S);
      // Advance the search window past this anchor.
      // search_end becomes the next anchor's sorted index (or the end of
      // sorted_stages when all anchors have been processed).
      search_begin = anchor_idx + 1;
      search_end = (next_anchor_cursor < anchor_sorted_indices.size())
                       ? anchor_sorted_indices[next_anchor_cursor]
                       : sorted_stages.size();
      i += 1;
      continue;
    }
  }

  // Rebuild linear dependency chain over ordered_stages.
  // First, clear all existing dependencies.
  llvm::SmallVector<StageNode*> all_pipeline_stages = pipeline->getStages();
  for (StageNode* s : all_pipeline_stages) {
    s->nullifyAllDependencies();
    s->removeNullifiedDependencies();
  }

  for (size_t k = 0; k + 1 < ordered_stages.size(); ++k) {
    ordered_stages[k]->addDependency(ordered_stages[k + 1]);
  }

  return ordered_stages;
}

//===----------------------------------------------------------------------===//
// Step 2: Classify Original Stages and Create Private Resources
//===----------------------------------------------------------------------===//

/// Step 2: Walk the expanded stage list; for each original stage, determine
/// its materialization kind and create PrivateResourceSpec objects.
/// Two sub-passes are used to match the old code's allocation order:
///   Sub-pass A: buffer specs for transfer stages (kAdaptTransfer)
///   Sub-pass B: FIFO specs for compute stages (kAdaptFifoKinds)
/// This ensures all memory buffers appear before FIFOs in ktdf.private,
/// matching the expected output.
static mlir::LogicalResult classifyOriginalStages(
    llvm::ArrayRef<StageNode*> expanded_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan) {
  using FifoKey = std::pair<mlir::Attribute, mlir::Attribute>;
  llvm::DenseMap<FifoKey, PrivateResourceSpec*> fifo_specs;
  llvm::DenseMap<FifoKey, size_t> fifo_next_slot;

  // Sub-pass A: buffer specs for transfer stages
  for (size_t i = 0; i < expanded_stages.size(); ++i) {
    StageNode* current_stage = expanded_stages[i];
    if (isIntermediateStage(current_stage)) continue;

    StageMaterializationInfo& stage_info = plan->stage_info[current_stage];

    StageNode* prev_stage = (i > 0) ? expanded_stages[i - 1] : nullptr;
    StageNode* next_stage =
        (i + 1 < expanded_stages.size()) ? expanded_stages[i + 1] : nullptr;

    auto stage_op =
        mlir::cast<mlir::ktdf::StageOp>(current_stage->getOperation());

    bool prev_is_intermediate = prev_stage && isIntermediateStage(prev_stage);
    bool next_is_intermediate = next_stage && isIntermediateStage(next_stage);

    if (!prev_is_intermediate && !next_is_intermediate) continue;

    // --- Transfer stage (kAdaptTransfer): memref ↔ buffer ---
    stage_op.walk([&](mlir::ktdf::DataTransferOp transfer) {
      mlir::Type src_type = transfer.getSource().getType();
      mlir::Type dest_type = transfer.getDestination().getType();

      bool src_is_memref = mlir::isa<mlir::MemRefType>(src_type);
      bool dest_is_memref = mlir::isa<mlir::MemRefType>(dest_type);

      // Transfer involves an intermediate buffer if exactly one side is memref
      if (src_is_memref == dest_is_memref) return mlir::WalkResult::advance();

      bool intermediate_is_source = prev_is_intermediate;
      ResourceType intermediate_resource =
          intermediate_is_source ? plan->stage_info[prev_stage].stage_resource
                                 : plan->stage_info[next_stage].stage_resource;

      // Buffer shape from the memref tile sizes
      mlir::MemRefType memref_type =
          src_is_memref ? mlir::cast<mlir::MemRefType>(src_type)
                        : mlir::cast<mlir::MemRefType>(dest_type);

      llvm::SmallVector<mlir::OpFoldResult> tile_sizes =
          src_is_memref ? transfer.getMixedSourceSizes()
                        : transfer.getMixedDestSizes();
      llvm::SmallVector<int64_t> buffer_shape;
      for (mlir::OpFoldResult ofr : tile_sizes) {
        auto int_attr =
            mlir::dyn_cast<mlir::IntegerAttr>(ofr.dyn_cast<mlir::Attribute>());
        assert(int_attr &&
               "Expected static tile sizes for intermediate buffer");
        buffer_shape.push_back(int_attr.getInt());
      }

      PrivateResourceSpec* buffer_spec =
          plan->resource_factory.createMemoryBuffer(
              intermediate_resource, buffer_shape,
              memref_type.getElementType());

      // Build a minimal EdgeInfo for the factory (source/target are
      // placeholders since the new EdgeInfo only carries cost)
      scheduler::arch_view::RoutingGraph::NodeId src_node_id =
          arch_graph.getNodeIdForResource(intermediate_is_source
                                              ? intermediate_resource
                                              : stage_info.stage_resource);
      scheduler::arch_view::RoutingGraph::NodeId dst_node_id =
          arch_graph.getNodeIdForResource(intermediate_is_source
                                              ? stage_info.stage_resource
                                              : intermediate_resource);
      auto edge_opt = arch_graph.getEdgeInfo(src_node_id, dst_node_id);
      // For new graph, an edge may not exist directly between memory resource
      // and DDR/L1 (they are separated by LS node). Use a dummy edge.
      scheduler::arch_view::RoutingGraph::EdgeInfo edge{src_node_id,
                                                        dst_node_id, 1};
      if (edge_opt) edge = *edge_opt;

      mlir::OpBuilder builder(transfer->getContext());
      TransferMaterializationInfo* transfer_info =
          plan->transfer_factory.createFromTemplateWithBuffer(
              transfer, edge, intermediate_resource, stage_info.stage_resource,
              intermediate_is_source, buffer_spec, builder);

      stage_info.transfers.push_back(transfer_info);
      stage_info.kind = StageMaterializationInfo::Kind::kAdaptTransfer;

      return mlir::WalkResult::advance();
    });
  }

  // Sub-pass B: FIFO specs for compute stages
  for (size_t i = 0; i < expanded_stages.size(); ++i) {
    StageNode* current_stage = expanded_stages[i];
    if (isIntermediateStage(current_stage)) continue;

    StageMaterializationInfo& stage_info = plan->stage_info[current_stage];

    StageNode* prev_stage = (i > 0) ? expanded_stages[i - 1] : nullptr;
    StageNode* next_stage =
        (i + 1 < expanded_stages.size()) ? expanded_stages[i + 1] : nullptr;

    auto stage_op =
        mlir::cast<mlir::ktdf::StageOp>(current_stage->getOperation());

    bool prev_is_intermediate = prev_stage && isIntermediateStage(prev_stage);
    bool next_is_intermediate = next_stage && isIntermediateStage(next_stage);

    if (!prev_is_intermediate && !next_is_intermediate) continue;

    // --- Compute stage (kAdaptFifoKinds): read_from_fifo / write_to_fifo ---
    stage_op.walk([&](mlir::Operation* op) {
      auto read_op = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(op);
      auto write_op = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(op);
      if (!read_op && !write_op) return mlir::WalkResult::advance();

      bool is_read = (read_op != nullptr);
      mlir::Value fifo_slot =
          is_read ? read_op.getFifoSlot() : write_op.getFifoSlot();
      auto fifo_slot_type =
          mlir::cast<mlir::ktdf::FifoSlotType>(fifo_slot.getType());

      StageNode* adjacent_stage = is_read ? prev_stage : next_stage;
      bool adjacent_is_intermediate =
          is_read ? prev_is_intermediate : next_is_intermediate;

      if (!adjacent_is_intermediate) return mlir::WalkResult::advance();

      // fifo_src = applicable_unit of the adjacent LS-unit stage (load side)
      // fifo_dest = applicable_unit of the adjacent LS-unit stage (store side)
      // For read (consuming from left): src = adjacent.applicable_unit,
      //   dest = stage.stage_resource
      // For write (producing to right): src = stage.stage_resource,
      //   dest = adjacent.applicable_unit

      ResourceType fifo_src, fifo_dest;
      if (is_read) {
        // Left neighbor (LS-unit stage) → this compute stage
        auto adj_it = plan->stage_info.find(adjacent_stage);
        assert(adj_it != plan->stage_info.end());
        fifo_src = adj_it->second.applicable_unit.value_or(nullptr);
        fifo_dest = stage_info.stage_resource;
      } else {
        // This compute stage → right neighbor (LS-unit stage)
        auto adj_it = plan->stage_info.find(adjacent_stage);
        assert(adj_it != plan->stage_info.end());
        fifo_src = stage_info.stage_resource;
        fifo_dest = adj_it->second.applicable_unit.value_or(nullptr);
      }
      assert(fifo_src && fifo_dest &&
             "Expected valid FIFO endpoint attributes");

      mlir::Type element_type = fifo_slot_type.getElementType();
      assert(!fifo_slot_type.isDynamicNumElements());
      int64_t num_elements = fifo_slot_type.getStaticNumElements();

      FifoKey fifo_key = {fifo_src, fifo_dest};
      if (fifo_specs.find(fifo_key) == fifo_specs.end()) {
        PrivateResourceSpec* spec = plan->resource_factory.createFifo(
            fifo_src, fifo_dest, {num_elements}, element_type);
        fifo_specs[fifo_key] = spec;
        fifo_next_slot[fifo_key] = 0;
      } else {
        fifo_specs[fifo_key]->elements_per_slot.push_back(num_elements);
      }

      size_t slot_idx = fifo_next_slot[fifo_key]++;
      validateFifoSlotIndex(fifo_slot, slot_idx, fifo_specs[fifo_key]);

      // Build edge info (source/target nodes from adjacent stage resources)
      ResourceType source_resource = fifo_src;
      ResourceType dest_resource = fifo_dest;
      // Use a dummy edge since the graph may not have a direct edge between
      // stage_resource nodes (they go through LS-unit nodes).
      scheduler::arch_view::RoutingGraph::NodeId src_node_id =
          arch_graph.getNodeIdForResource(stage_info.stage_resource);
      scheduler::arch_view::RoutingGraph::NodeId adj_node_id =
          arch_graph.getNodeIdForResource(
              plan->stage_info[adjacent_stage].stage_resource);
      scheduler::arch_view::RoutingGraph::EdgeInfo edge{
          is_read ? adj_node_id : src_node_id,
          is_read ? src_node_id : adj_node_id, 1};

      TransferMaterializationInfo* transfer_info =
          plan->transfer_factory.createFromFifoOp(
              op, edge, source_resource, dest_resource, fifo_specs[fifo_key],
              slot_idx, is_read);

      stage_info.transfers.push_back(transfer_info);
      stage_info.kind = StageMaterializationInfo::Kind::kAdaptFifoKinds;

      LLVM_DEBUG({
        llvm::dbgs() << "  Stage " << current_stage->getStageId() << ": "
                     << (is_read ? "read_from_fifo" : "write_to_fifo")
                     << " - FIFO src=" << fifo_src << ", dest=" << fifo_dest
                     << ", slot " << slot_idx << ", elements=" << num_elements
                     << "\n";
      });

      return mlir::WalkResult::advance();
    });
  }  // end Sub-pass B

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Step 3: Populate Synthetic Stage Transfers
//===----------------------------------------------------------------------===//

/// Step 3: Walk intermediate stages and wire up their
/// TransferMaterializationInfo by reading adjacent stages' already-populated
/// transfer specs. Reads source/dest resources directly from adjacent stage
/// stage_resource fields (no graph edge-walk).
static mlir::LogicalResult populateIntermediateStageTransfers(
    llvm::ArrayRef<StageNode*> sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan) {
  LDBG(1) << "Populating intermediate stage transfers for "
          << sorted_stages.size() << " stages";

  for (size_t i = 0; i < sorted_stages.size(); ++i) {
    StageNode* current_stage = sorted_stages[i];

    if (!isIntermediateStage(current_stage)) continue;

    assert(plan->stage_info.count(current_stage));
    StageMaterializationInfo& stage_info = plan->stage_info[current_stage];

    StageNode* prev_stage = (i > 0) ? sorted_stages[i - 1] : nullptr;
    StageNode* next_stage =
        (i + 1 < sorted_stages.size()) ? sorted_stages[i + 1] : nullptr;

    assert(prev_stage && next_stage &&
           "Intermediate stage should have both neighbors");

    ResourceType intermediate_resource = stage_info.stage_resource;

    LLVM_DEBUG(llvm::dbgs() << "  Processing intermediate stage "
                            << current_stage->getStageId() << " (resource: " <<
               [&]() {
                 std::string s;
                 llvm::raw_string_ostream os(s);
                 intermediate_resource.print(os);
                 return os.str();
               }() << ")\n");

    stage_info.kind = StageMaterializationInfo::Kind::kSyntheticTransfer;

    if (plan->stage_info.count(prev_stage) == 0) continue;
    StageMaterializationInfo& prev_info = plan->stage_info[prev_stage];

    // Adjacent buffer transfers record the memory resource (stage_resource) as
    // their endpoint. Adjacent FIFO transfers record the LS unit
    // (applicable_unit). Accept either when matching transfers from neighbours.
    ResourceType ls_unit = stage_info.applicable_unit.value_or(nullptr);
    auto matchesIntermediate = [&](ResourceType r) -> bool {
      return r == intermediate_resource || (ls_unit && r == ls_unit);
    };

    // Read source/dest resources directly from adjacent stage stage_resource
    ResourceType inferred_source_resource =
        plan->stage_info[prev_stage].stage_resource;
    ResourceType inferred_dest_resource =
        plan->stage_info[next_stage].stage_resource;

    llvm::SmallPtrSet<const TransferMaterializationInfo*, 4>
        visited_next_transfer;
    for (const TransferMaterializationInfo* prev_transfer :
         prev_info.transfers) {
      if (!matchesIntermediate(prev_transfer->dest_resource)) continue;

      assert(prev_transfer->dest_private_resource);
      const PrivateResourceSpec* source_resource_spec =
          prev_transfer->dest_private_resource;

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
        if (matchesIntermediate(next_transfer->source_resource)) {
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

      // Build edge: use source/dest node IDs derived from adjacent stage
      // resources
      scheduler::arch_view::RoutingGraph::NodeId src_nid =
          arch_graph.getNodeIdForResource(inferred_source_resource);
      scheduler::arch_view::RoutingGraph::NodeId dst_nid =
          arch_graph.getNodeIdForResource(inferred_dest_resource);
      scheduler::arch_view::RoutingGraph::EdgeInfo edge{src_nid, dst_nid, 1};
      // Try to find a real edge (may not exist if going through LS nodes)
      if (auto real_edge = arch_graph.getEdgeInfo(src_nid, dst_nid))
        edge = *real_edge;

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
              edge, inferred_source_resource, inferred_dest_resource,
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
              inferred_source_resource.print(os);
              return os.str();
            }() << " -> "
                     <<
            [&]() {
              std::string s;
              llvm::raw_string_ostream os(s);
              inferred_dest_resource.print(os);
              return os.str();
            }() << "\n";
      });
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Orchestration
//===----------------------------------------------------------------------===//

/// Execute the 3-step path expansion algorithm
static mlir::LogicalResult executeExpansionSteps(
    llvm::SmallVector<StageNode*>& expanded_stages, PipelineTree& tree,
    PipelineNode* pipeline,
    const scheduler::arch_view::RoutingGraph::Path& full_path,
    const llvm::SmallVector<StageNode*>& sorted_stages,
    const scheduler::arch_view::RoutingGraph& arch_graph,
    PathExpansionPlan* plan, int& next_stage_id) {
  // STEP 1: Build expanded stage list
  LLVM_DEBUG(llvm::dbgs() << "\n=== Step 1: Build Expanded Stage List ===\n");
  auto expanded_stages_or =
      buildExpandedStageList(tree, pipeline, full_path, sorted_stages,
                             arch_graph, plan, next_stage_id);
  if (mlir::failed(expanded_stages_or)) {
    return mlir::failure();
  }
  expanded_stages = *expanded_stages_or;
  LLVM_DEBUG(
      debugPrintStageList(llvm::dbgs(), expanded_stages, plan, "After Step 1"));

  // STEP 2: Classify original stages and create private resources
  LLVM_DEBUG(llvm::dbgs() << "\n=== Step 2: Classify Original Stages ===\n");
  if (mlir::failed(classifyOriginalStages(expanded_stages, arch_graph, plan))) {
    return mlir::failure();
  }
  LLVM_DEBUG(
      debugPrintStageList(llvm::dbgs(), expanded_stages, plan, "After Step 2"));

  // STEP 3: Populate synthetic stage transfers
  LLVM_DEBUG(
      llvm::dbgs() << "\n=== Step 3: Populate Synthetic Stage Transfers ===\n");
  if (mlir::failed(populateIntermediateStageTransfers(expanded_stages,
                                                      arch_graph, plan))) {
    return mlir::failure();
  }
  LLVM_DEBUG(
      debugPrintStageList(llvm::dbgs(), expanded_stages, plan, "After Step 3"));

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

  // Prep Step 2: Require at least one anchor stage (a stage whose
  // applicable_unit is already specified in the input IR). Without one there
  // is no compute resource to route from/to.
  if (llvm::none_of(sorted_stages, [](StageNode* s) {
        auto op =
            mlir::dyn_cast_or_null<mlir::ktdf::StageOp>(s->getOperation());
        return op && op.getApplicableUnits().has_value();
      })) {
    return nullptr;
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

  // Prep Step 3: Assign stage_resource to every original stage.
  // This must happen before building endpoint_path so that the resource values
  // are available for the endpoint_path loop below.
  assignOriginalStageResources(sorted_stages, plan.get());

  // Prep Step 5: Build the endpoint_path (deduplicated resource sequence).
  llvm::SmallVector<ResourceType> endpoint_path;
  {
    // Collect resources for all original stages in order.
    // assignOriginalStageResources has already populated plan->stage_info at
    // this point, so read stage_resource directly from there.
    llvm::SmallVector<ResourceType> all_resources;
    for (size_t i = 0; i < sorted_stages.size(); ++i) {
      all_resources.push_back(
          plan->stage_info[sorted_stages[i]].stage_resource);
    }
    // Build the path from first resource to last via each intermediate
    // (deduplicating adjacent duplicates)
    for (ResourceType r : all_resources) {
      if (r && (endpoint_path.empty() || endpoint_path.back() != r))
        endpoint_path.push_back(r);
    }
  }

  if (endpoint_path.size() < 2) {
    return nullptr;
  }

  // Prep Step 5: Build full shortest path (node-ID sequence) by concatenating
  // per-segment BFS results across all consecutive resource pairs.
  std::optional<scheduler::arch_view::RoutingGraph::Path> full_path_opt =
      buildFullShortestPath(endpoint_path, arch_graph);
  if (!full_path_opt) {
    LLVM_DEBUG(llvm::dbgs() << "No path found for full resource path\n");
    return nullptr;
  }
  scheduler::arch_view::RoutingGraph::Path full_path = *full_path_opt;

  LLVM_DEBUG({
    llvm::dbgs() << "Full shortest path (node IDs): ";
    for (auto nid : full_path) llvm::dbgs() << nid << " ";
    llvm::dbgs() << "\n";
  });

  // Check if expansion is needed by comparing endpoint_path against the full
  // path with LS-unit nodes stripped out.  If they match, every hop in the
  // original pipeline already passes through the correct LS units (via the
  // existing FIFO types) and no structural changes are needed.
  llvm::SmallVector<ResourceType> full_path_no_ls;
  for (auto nid : full_path) {
    auto node_opt = arch_graph.getNode(nid);
    assert(node_opt);
    if (node_opt->kind != scheduler::arch_view::RoutingGraph::ResourceNode::
                              ResourceKind::LoadStoreUnit) {
      full_path_no_ls.push_back(node_opt->resource);
    }
  }
  if (full_path_no_ls == endpoint_path) {
    plan->changed = false;
    LDBG(1) << "Pipeline already legal";
    return plan;
  }

  plan->changed = true;
  plan->modified_tree = &tree;

  int next_stage_id = static_cast<int>(sorted_stages.size());
  llvm::SmallVector<StageNode*> expanded_stages;

  if (mlir::failed(executeExpansionSteps(expanded_stages, tree, pipeline,
                                         full_path, sorted_stages, arch_graph,
                                         plan.get(), next_stage_id))) {
    return nullptr;
  }

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
