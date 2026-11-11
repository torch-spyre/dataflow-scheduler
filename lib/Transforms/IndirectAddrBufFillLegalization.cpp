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
//
// IndirectAddrBufFillLegalization: stage indirect address buffer fills through
// a memory the filling unit can read from.
//
// The indirect address buffer (IAB) is co-located with a load/store unit, not
// reachable over a datapath, so the unit that fills it is the unit its stage is
// mapped to. That unit can only read from a memory that has a datapath into it:
// a store unit reads from L1 and cannot fill the buffer straight out of global
// memory. This pass runs after path expansion -- once the pipeline stages and
// their `applicable_units` exist -- and reroutes such a fill through the memory
// the filling unit does read from.
//
//===----------------------------------------------------------------------===//

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>
#include <mlir/Support/WalkResult.h>

#include <memory>

#include "dataflow-scheduler/Analysis/ArchViews/RoutingGraph.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "indirect-addr-buf-fill-legalization"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_INDIRECTADDRBUFFILLLEGALIZATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Indirect Address Buffer Fill Legalization pass"),
    llvm::cl::init(false));

namespace {

using RoutingGraph = arch_view::RoutingGraph;

//===----------------------------------------------------------------------===//
// Architecture Queries
//===----------------------------------------------------------------------===//

/// The memories \p unit_kind has an incoming datapath from -- the memories it
/// can legitimately read. \p exclude is left out of the result; it is the
/// address buffer itself, which is co-located rather than routed to.
llvm::SmallVector<ResourceType> memoriesFeeding(const RoutingGraph& graph,
                                                ResourceType unit_kind,
                                                ResourceType exclude) {
  llvm::SmallSetVector<ResourceType, 4> result;
  for (RoutingGraph::NodeId unit : graph.getNodeIdsForResource(unit_kind)) {
    for (RoutingGraph::NodeId candidate : graph.getAllNodeIds()) {
      auto node = graph.getNode(candidate);
      if (!node ||
          node->kind != RoutingGraph::ResourceNode::ResourceKind::Memory) {
        continue;
      }
      if (node->resource == exclude) continue;
      if (graph.getEdgeInfo(candidate, unit)) result.insert(node->resource);
    }
  }
  return result.takeVector();
}

/// A single step in a multi-hop transfer path: Memory -> LSU -> Memory.
struct TransferHop {
  ResourceType src_memory;
  ResourceType unit;
  ResourceType dst_memory;
};

/// Finds the shortest sequence of (Memory -> LSU -> Memory) hops connecting
/// \p source to \p dest in \p graph .
std::optional<llvm::SmallVector<TransferHop>> findTransferHops(
    const RoutingGraph& graph, ResourceType source, ResourceType dest) {
  auto src_ids = graph.getNodeIdsForResource(source);
  auto dst_ids = graph.getNodeIdsForResource(dest);
  if (src_ids.empty() || dst_ids.empty()) return std::nullopt;

  std::optional<RoutingGraph::Path> best_path;
  for (auto src_id : src_ids) {
    for (auto dst_id : dst_ids) {
      auto path = graph.findShortestPath(src_id, dst_id);
      if (path && (!best_path || path->size() < best_path->size())) {
        best_path = std::move(path);
      }
    }
  }

  if (!best_path || best_path->size() < 3 || best_path->size() % 2 == 0)
    return std::nullopt;

  llvm::SmallVector<TransferHop> hops;
  size_t num_hops = best_path->size() / 2;
  for (size_t i = 0; i < num_hops; ++i) {
    auto src_node = graph.getNode((*best_path)[2 * i]);
    auto unit_node = graph.getNode((*best_path)[2 * i + 1]);
    auto dst_node = graph.getNode((*best_path)[2 * i + 2]);
    if (!src_node || !unit_node || !dst_node) return std::nullopt;
    hops.push_back(
        {src_node->resource, unit_node->resource, dst_node->resource});
  }
  return hops;
}

/// The single unit \p stage is mapped to, or null if it is not mapped to
/// exactly one.
ResourceType getStageUnit(mlir::ktdf::StageOp stage) {
  const auto units = stage.getApplicableUnits();
  if (!units || units->size() != 1) return nullptr;
  return mlir::dyn_cast_if_present<ResourceType>(units->getValue().front());
}

//===----------------------------------------------------------------------===//
// Fill Sites
//===----------------------------------------------------------------------===//

/// A transfer filling an indirect address buffer, with the stage -- and hence
/// the unit -- that performs it.
struct FillSite {
  mlir::ktdf::DataTransferOp fill;
  mlir::ktdf::StageOp stage;
};

/// Whether \p space is the memory space of an indirect address buffer.
bool isAddressBufferSpace(const mlir::ktdf_arch::ResourceKinds& resource_kinds,
                          ResourceType space) {
  if (!space) return false;
  return resource_kinds
             .getFeature<mlir::ktdf_arch::feature::IndirectAddressBuffer>(
                 space) != nullptr;
}

void collectFillSites(mlir::ktdf::PipelineOp pipeline,
                      const mlir::ktdf_arch::ResourceKinds& resource_kinds,
                      llvm::SmallVectorImpl<FillSite>& sites) {
  for (mlir::ktdf::StageOp stage : pipeline.getStages()) {
    stage.walk([&](mlir::ktdf::DataTransferOp transfer) {
      auto dest =
          mlir::dyn_cast<mlir::MemRefType>(transfer.getDestination().getType());
      if (!dest) return;
      auto space =
          mlir::dyn_cast_if_present<ResourceType>(dest.getMemorySpace());
      if (!space || !isAddressBufferSpace(resource_kinds, space)) return;
      sites.push_back({transfer, stage});
    });
  }
}

//===----------------------------------------------------------------------===//
// IR Construction
//===----------------------------------------------------------------------===//

/// An all-zero subscript for a \p rank -dimensional buffer: a map of constants
/// taking no index operands, the form path expansion leaves on the private
/// buffers it introduces.
mlir::AffineMap createZeroSubscript(int64_t rank, mlir::MLIRContext* ctx) {
  llvm::SmallVector<mlir::AffineExpr> zeros(
      rank, mlir::getAffineConstantExpr(0, ctx));
  return mlir::AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, zeros, ctx);
}

/// Adds a resource of \p type to \p pipeline 's private resources and returns
/// the result handing it out. \p create builds the yielded value inside the
/// private region. The private op is rebuilt, since results cannot be appended
/// to an existing operation.
mlir::Value appendPrivateResource(
    mlir::ktdf::PipelineOp pipeline, mlir::Type type,
    llvm::function_ref<mlir::Value(mlir::OpBuilder&, mlir::Location)> create) {
  mlir::ktdf::PrivateOp private_op = pipeline.getPrivateOp();
  assert(private_op && "pipeline must have a private region");

  llvm::SmallVector<mlir::Type> result_types(private_op.getResultTypes());
  result_types.push_back(type);

  mlir::OpBuilder builder(private_op);
  auto new_private =
      mlir::ktdf::PrivateOp::create(builder, private_op.getLoc(), result_types);
  new_private.getRegion().takeBody(private_op.getRegion());

  mlir::ktdf::PrivateYieldOp yield = new_private.getYieldOp();
  builder.setInsertionPoint(yield);
  mlir::Value resource = create(builder, yield.getLoc());

  llvm::SmallVector<mlir::Value> yield_operands(yield.getOperands());
  yield_operands.push_back(resource);
  mlir::ktdf::PrivateYieldOp::create(builder, yield.getLoc(), yield_operands);
  yield.erase();

  for (auto [old_result, new_result] :
       llvm::zip(private_op.getResults(), new_private.getResults())) {
    old_result.replaceAllUsesWith(new_result);
  }
  private_op.erase();

  return new_private.getResults().back();
}

/// Adds a buffer of \p type to \p pipeline 's private resources.
mlir::Value appendPrivateBuffer(mlir::ktdf::PipelineOp pipeline,
                                mlir::MemRefType type) {
  return appendPrivateResource(
      pipeline, type, [&](mlir::OpBuilder& builder, mlir::Location loc) {
        return mlir::memref::AllocOp::create(builder, loc, type).getResult();
      });
}

/// The earliest stage before \p fill_stage whose unit matches \p unit.
mlir::ktdf::StageOp findStageOnUnit(mlir::ktdf::PipelineOp pipeline,
                                    mlir::ktdf::StageOp fill_stage,
                                    ResourceType unit) {
  for (mlir::ktdf::StageOp stage : pipeline.getStages()) {
    if (stage == fill_stage) break;
    if (getStageUnit(stage) == unit) return stage;
  }
  return nullptr;
}

/// Adds a stage mapped to \p unit ahead of \p insert_before (or at pipeline
/// start). It signals a fresh private token which \p dependent is made to wait
/// on.
mlir::ktdf::StageOp createStagingStage(mlir::ktdf::PipelineOp pipeline,
                                       mlir::ktdf::StageOp insert_before,
                                       mlir::ktdf::StageOp dependent,
                                       ResourceType unit) {
  auto token_type = mlir::ktdf::TokenType::get(pipeline.getContext());
  mlir::Value token = appendPrivateResource(
      pipeline, token_type, [&](mlir::OpBuilder& builder, mlir::Location loc) {
        return mlir::ktdf::CreateTokenOp::create(builder, loc).getResult();
      });

  mlir::OpBuilder builder(insert_before ? insert_before.getOperation()
                                        : *pipeline.getStages().begin());
  mlir::Location loc =
      insert_before ? insert_before.getLoc()
                    : (dependent ? dependent.getLoc() : pipeline.getLoc());
  auto staging_stage = mlir::ktdf::StageOp::create(
      builder, loc, /*depends_in=*/mlir::ValueRange{},
      /*depends_out=*/mlir::ValueRange{token});
  staging_stage.setApplicableUnitsAttr(builder.getArrayAttr({unit}));

  if (dependent) {
    dependent.addInDependency(
        mlir::cast<mlir::TypedValue<mlir::ktdf::TokenType>>(token));
  }
  return staging_stage;
}

/// Makes \p value available at \p builder 's insertion point. Values defined
/// outside \p stage already are; anything the stage itself computes is
/// recreated there, which is only sound for side-effect-free ops.
///
/// \returns the value to use, or null if it cannot be recreated.
mlir::Value recreateOutside(mlir::Value value, mlir::ktdf::StageOp stage,
                            mlir::OpBuilder& builder, mlir::IRMapping& map) {
  if (mlir::Value mapped = map.lookupOrNull(value)) return mapped;

  mlir::Operation* def = value.getDefiningOp();
  if (!def || !stage->isProperAncestor(def)) return value;
  if (!mlir::isMemoryEffectFree(def)) return nullptr;

  for (mlir::Value operand : def->getOperands()) {
    mlir::Value recreated = recreateOutside(operand, stage, builder, map);
    if (!recreated) return nullptr;
    map.map(operand, recreated);
  }

  mlir::Operation* clone = builder.clone(*def, map);
  map.map(def, clone);
  return clone->getResult(mlir::cast<mlir::OpResult>(value).getResultNumber());
}

/// Erases the ops inside \p stage that define \p values once nothing uses them,
/// following their operands back. Cleans up what a moved transfer leaves behind
/// once recreateOutside has copied it elsewhere.
void eraseDeadDefs(mlir::ValueRange values, mlir::ktdf::StageOp stage) {
  llvm::SetVector<mlir::Operation*> worklist;
  auto push = [&](mlir::Value value) {
    mlir::Operation* def = value.getDefiningOp();
    if (def && stage->isProperAncestor(def)) worklist.insert(def);
  };
  llvm::for_each(values, push);
  // An op still in use is dropped here; erasing its last user re-pushes it.
  while (!worklist.empty()) {
    mlir::Operation* op = worklist.pop_back_val();
    if (!mlir::isOpTriviallyDead(op)) continue;
    llvm::for_each(op->getOperands(), push);
    op->erase();
  }
}

//===----------------------------------------------------------------------===//
// Legalization
//===----------------------------------------------------------------------===//

/// The scf.if ops guarding \p op inside \p stage, outermost first. Fails if
/// some other region-holding op stands in between, whose semantics this pass
/// cannot reproduce in another stage.
mlir::LogicalResult collectGuards(
    mlir::Operation* op, mlir::ktdf::StageOp stage,
    llvm::SmallVectorImpl<mlir::scf::IfOp>& guards) {
  for (mlir::Operation* parent = op->getParentOp();
       parent != stage.getOperation(); parent = parent->getParentOp()) {
    auto if_op = mlir::dyn_cast<mlir::scf::IfOp>(parent);
    if (!if_op) {
      return op->emitError()
             << PASS_NAME << ": indirect address buffer fill is nested in '"
             << parent->getName() << "', which cannot be replicated";
    }
    guards.push_back(if_op);
  }
  std::reverse(guards.begin(), guards.end());
  return mlir::success();
}

/// Reroutes \p site 's fill through a memory its unit reads from, if it does
/// not already read from the memory the fill sources.
mlir::LogicalResult legalizeFillSite(const FillSite& site,
                                     const RoutingGraph& graph) {
  mlir::ktdf::DataTransferOp fill = site.fill;
  mlir::ktdf::StageOp stage = site.stage;
  auto pipeline = stage->getParentOfType<mlir::ktdf::PipelineOp>();
  assert(pipeline && "stage must be inside a pipeline");

  ResourceType fill_unit = getStageUnit(stage);
  if (!fill_unit) {
    return stage.emitError()
           << PASS_NAME
           << ": stage filling an indirect address buffer must be mapped to "
              "exactly one unit";
  }

  auto source_type =
      mlir::dyn_cast<mlir::MemRefType>(fill.getSource().getType());
  if (!source_type) {
    return fill.emitError()
           << PASS_NAME
           << ": indirect address buffer fill must read a memref in a known "
              "memory space";
  }
  ResourceType source_memory =
      mlir::dyn_cast_if_present<ResourceType>(source_type.getMemorySpace());
  if (!source_memory) {
    return fill.emitError()
           << PASS_NAME
           << ": indirect address buffer fill must read a memref in a known "
              "memory space";
  }
  // The destination is always a MemRefType with a KindAttr memory space:
  // collectFillSites only admits transfers whose destination space passes
  // isAddressBufferSpace, which rejects null and non-KindAttr spaces.
  ResourceType dest_memory = mlir::cast<ResourceType>(
      mlir::cast<mlir::MemRefType>(fill.getDestination().getType())
          .getMemorySpace());

  llvm::SmallVector<ResourceType> reachable =
      memoriesFeeding(graph, fill_unit, dest_memory);
  if (llvm::is_contained(reachable, source_memory)) {
    LDBG(1) << "  fill from " << source_memory << " on " << fill_unit
            << " is already legal";
    return mlir::success();
  }

  if (reachable.size() != 1) {
    return fill.emitError()
           << PASS_NAME << ": cannot stage the fill of " << dest_memory
           << " from " << source_memory << ": unit " << fill_unit << " reads "
           << reachable.size() << " memories, expected exactly one";
  }
  ResourceType staging_memory = reachable.front();

  // Discover the multi-hop path from source to staging memory.
  auto hops = findTransferHops(graph, source_memory, staging_memory);
  if (!hops || hops->empty()) {
    return fill.emitError() << PASS_NAME << ": no routing path found from "
                            << source_memory << " to " << staging_memory;
  }

  const auto buffer_shape = fill.getStaticSourceSizesArray();
  if (!buffer_shape) {
    return fill.emitError()
           << PASS_NAME
           << ": indirect address buffer fill must have static sizes to be "
              "staged";
  }

  if (!pipeline.getPrivateOp()) {
    return pipeline.emitError()
           << PASS_NAME
           << ": pipeline has no private region to hold the staging buffer";
  }

  llvm::SmallVector<mlir::scf::IfOp> guards;
  if (mlir::failed(collectGuards(fill, stage, guards))) return mlir::failure();

  mlir::MLIRContext* ctx = fill.getContext();
  mlir::AffineMap zero_subscript =
      createZeroSubscript(buffer_shape->size(), ctx);
  mlir::DictionaryAttr fill_attrs = fill->getDiscardableAttrDictionary();
  llvm::SmallVector<mlir::OpFoldResult> sizes = fill.getMixedSourceSizes();

  // 1. Resolve or synthesize stages for all hops.
  // Rebuilding the private op invalidates the values it handed out earlier, so
  // any stage tokens must be added before staging buffers are allocated.
  llvm::SmallVector<mlir::ktdf::StageOp> hop_stages;
  mlir::ktdf::StageOp prev_stage = nullptr;
  for (size_t hop_idx = 0; hop_idx < hops->size(); ++hop_idx) {
    const TransferHop& hop = (*hops)[hop_idx];
    mlir::ktdf::StageOp hop_stage = findStageOnUnit(pipeline, stage, hop.unit);
    if (!hop_stage) {
      // The dependent stage is the next hop's stage, or the fill stage for the
      // final hop.
      mlir::ktdf::StageOp dependent =
          (hop_idx + 1 == hops->size()) ? stage : nullptr;
      mlir::ktdf::StageOp insert_before =
          hop_stages.empty() ? *pipeline.getStages().begin() : stage;
      hop_stage =
          createStagingStage(pipeline, insert_before, dependent, hop.unit);
    }
    if (prev_stage && !hop_stage.getDependsOut().empty()) {
      // Chain previous newly created hop stage to this hop stage.
      if (!prev_stage.getDependsOut().empty()) {
        auto prev_token = mlir::cast<mlir::TypedValue<mlir::ktdf::TokenType>>(
            prev_stage.getDependsOut().front());
        hop_stage.addInDependency(prev_token);
      }
    }
    prev_stage = hop_stage;
    hop_stages.push_back(hop_stage);
  }

  // 2. Allocate intermediate private buffers for each hop.
  llvm::SmallVector<size_t> buffer_result_indices;
  for (const TransferHop& hop : *hops) {
    auto buffer_type = mlir::MemRefType::get(
        *buffer_shape, source_type.getElementType(),
        mlir::MemRefLayoutAttrInterface(), hop.dst_memory);
    appendPrivateBuffer(pipeline, buffer_type);
    buffer_result_indices.push_back(pipeline.getPrivateOp().getNumResults() -
                                    1);
  }

  // 3. Emit intermediate transfers for each hop.
  mlir::Value current_source = fill.getSource();
  mlir::AffineMap current_source_map =
      fill.getSourceMap().value_or(mlir::AffineMap());
  llvm::SmallVector<mlir::Value> current_source_indices(
      fill.getSourceIndices());

  for (size_t hop_idx = 0; hop_idx < hops->size(); ++hop_idx) {
    const TransferHop& hop = (*hops)[hop_idx];
    mlir::ktdf::StageOp hop_stage = hop_stages[hop_idx];
    mlir::Value dst_buffer =
        pipeline.getPrivateOp().getResult(buffer_result_indices[hop_idx]);

    LDBG(1) << "  staging fill hop " << hop_idx << " (" << hop.src_memory
            << " -> " << hop.dst_memory << ") on " << hop.unit;

    mlir::OpBuilder builder(ctx);
    builder.setInsertionPointToEnd(hop_stage.getBody());
    mlir::IRMapping map;
    for (mlir::scf::IfOp guard : guards) {
      mlir::Value condition =
          recreateOutside(guard.getCondition(), stage, builder, map);
      if (!condition) {
        return guard.emitError()
               << PASS_NAME
               << ": guard condition cannot be recreated in the staging stage";
      }
      auto staged_guard =
          mlir::scf::IfOp::create(builder, guard.getLoc(), condition,
                                  /*withElseRegion=*/false);
      builder.setInsertionPointToStart(staged_guard.thenBlock());
    }

    mlir::Value staged_source = current_source;
    if (hop_idx == 0) {
      staged_source = recreateOutside(fill.getSource(), stage, builder, map);
      if (!staged_source) {
        return fill.emitError()
               << PASS_NAME
               << ": fill source cannot be recreated in the staging stage";
      }
    }

    llvm::SmallVector<mlir::Value> staged_indices;
    if (hop_idx == 0) {
      for (mlir::Value index : current_source_indices) {
        mlir::Value recreated = recreateOutside(index, stage, builder, map);
        if (!recreated) {
          return fill.emitError()
                 << PASS_NAME
                 << ": fill source subscript cannot be recreated in the "
                    "staging stage";
        }
        staged_indices.push_back(recreated);
      }
    }

    auto staging = mlir::ktdf::DataTransferOp::create(
        builder, fill.getLoc(), staged_source, current_source_map,
        staged_indices, sizes, dst_buffer, zero_subscript,
        /*dest_indices=*/mlir::ValueRange{}, sizes);
    staging->setDiscardableAttrs(fill_attrs);

    current_source = dst_buffer;
    current_source_map = zero_subscript;
    current_source_indices.clear();
  }

  // 4. The fill itself, now reading the final staging buffer.
  mlir::Value final_buffer =
      pipeline.getPrivateOp().getResult(buffer_result_indices.back());
  mlir::OpBuilder builder(fill);
  auto staged_fill = mlir::ktdf::DataTransferOp::create(
      builder, fill.getLoc(), final_buffer, zero_subscript,
      /*source_indices=*/mlir::ValueRange{}, sizes, fill.getDestination(),
      fill.getDestMap().value_or(mlir::AffineMap()), fill.getDestIndices(),
      fill.getMixedDestSizes(), fill.getTransferMode());
  staged_fill->setDiscardableAttrs(fill_attrs);
  llvm::SmallVector<mlir::Value> old_operands(fill->getOperands());
  fill.erase();
  eraseDeadDefs(old_operands, stage);

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct IndirectAddrBufFillLegalizationPass
    : public impl::IndirectAddrBufFillLegalizationPassBase<
          IndirectAddrBufFillLegalizationPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    mlir::ModuleOp module_op = getOperation();

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module_op->emitError(
          "Unable to import the device specification. This could happen if the "
          "device spec file is empty or contains multiple devices");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<mlir::ktdf_arch::ResourceKinds>(*device);
    auto& routing_graph = device_manager.getOrCreateView<RoutingGraph>(*device);

    llvm::SmallVector<FillSite> sites;
    module_op.walk([&](mlir::ktdf::PipelineOp pipeline) {
      collectFillSites(pipeline, resource_kinds, sites);
    });

    for (const FillSite& site : sites) {
      if (mlir::failed(legalizeFillSite(site, routing_graph))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
scheduler::createIndirectAddrBufFillLegalizationPass() {
  return std::make_unique<IndirectAddrBufFillLegalizationPass>();
}
