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

/// All nodes standing for resources of kind \p kind. A kind may appear more
/// than once: the routing graph keeps one node per declaration, and a unit or
/// memory declared in several sub-cores is several nodes.
llvm::SmallVector<RoutingGraph::NodeId> nodesOfKind(const RoutingGraph& graph,
                                                    mlir::Attribute kind) {
  llvm::SmallVector<RoutingGraph::NodeId> result;
  for (RoutingGraph::NodeId node_id : graph.getAllNodeIds()) {
    auto node = graph.getNode(node_id);
    if (node && node->resource == kind) result.push_back(node_id);
  }
  return result;
}

/// The memories \p unit_kind has an incoming datapath from -- the memories it
/// can legitimately read. \p exclude is left out of the result; it is the
/// address buffer itself, which is co-located rather than routed to.
llvm::SmallVector<mlir::Attribute> memoriesFeeding(const RoutingGraph& graph,
                                                   mlir::Attribute unit_kind,
                                                   mlir::Attribute exclude) {
  llvm::SmallSetVector<mlir::Attribute, 4> result;
  for (RoutingGraph::NodeId unit : nodesOfKind(graph, unit_kind)) {
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

/// Whether \p unit_kind can carry data from \p source to \p dest, i.e. has an
/// incoming datapath from \p source and an outgoing one to \p dest.
bool unitConnects(const RoutingGraph& graph, mlir::Attribute unit_kind,
                  mlir::Attribute source, mlir::Attribute dest) {
  for (RoutingGraph::NodeId unit : nodesOfKind(graph, unit_kind)) {
    for (RoutingGraph::NodeId from : nodesOfKind(graph, source)) {
      if (!graph.getEdgeInfo(from, unit)) continue;
      for (RoutingGraph::NodeId to : nodesOfKind(graph, dest)) {
        if (graph.getEdgeInfo(unit, to)) return true;
      }
    }
  }
  return false;
}

/// The single unit \p stage is mapped to, or null if it is not mapped to
/// exactly one.
mlir::Attribute getStageUnit(mlir::ktdf::StageOp stage) {
  const auto units = stage.getApplicableUnits();
  if (!units || units->size() != 1) return nullptr;
  return units->getValue().front();
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
                          mlir::Attribute space) {
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
      if (!dest || !isAddressBufferSpace(resource_kinds, dest.getMemorySpace()))
        return;
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

/// Adds a buffer of \p type to \p pipeline 's private resources and returns the
/// result handing it out. The private op is rebuilt, since results cannot be
/// appended to an existing operation.
mlir::Value appendPrivateBuffer(mlir::ktdf::PipelineOp pipeline,
                                mlir::MemRefType type) {
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
  auto alloc = mlir::memref::AllocOp::create(builder, yield.getLoc(), type);

  llvm::SmallVector<mlir::Value> yield_operands(yield.getOperands());
  yield_operands.push_back(alloc.getResult());
  mlir::ktdf::PrivateYieldOp::create(builder, yield.getLoc(), yield_operands);
  yield.erase();

  for (auto [old_result, new_result] :
       llvm::zip(private_op.getResults(), new_private.getResults())) {
    old_result.replaceAllUsesWith(new_result);
  }
  private_op.erase();

  return new_private.getResults().back();
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

/// The earliest stage before \p fill_stage whose unit can carry data from
/// \p source to \p staging .
mlir::ktdf::StageOp findStagingStage(mlir::ktdf::PipelineOp pipeline,
                                     mlir::ktdf::StageOp fill_stage,
                                     const RoutingGraph& graph,
                                     mlir::Attribute source,
                                     mlir::Attribute staging) {
  for (mlir::ktdf::StageOp stage : pipeline.getStages()) {
    if (stage == fill_stage) break;
    mlir::Attribute unit = getStageUnit(stage);
    if (unit && unitConnects(graph, unit, source, staging)) return stage;
  }
  return nullptr;
}

/// Reroutes \p site 's fill through a memory its unit reads from, if it does
/// not already read from the memory the fill sources.
mlir::LogicalResult legalizeFillSite(const FillSite& site,
                                     const RoutingGraph& graph) {
  mlir::ktdf::DataTransferOp fill = site.fill;
  mlir::ktdf::StageOp stage = site.stage;
  auto pipeline = stage->getParentOfType<mlir::ktdf::PipelineOp>();
  assert(pipeline && "stage must be inside a pipeline");

  mlir::Attribute fill_unit = getStageUnit(stage);
  if (!fill_unit) {
    return stage.emitError()
           << PASS_NAME
           << ": stage filling an indirect address buffer must be mapped to "
              "exactly one unit";
  }

  auto source_type =
      mlir::dyn_cast<mlir::MemRefType>(fill.getSource().getType());
  if (!source_type || !source_type.getMemorySpace()) {
    return fill.emitError()
           << PASS_NAME
           << ": indirect address buffer fill must read a memref in a known "
              "memory space";
  }
  mlir::Attribute source_memory = source_type.getMemorySpace();
  auto dest_memory =
      mlir::cast<mlir::MemRefType>(fill.getDestination().getType())
          .getMemorySpace();

  llvm::SmallVector<mlir::Attribute> reachable =
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
  mlir::Attribute staging_memory = reachable.front();

  mlir::ktdf::StageOp staging_stage =
      findStagingStage(pipeline, stage, graph, source_memory, staging_memory);
  if (!staging_stage) {
    return fill.emitError()
           << PASS_NAME << ": no earlier stage can carry the index data from "
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

  LDBG(1) << "  staging fill of " << dest_memory << " through "
          << staging_memory << " in stage on " << getStageUnit(staging_stage);

  mlir::MLIRContext* ctx = fill.getContext();
  auto buffer_type =
      mlir::MemRefType::get(*buffer_shape, source_type.getElementType(),
                            mlir::MemRefLayoutAttrInterface(), staging_memory);
  mlir::AffineMap buffer_subscript =
      createZeroSubscript(buffer_type.getRank(), ctx);
  mlir::Value buffer = appendPrivateBuffer(pipeline, buffer_type);

  // The source -> staging copy, under the fill's own guards so that it only
  // runs on the iterations the fill needs it.
  mlir::OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(staging_stage.getBody());
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

  llvm::SmallVector<mlir::Value> source_indices;
  for (mlir::Value index : fill.getSourceIndices()) {
    mlir::Value recreated = recreateOutside(index, stage, builder, map);
    if (!recreated) {
      return fill.emitError()
             << PASS_NAME
             << ": fill source subscript cannot be recreated in the staging "
                "stage";
    }
    source_indices.push_back(recreated);
  }

  llvm::SmallVector<mlir::OpFoldResult> sizes = fill.getMixedSourceSizes();
  mlir::ktdf::DataTransferOp::create(
      builder, fill.getLoc(), fill.getSource(),
      fill.getSourceMap().value_or(mlir::AffineMap()), source_indices, sizes,
      buffer, buffer_subscript, /*dest_indices=*/mlir::ValueRange{}, sizes);

  // The fill itself, now reading the staging buffer.
  builder.setInsertionPoint(fill);
  mlir::ktdf::DataTransferOp::create(
      builder, fill.getLoc(), buffer, buffer_subscript,
      /*source_indices=*/mlir::ValueRange{}, sizes, fill.getDestination(),
      fill.getDestMap().value_or(mlir::AffineMap()), fill.getDestIndices(),
      fill.getMixedDestSizes());
  fill.erase();

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
