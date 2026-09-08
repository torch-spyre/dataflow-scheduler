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
/// Phase 1: Operand-Based KTDF-to-DFIR Lowering
///
/// Converts KTDF pipelines/stages with applicable_units attributes to
/// operand-based ktdf_lowering IR using dataflow.get_unit, uniform maps,
/// and uniform queries.
///
//
//===----------------------------------------------------------------------===//

#include <map>

#include "dataflow-scheduler/Analysis/WriteSetScan.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/ComponentClassifier.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/PipelineExecutionTransform.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/ScratchpadConflicts.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/SignalInsertion.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UniformInfra.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UnitMaterializer.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/GlobalStageDAG.h"
#include "dataflow-scheduler/Dialect/KTDF/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "ktir/Dialect/KTDP/KTDP.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "ktdf-to-ktdflowering"
#define DEBUG_TYPE PASS_NAME

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_KTDFTOKTDFLOWERINGPASS
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/Passes.h.inc"
}  // namespace scheduler

namespace {

// ---------------------------------------------------------------------------
// Check whether there is a genuine cross-iteration scratchpad dependency
// between load_stage and store_stage.
//
// The load stage reads from a shared scratchpad (memref source) into FIFOs or
// other local buffers.  The store stage writes the computed result back into
// that same scratchpad on the next iteration, creating a loop-carried RAW.
//
// Algorithm:
//   1. Collect every ktdf.data_transfer inside load_stage whose source
//      is a memref (i.e. not a FIFO).
//   2. For each such transfer, ask scheduler::regionWritesTo whether
//      store_stage's region writes through the same source memref.
//   3. If any transfer matches, a back-edge is needed.  All matching
//      transfers are returned via transfers_with_dependency so the caller
//      can attach them to BackEdgeInfo for later use during signal insertion.
//
// Returns true iff at least one transfer with a dependency was found.
// ---------------------------------------------------------------------------
static bool stageHasBackedgeDependency(
    mlir::ktdf::StageOp load_stage, mlir::ktdf::StageOp store_stage,
    llvm::SmallVectorImpl<mlir::ktdf::DataTransferOp>&
        transfers_with_dependency) {
  // Collect data transfers in load_stage whose source is a memref (not a
  // FIFO) — these are the reads from the shared scratchpad.
  llvm::SmallVector<mlir::ktdf::DataTransferOp, 4> load_transfers;
  load_stage.walk([&](mlir::ktdf::DataTransferOp xfer) {
    if (!xfer.isSourceFifo()) load_transfers.push_back(xfer);
  });

  // For each load transfer, check whether store_stage writes to the same
  // scratchpad memref that load_stage reads from.
  mlir::Region& store_region = store_stage.getRegion();
  for (mlir::ktdf::DataTransferOp xfer : load_transfers) {
    if (scheduler::regionWritesTo(store_region, xfer.getSource()))
      transfers_with_dependency.push_back(xfer);
  }
  return !transfers_with_dependency.empty();
}

// ---------------------------------------------------------------------------
// Collect all scf.for loops that are ancestors of `pipeline`.
// ---------------------------------------------------------------------------
static llvm::DenseSet<mlir::Operation*> collectAncestorLoops(
    mlir::ktdf::PipelineOp pipeline) {
  llvm::DenseSet<mlir::Operation*> loops;
  for (mlir::Operation* p = pipeline->getParentOp(); p; p = p->getParentOp())
    if (mlir::isa<mlir::scf::ForOp>(p)) loops.insert(p);
  return loops;
}

// ---------------------------------------------------------------------------
// Walk the def-use chains of each transfer's indices and collect the distinct
// scf.for loops whose induction variables appear in those chains, restricted
// to loops in `ancestor_loops` (all ancestor scf.for loops of the enclosing
// pipeline).
//
// Algorithm:
//   - Maintain a worklist of Values, seeded with:
//       * the source indices of each load-stage transfer (scratchpad read
//         address), and
//       * the dest indices of each store-stage transfer (scratchpad write
//         address) — these may involve additional loop IVs that must also
//         contribute guards.
//   - For each Value: if it is a BlockArgument that is the induction variable
//     of an scf.for in ancestor_loops, record that ForOp (once) unless
//     its static trip count is <= 1.
//   - If it is an Operation result, push all operands of the defining op onto
//     the worklist to follow the chain transitively.
// ---------------------------------------------------------------------------
static llvm::SmallVector<mlir::scf::ForOp, 2> collectDependentLoops(
    llvm::ArrayRef<mlir::ktdf::DataTransferOp> load_transfers,
    llvm::ArrayRef<mlir::ktdf::DataTransferOp> store_transfers,
    const llvm::DenseSet<mlir::Operation*>& ancestor_loops) {
  llvm::SmallVector<mlir::scf::ForOp, 2> result;
  llvm::DenseSet<mlir::Operation*> seen;
  llvm::DenseSet<mlir::Value> visited;
  llvm::SmallVector<mlir::Value> worklist;

  for (mlir::ktdf::DataTransferOp xfer : load_transfers)
    for (mlir::Value idx : xfer.getSourceIndices()) worklist.push_back(idx);
  for (mlir::ktdf::DataTransferOp xfer : store_transfers)
    for (mlir::Value idx : xfer.getDestIndices()) worklist.push_back(idx);

  // Also seed with operands of every parent op up to the stage boundary for
  // both load and store transfers.  This is a conservative approximation of
  // control + data dependence: it covers the pattern where the scratchpad
  // address is constant (e.g. always [%arg0, 0, 0]) so the chunk IV does not
  // appear in any index, but does appear in an enclosing scf.if condition or
  // scf.for bound — any of which may be a parent op on the path to the stage.
  // Walking all parent operands (not just the first scf.if condition) handles
  // multiple levels of nested conditions and for-loop bounds uniformly.
  auto seedParentOperands = [&](mlir::ktdf::DataTransferOp xfer) {
    for (mlir::Operation* p = xfer->getParentOp(); p; p = p->getParentOp()) {
      // Stop at the stage boundary — don't escape into enclosing stages.
      if (mlir::isa<mlir::ktdf::StageOp>(p)) break;
      for (mlir::Value operand : p->getOperands()) worklist.push_back(operand);
    }
  };
  for (mlir::ktdf::DataTransferOp xfer : load_transfers)
    seedParentOperands(xfer);
  for (mlir::ktdf::DataTransferOp xfer : store_transfers)
    seedParentOperands(xfer);

  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) continue;

    if (auto block_arg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
      mlir::Operation* parent = block_arg.getOwner()->getParentOp();
      auto for_op = mlir::dyn_cast_or_null<mlir::scf::ForOp>(parent);
      if (!for_op) continue;
      if (for_op.getInductionVar() != v) continue;
      // Discard trivial loops (trip count statically known to be <= 1).
      if (auto tc = for_op.getStaticTripCount(); tc && tc->ule(1)) continue;
      if (!ancestor_loops.count(for_op.getOperation())) continue;
      if (!seen.insert(for_op.getOperation()).second) continue;
      result.push_back(for_op);
    } else if (mlir::Operation* def = v.getDefiningOp()) {
      for (mlir::Value operand : def->getOperands())
        worklist.push_back(operand);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Augment global_dag with cross-iteration back-edges and populate
// back_edges_out for every pipeline that has a genuine scratchpad RAW
// dependency between its load and store stages.
//
// Load and store stages are identified from the local token-chain topology of
// each pipeline's direct child stages:
//   - Load stage candidates: stages with no token predecessors (DAG sources).
//   - Store stage candidates: stages with no token successors (DAG sinks).
//
// For each (load, store) source/sink pair, stageHasBackedgeDependency verifies
// that store_stage actually writes to a memref that load_stage loads from
// before recording the back-edge.  collectDependentLoops then walks the
// def-use chains of the qualifying transfers to find all non-trivial loops
// whose IVs index the scratchpad — these are stored in BackEdgeInfo as
// dependent_loops and drive the multi-loop conjunction guards emitted by
// insertSignals.  If no dependent loops are found the candidate is skipped
// (no back-edge needed).
//
// The back-edge is:
//   store_stage → load_stage
// injected into global_dag so computeScratchpadConflicts fires on the edge,
// while back_edges_out lets insertSignals emit guarded signals instead of the
// unconditional post-producer signal used for normal edges.
// ---------------------------------------------------------------------------
static void addScfForPipelineBackEdges(
    mlir::func::FuncOp func, mlir::ktdf::StageDependencyDAG& global_dag,
    llvm::SmallVectorImpl<BackEdgeInfo>& back_edges_out) {
  func.walk([&](mlir::ktdf::PipelineOp pipeline) {
    // Collect this pipeline's direct child stages.
    llvm::SmallVector<mlir::ktdf::StageOp, 8> pipeline_stages;
    for (auto& op : pipeline.getBodyRegion().front()) {
      if (auto stage = mlir::dyn_cast<mlir::ktdf::StageOp>(op))
        pipeline_stages.push_back(stage);
    }
    if (pipeline_stages.size() < 2) return;

    // Build a local token-chain DAG for just the direct children of this
    // pipeline so we can identify source and sink stages.
    mlir::ktdf::StageDependencyDAG local_dag;
    if (mlir::failed(
            mlir::ktdf::analyzeStageDependencies(pipeline_stages, local_dag)))
      return;

    // Load stage candidates = DAG sources (no token predecessors).
    // Store stage candidates = DAG sinks (no token successors).
    llvm::SmallVector<mlir::ktdf::StageOp, 2> load_candidates, store_candidates;
    for (auto stage : pipeline_stages) {
      mlir::Operation* op = stage.getOperation();
      auto& preds = local_dag.predecessors[op];
      auto& succs = local_dag.successors[op];
      if (preds.empty()) load_candidates.push_back(stage);
      if (succs.empty()) store_candidates.push_back(stage);
    }

    // Check every (load, store) candidate pair for a genuine cross-iteration
    // scratchpad dependency and record a back-edge if one is found.
    for (auto load_stage : load_candidates) {
      for (auto store_stage : store_candidates) {
        if (load_stage == store_stage) continue;

        mlir::Operation* load_stage_op = load_stage.getOperation();
        mlir::Operation* store_stage_op = store_stage.getOperation();

        llvm::SmallVector<mlir::ktdf::DataTransferOp, 4>
            transfers_with_dependency;
        if (!stageHasBackedgeDependency(load_stage, store_stage,
                                        transfers_with_dependency))
          continue;

        // Collect store-stage transfers that write to a non-FIFO destination
        // (scratchpad write-back).
        // TODO: this considers all non-FIFO store transfers, even those whose
        // destination memref has no dependency with load_stage.  A tighter
        // filter would restrict to transfers whose dest memref matches one of
        // the sources in transfers_with_dependency.
        llvm::SmallVector<mlir::ktdf::DataTransferOp, 4> store_transfers;
        store_stage.walk([&](mlir::ktdf::DataTransferOp xfer) {
          if (!xfer.isDestFifo()) store_transfers.push_back(xfer);
        });

        llvm::DenseSet<mlir::Operation*> ancestor_loops =
            collectAncestorLoops(pipeline);
        llvm::SmallVector<mlir::scf::ForOp, 2> dependent_loops =
            collectDependentLoops(transfers_with_dependency, store_transfers,
                                  ancestor_loops);
        if (dependent_loops.empty()) continue;

        LDBG(1) << "  Adding scf.for pipeline back-edge: store_stage -> "
                   "load_stage (cross-iteration scratchpad RAW)";
        global_dag.successors[store_stage_op].push_back(load_stage_op);
        global_dag.predecessors[load_stage_op].push_back(store_stage_op);

        back_edges_out.emplace_back(store_stage_op, load_stage_op,
                                    std::move(dependent_loops),
                                    std::move(transfers_with_dependency));
      }
    }
  });
}

struct KTDFToKTDFLoweringPass
    : public impl::KTDFToKTDFLoweringPassBase<KTDFToKTDFLoweringPass> {
  KTDFToKTDFLoweringPass()
      : scheduler_ctx_(SchedulerExtContext::dummyContext()) {}

  KTDFToKTDFLoweringPass(const SchedulerExtContext& scheduler_ctx)
      : scheduler_ctx_(scheduler_ctx) {}

  void runOnOperation() override {
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

    llvm::SmallVector<mlir::func::FuncOp, 4> funcs;
    module_op.walk([&](mlir::func::FuncOp func) {
      funcs.push_back(func);
      return mlir::WalkResult::skip();
    });
    for (auto func : funcs) {
      LDBG(1) << "Running " << PASS_NAME << " on " << func.getName();

      // Pre-compute stages for reuse across multiple steps
      llvm::SmallVector<mlir::ktdf::StageOp, 8> stages;
      mlir::ktdf::collectStages(func, stages);

      if (stages.empty()) {
        LDBG(1) << "  No stages found - skipped";
        continue;
      }

      // Step 1: Classify components
      ComponentClassifier classifier(func);
      ComponentClassification components;
      if (mlir::failed(classifier.classify(stages, components))) {
        return signalPassFailure();
      }

      // Step 2: Extract grid size
      int grid_size = 0;
      if (mlir::failed(extractGridSize(func, grid_size))) {
        return signalPassFailure();
      }

      // Step 3: Materialize units
      UnitSSAMap unit_ssa_map;
      mlir::OpBuilder builder(&func.getBody().front(),
                              func.getBody().front().begin());

      UnitMaterializer materializer(func);
      if (mlir::failed(materializer.materialize(components, grid_size,
                                                unit_ssa_map, builder))) {
        return signalPassFailure();
      }

      // Step 4: Create uniform maps and queries
      QueriedUnitsMap queried_units;
      UniformMapsStorage uniform_maps;

      UniformInfra uniform_infra(func);
      if (mlir::failed(uniform_infra.createMapsAndQueries(
              components, grid_size, unit_ssa_map, queried_units, uniform_maps,
              builder))) {
        return signalPassFailure();
      }

      // Step 5: Wire queried units from Step 4 to stages based on
      // applicable_units
      StageToUnitsMap stage_to_units;
      int wired_queries = 0;
      for (auto stage : stages) {
        auto applicable_units = stage.getApplicableUnitsAttr();
        assert(applicable_units && "Stage should have applicable units");

        for (auto component : applicable_units.getValue()) {
          // Check if stage is in a parallel region
          mlir::Operation* parallel_parent =
              mlir::ktdf::findParallelParent(stage);

          if (parallel_parent) {
            // Parallel stage: look in queried_units.parallel for all corelets
            auto parallel_op =
                mlir::dyn_cast<mlir::ktdf::ParallelOp>(parallel_parent);
            int num_corelets = parallel_op.getNumInstances();
            for (int corelet = 0; corelet < num_corelets; ++corelet) {
              auto parallel_key = std::make_pair(
                  std::make_pair(parallel_parent, component), corelet);
              auto query_it = queried_units.parallel.find(parallel_key);
              if (query_it != queried_units.parallel.end()) {
                stage_to_units.mapping[stage.getOperation()].push_back(
                    query_it->second);
                wired_queries++;
              }
            }
          } else {
            // Non-parallel stage: first try queried_units.non_parallel
            auto query_it = queried_units.non_parallel.find(component);
            if (query_it != queried_units.non_parallel.end()) {
              stage_to_units.mapping[stage.getOperation()].push_back(
                  query_it->second);
              wired_queries++;
            } else {
              // If not found in non_parallel, this component might be in a
              // parallel region Find any parallel region that has this
              // component and use those units
              for (auto& [parallel_op, parallel_comps] :
                   components.parallel_components_map) {
                if (parallel_comps.contains(component)) {
                  auto parallel_parent_op =
                      mlir::dyn_cast<mlir::ktdf::ParallelOp>(parallel_op);
                  int num_corelets = parallel_parent_op.getNumInstances();
                  for (int corelet = 0; corelet < num_corelets; ++corelet) {
                    auto parallel_key = std::make_pair(
                        std::make_pair(parallel_op, component), corelet);
                    auto parallel_query_it =
                        queried_units.parallel.find(parallel_key);
                    if (parallel_query_it != queried_units.parallel.end()) {
                      stage_to_units.mapping[stage.getOperation()].push_back(
                          parallel_query_it->second);
                      wired_queries++;
                    }
                  }
                  break;  // Found the component in a parallel region, stop
                          // searching
                }
              }
            }
          }
        }
      }

      if (wired_queries == 0) {
        func.emitError("no queries wired to stages");
        return signalPassFailure();
      }

      // Step 6: Build the global flat stage DAG once, spanning all nesting
      // levels. Nodes are leaf StageOps only; used for conflict detection (Step
      // 7) and signal insertion (Step 8).
      mlir::ktdf::StageDependencyDAG global_dag;
      if (mlir::failed(mlir::ktdf::buildGlobalStageDAG(func, global_dag))) {
        func.emitError("failed to build global stage DAG");
        return signalPassFailure();
      }

      // Step 6b: Augment global_dag with cross-iteration back-edges for every
      // pipeline that has a genuine scratchpad RAW dependency between its load
      // and store stages.  Also collects BackEdgeInfo so insertSignals can emit
      // loop-IV-guarded signals for these edges.
      llvm::SmallVector<BackEdgeInfo> back_edges;
      addScfForPipelineBackEdges(func, global_dag, back_edges);

      // Step 7: Compute scratchpad conflicts across all pipelines using the
      // global leaf-stage DAG.
      std::map<std::pair<mlir::Operation*, mlir::Operation*>,
               llvm::SmallVector<scheduler::ResourceType, 2>>
          conflicts;
      if (mlir::failed(computeScratchpadConflicts(stage_to_units, global_dag,
                                                  resource_kinds, conflicts))) {
        func.emitError("failed to compute scratchpad conflicts");
        return signalPassFailure();
      }
      LDBG(1) << "Number of scratchpad conflicts found: " << conflicts.size();

      // Step 8: Insert signal operations for all conflicting global DAG edges,
      // before any pipeline transformation mutates the IR.  Back-edge signals
      // are wrapped in scf.if guards (iv != lb / iv != ub-step).
      if (mlir::failed(insertSignals(func.getLoc(), stage_to_units, global_dag,
                                     conflicts, back_edges))) {
        return signalPassFailure();
      }

      // Steps 9-12: Process each pipeline independently (innermost
      // first).
      llvm::SmallVector<mlir::ktdf::PipelineOp, 8> pipelines;
      func.walk([&](mlir::ktdf::PipelineOp pipeline) {
        pipelines.push_back(pipeline);
      });

      if (!pipelines.empty()) {
        LDBG(1) << "  Processing " << pipelines.size() << " pipelines";

        // Process pipelines in reverse order (post-order walk for nested
        // pipelines)
        for (auto pipeline : llvm::reverse(pipelines)) {
          LDBG(1) << "  Processing pipeline at " << pipeline.getLoc() << "";

          // Collect stages that are direct children of this pipeline (not
          // nested)
          llvm::SmallVector<mlir::ktdf::StageOp, 8> pipeline_stages;
          for (auto& op : pipeline.getBodyRegion().front()) {
            if (auto stage = mlir::dyn_cast<mlir::ktdf::StageOp>(op)) {
              pipeline_stages.push_back(stage);
            }
          }

          if (pipeline_stages.empty()) {
            // Skip pipelines that have no direct stages (already transformed or
            // empty)
            LDBG(1) << "  Skipping pipeline with no direct stages";
            continue;
          }

          // Step 9: Analyze per-pipeline stage dependencies for topo-sort.
          mlir::ktdf::StageDependencyDAG dag;
          if (mlir::failed(
                  mlir::ktdf::analyzeStageDependencies(pipeline_stages, dag))) {
            return signalPassFailure();
          }

          // Step 10: Topologically sort stages
          llvm::SmallVector<mlir::ktdf::StageOp, 8> sorted_stages;
          if (mlir::failed(mlir::ktdf::topologicalSortStages(
                  pipeline_stages, dag, sorted_stages))) {
            pipeline.emitError("topological sort of stages failed");
            return signalPassFailure();
          }

          // Step 11: Transform stages to execute_on (inside-out: stages first)
          if (mlir::failed(transformStagesToExecuteOn(pipeline, sorted_stages,
                                                      stage_to_units))) {
            return signalPassFailure();
          }

          // Step 12: Transform pipeline to execute_on (wrapping everything)
          if (mlir::failed(transformPipelineToExecuteOn(pipeline, sorted_stages,
                                                        stage_to_units))) {
            return signalPassFailure();
          }

          // erasing the stages at the very end.
          for (auto& stage : sorted_stages) {
            stage.erase();
          }
        }

        LDBG(1) << "  Phase 2 complete";
      }

      // Remove loop_type attributes from all scf.for loops now that
      // lowering is complete and the attribute is no longer needed.
      func.walk(
          [](mlir::scf::ForOp for_op) { for_op->removeAttr("loop_type"); });

      LDBG(1) << "Lowering complete for " << func.getName() << "";
    }
  }

 private:
  const SchedulerExtContext& schedulerExtContext() const {
    return scheduler_ctx_;
  }

  const SchedulerExtContext& scheduler_ctx_;
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createKTDFToKTDFLoweringPass() {
  return std::make_unique<KTDFToKTDFLoweringPass>();
}

std::unique_ptr<mlir::Pass> scheduler::createKTDFToKTDFLoweringPass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<KTDFToKTDFLoweringPass>(scheduler_ctx);
}
