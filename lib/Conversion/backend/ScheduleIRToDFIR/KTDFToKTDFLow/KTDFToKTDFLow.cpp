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

#include "Ktdp/KtdpOps.hpp"
#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
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
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/Support/Debug.h"
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
const char VerboseDebug[] = DEBUG_TYPE "-verbose";

struct KTDFToKTDFLoweringPass
    : public impl::KTDFToKTDFLoweringPassBase<KTDFToKTDFLoweringPass> {
  KTDFToKTDFLoweringPass()
      : scheduler_ctx_(SchedulerExtContext::dummyContext()) {}

  KTDFToKTDFLoweringPass(const SchedulerExtContext& scheduler_ctx)
      : scheduler_ctx_(scheduler_ctx) {}

  void runOnOperation() override {
    DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs() << PASS_NAME " running\n");
    mlir::ModuleOp module = getOperation();

    auto& devices = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = devices.getOrImportDevice();
    if (!device) {
      LLVM_DEBUG(llvm::dbgs() << "No device found.\n");
      signalPassFailure();
      return;
    }
    auto& resource_kinds = getChildAnalysis<arch_view::ResourceKinds>(**device);

    llvm::SmallVector<mlir::func::FuncOp, 4> funcs;
    module.walk([&](mlir::func::FuncOp func) {
      funcs.push_back(func);
      return mlir::WalkResult::skip();
    });
    for (auto func : funcs) {
      LLVM_DEBUG(llvm::dbgs() << "Running " << PASS_NAME << " on "
                              << func.getName() << "\n");

      // Pre-compute stages for reuse across multiple steps
      llvm::SmallVector<mlir::ktdf::StageOp, 8> stages;
      mlir::ktdf::collectStages(func, stages);

      if (stages.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "  No stages found - skipped\n");
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

      // Step 5: Wire queried units to stages
      mlir::OpBuilder phase2_builder(func.getContext());

      // Wire queried units from Step 4 to stages based on applicable_units
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
        LLVM_DEBUG(llvm::dbgs() << "  No queries wired to stages\n");
        return signalPassFailure();
      }

      // Pre-compute global flat stage DAG for signal narrowing (Step 9).
      mlir::ktdf::StageDependencyDAG global_dag;
      if (mlir::failed(mlir::ktdf::buildGlobalStageDAG(func, global_dag))) {
        return signalPassFailure();
      }

      // Steps 6-11: Process each pipeline independently
      llvm::SmallVector<mlir::ktdf::PipelineOp, 8> pipelines;
      func.walk([&](mlir::ktdf::PipelineOp pipeline) {
        pipelines.push_back(pipeline);
      });

      if (!pipelines.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  Processing " << pipelines.size() << " pipelines\n");

        // Process pipelines in reverse order (post-order walk for nested
        // pipelines)
        for (auto pipeline : llvm::reverse(pipelines)) {
          LLVM_DEBUG(llvm::dbgs() << "  Processing pipeline at "
                                  << pipeline.getLoc() << "\n");

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
            LLVM_DEBUG(llvm::dbgs()
                       << "  Skipping pipeline with no direct stages\n");
            continue;
          }

          // Step 6: Analyze stage dependencies
          mlir::ktdf::StageDependencyDAG dag;
          if (mlir::failed(
                  mlir::ktdf::analyzeStageDependencies(pipeline_stages, dag))) {
            return signalPassFailure();
          }

          // Step 7: Compute scratchpad conflicts
          std::map<std::pair<mlir::Operation*, mlir::Operation*>,
                   llvm::SmallVector<scheduler::ResourceType, 2>>
              conflicts;
          LLVM_DEBUG(llvm::dbgs()
                     << "Step 7: Computing scratchpad conflicts\n");
          if (mlir::failed(computeScratchpadConflicts(
                  stage_to_units, dag, resource_kinds, conflicts))) {
            LLVM_DEBUG(llvm::dbgs() << "Step 7 failed\n");
            return signalPassFailure();
          }
          LLVM_DEBUG(llvm::dbgs() << "Step 7 complete: " << conflicts.size()
                                  << " conflicts found\n");

          // Step 8: Topologically sort stages
          llvm::SmallVector<mlir::ktdf::StageOp, 8> sorted_stages;
          if (mlir::failed(mlir::ktdf::topologicalSortStages(
                  pipeline_stages, dag, sorted_stages))) {
            return signalPassFailure();
          }

          // Step 9: Insert signal operations between stages (before
          // transformation)
          if (mlir::failed(insertSignalsInPipeline(pipeline, sorted_stages,
                                                   stage_to_units, conflicts,
                                                   phase2_builder))) {
            return signalPassFailure();
          }

          // Step 10: Transform stages to execute_on (inside-out: stages first)
          if (mlir::failed(transformStagesToExecuteOn(
                  pipeline, sorted_stages, stage_to_units, phase2_builder))) {
            return signalPassFailure();
          }

          // Step 11: Transform pipeline to execute_on (wrapping everything)
          if (mlir::failed(transformPipelineToExecuteOn(
                  pipeline, sorted_stages, stage_to_units, phase2_builder))) {
            return signalPassFailure();
          }

          // erasing the stages at the very end.
          for (auto& stage : sorted_stages) {
            stage.erase();
          }
        }

        LLVM_DEBUG(llvm::dbgs() << "  Phase 2 complete\n");
      }

      // Remove loop_type attributes from all scf.for loops now that
      // lowering is complete and the attribute is no longer needed.
      func.walk(
          [](mlir::scf::ForOp for_op) { for_op->removeAttr("loop_type"); });

      LLVM_DEBUG(llvm::dbgs()
                 << "Lowering complete for " << func.getName() << "\n");
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
