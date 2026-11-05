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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/SignalInsertion.h"

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "signal-insertion"

using namespace scheduler;

// Returns the insertion anchor for a signal between leaf stages. For sibling
// pipelines the signal goes after the producer pipeline, where both pipelines'
// units can execute it. Otherwise the anchor is the producer-side StageOp in
// the innermost PipelineOp shared by both leaves.
static mlir::Operation* sharedLevelAncestor(mlir::Operation* producer_op,
                                            mlir::Operation* consumer_op) {
  for (mlir::Operation* p = producer_op->getParentOp(); p;
       p = p->getParentOp()) {
    if (!mlir::isa<mlir::ktdf::PipelineOp>(p)) continue;
    auto* sibling = p->getBlock()->findAncestorOpInBlock(*consumer_op);
    if (sibling && sibling != p && mlir::isa<mlir::ktdf::PipelineOp>(sibling)) {
      return p;
    }
  }

  // Collect all PipelineOp ancestors of consumer (including its own parent).
  llvm::DenseSet<mlir::Operation*> consumer_pipelines;
  assert(mlir::isa<mlir::ktdf::StageOp>(consumer_op));
  for (mlir::Operation* p = consumer_op->getParentOp(); p;
       p = p->getParentOp()) {
    if (mlir::isa<mlir::ktdf::PipelineOp>(p)) consumer_pipelines.insert(p);
  }

  // Walk up producer's ancestry. Track the last StageOp seen so we can return
  // the direct-child StageOp of the innermost shared PipelineOp.
  // We want the *innermost* shared pipeline, so we stop at the first hit.
  mlir::Operation* last_stage = producer_op;
  assert(mlir::isa<mlir::ktdf::StageOp>(producer_op));
  for (mlir::Operation* p = producer_op->getParentOp(); p;
       p = p->getParentOp()) {
    if (mlir::isa<mlir::ktdf::StageOp>(p)) last_stage = p;
    if (mlir::isa<mlir::ktdf::PipelineOp>(p) && consumer_pipelines.contains(p))
      return last_stage;
  }
  return producer_op;
}

// Build the combined unit list for a signal between two stages.
static llvm::SmallVector<mlir::Value, 8> collectSignalUnits(
    mlir::Operation* stage_a, mlir::Operation* stage_b,
    const StageToUnitsMap& stage_to_units) {
  llvm::SmallVector<mlir::Value, 8> units;
  auto collect = [&](mlir::Operation* op) {
    auto it = stage_to_units.mapping.find(op);
    if (it != stage_to_units.mapping.end())
      for (auto unit : it->second) units.push_back(unit);
  };
  collect(stage_a);
  collect(stage_b);
  return units;
}

mlir::LogicalResult scheduler::insertSignals(
    mlir::Location loc, const StageToUnitsMap& stage_to_units,
    const mlir::ktdf::StageDependencyDAG& global_dag,
    const std::map<std::pair<mlir::Operation*, mlir::Operation*>,
                   llvm::SmallVector<scheduler::ResourceType, 2>>& conflicts,
    llvm::ArrayRef<BackEdgeInfo> back_edges) {
  LDBG(1) << "Step 8: Insert signal operations";

  // Build a set of back-edge pairs so normal signal insertion can skip them —
  // they are handled separately below with loop-IV guards.
  llvm::DenseSet<std::pair<mlir::Operation*, mlir::Operation*>> back_edge_set;
  for (const auto& be : back_edges)
    back_edge_set.insert({be.store_stage, be.load_stage});

  // Normal (intra-iteration) edges: unconditional signal after producer.
  for (const auto& [producer_op, successors] : global_dag.successors) {
    for (auto* consumer_op : successors) {
      if (!conflicts.count({producer_op, consumer_op})) continue;
      if (back_edge_set.count({producer_op, consumer_op})) continue;

      LDBG(1) << "  Inserting signal between leaf stages";

      auto signal_units =
          collectSignalUnits(producer_op, consumer_op, stage_to_units);
      if (!signal_units.empty()) {
        mlir::Operation* anchor = sharedLevelAncestor(producer_op, consumer_op);
        mlir::OpBuilder builder(anchor);
        builder.setInsertionPointAfter(anchor);
        mlir::ktdf_lowering::SignalOp::create(builder, loc,
                                              mlir::ValueRange(signal_units));
      }
    }
  }

  // Cross-iteration back-edges: two guarded signals.
  for (const auto& be : back_edges) {
    if (!conflicts.count({be.store_stage, be.load_stage})) continue;

    auto signal_units =
        collectSignalUnits(be.store_stage, be.load_stage, stage_to_units);
    if (signal_units.empty()) continue;

    LDBG(1) << "  Inserting guarded back-edge signals (store→load)";

    // --- Signal at start of load_stage: recv guard ---
    // Condition: all dependent loop IVs are not on their first iteration
    //   i.e.  iv0 != lb0 && iv1 != lb1 && ...
    {
      auto load_stage = mlir::cast<mlir::ktdf::StageOp>(be.load_stage);
      mlir::Block* load_body = load_stage.getBody();
      mlir::OpBuilder builder(load_body, load_body->begin());

      mlir::Value cond;
      for (mlir::scf::ForOp loop : be.dependent_loops) {
        mlir::Value not_first = mlir::arith::CmpIOp::create(
            builder, loc, mlir::arith::CmpIPredicate::ne,
            loop.getInductionVar(), loop.getLowerBound());
        cond = cond ? mlir::arith::OrIOp::create(builder, loc, cond, not_first)
                          .getResult()
                    : not_first;
      }
      auto if_op = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                           cond, /*withElseRegion=*/false);
      mlir::OpBuilder then_builder(
          if_op.getThenRegion().front().getTerminator());
      mlir::ktdf_lowering::SignalOp::create(then_builder, loc,
                                            mlir::ValueRange(signal_units));
    }

    // --- Signal at end of store_stage: send guard ---
    // Condition: all dependent loop IVs are not on their last iteration
    //   i.e.  iv0 != ub0-step0 && iv1 != ub1-step1 && ...
    {
      auto store_stage = mlir::cast<mlir::ktdf::StageOp>(be.store_stage);
      mlir::Block* store_body = store_stage.getBody();
      mlir::OpBuilder builder(store_body, store_body->end());

      mlir::Value cond;
      for (mlir::scf::ForOp loop : be.dependent_loops) {
        mlir::Value last_iv = mlir::arith::SubIOp::create(
            builder, loc, loop.getUpperBound(), loop.getStep());
        mlir::Value not_last = mlir::arith::CmpIOp::create(
            builder, loc, mlir::arith::CmpIPredicate::ne,
            loop.getInductionVar(), last_iv);
        cond = cond ? mlir::arith::OrIOp::create(builder, loc, cond, not_last)
                          .getResult()
                    : not_last;
      }
      auto if_op = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                           cond, /*withElseRegion=*/false);
      mlir::OpBuilder then_builder(
          if_op.getThenRegion().front().getTerminator());
      mlir::ktdf_lowering::SignalOp::create(then_builder, loc,
                                            mlir::ValueRange(signal_units));
    }
  }

  LDBG(1) << "  Signal insertion complete";
  return mlir::success();
}
