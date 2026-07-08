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

#define DEBUG_TYPE "signal-insertion"

using namespace scheduler;

// Returns the StageOp that is the correct insertion anchor for a SignalOp
// between leaf stages producer_op and consumer_op. The anchor is the StageOp
// in the shared parent PipelineOp that is closest (innermost) to both leaves —
// i.e., the direct-child stage of the lowest common ancestor PipelineOp on
// the producer's side.
static mlir::Operation* sharedLevelAncestor(mlir::Operation* producer_op,
                                            mlir::Operation* consumer_op) {
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

mlir::LogicalResult scheduler::insertSignals(
    mlir::Location loc, const StageToUnitsMap& stage_to_units,
    const mlir::ktdf::StageDependencyDAG& global_dag,
    const std::map<std::pair<mlir::Operation*, mlir::Operation*>,
                   llvm::SmallVector<scheduler::ResourceType, 2>>& conflicts,
    mlir::OpBuilder& builder) {
  LDBG(1) << "Step 9: Insert signal operations";

  for (const auto& [producer_op, successors] : global_dag.successors) {
    for (auto* consumer_op : successors) {
      if (!conflicts.count({producer_op, consumer_op})) continue;

      LDBG(1) << "  Inserting signal between leaf stages";

      llvm::SmallVector<mlir::Value, 8> signal_units;
      auto collect = [&](mlir::Operation* op) {
        auto it = stage_to_units.mapping.find(op);
        if (it != stage_to_units.mapping.end())
          for (auto unit : it->second) signal_units.push_back(unit);
      };
      collect(producer_op);
      collect(consumer_op);

      if (!signal_units.empty()) {
        mlir::Operation* anchor = sharedLevelAncestor(producer_op, consumer_op);
        builder.setInsertionPointAfter(anchor);
        mlir::ktdf_lowering::SignalOp::create(builder, loc,
                                              mlir::ValueRange(signal_units));
      }
    }
  }

  LDBG(1) << "  Signal insertion complete";
  return mlir::success();
}
