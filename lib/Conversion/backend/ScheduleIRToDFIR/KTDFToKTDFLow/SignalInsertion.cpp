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

#include "dataflow-scheduler/Dialect/KTDF/Analysis/GlobalStageDAG.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "phase2-signal-insertion"

using namespace scheduler;

mlir::LogicalResult scheduler::insertSignalsInPipeline(
    mlir::ktdf::PipelineOp pipeline,
    const llvm::SmallVector<mlir::ktdf::StageOp, 8>& sorted_stages,
    const StageToUnitsMap& stage_to_units,
    const std::map<std::pair<mlir::Operation*, mlir::Operation*>,
                   llvm::SmallVector<scheduler::ResourceType, 2>>& conflicts,
    mlir::OpBuilder& builder) {
  LDBG(1) << "Step 9: Insert signal operations";

  auto loc = pipeline.getLoc();

  for (size_t i = 0; i + 1 < sorted_stages.size(); ++i) {
    auto producer_stage = sorted_stages[i];
    auto consumer_stage = sorted_stages[i + 1];

    bool has_conflict =
        conflicts.count(std::make_pair(producer_stage.getOperation(),
                                       consumer_stage.getOperation())) > 0;
    if (!has_conflict) continue;

    LDBG(1) << "  Inserting signal between stages";
    builder.setInsertionPointAfter(producer_stage);

    // Narrow to immediate leaf/root stages across any nested pipeline boundary.
    auto signal_producers = mlir::ktdf::getLeafStages(producer_stage);
    auto signal_consumers = mlir::ktdf::getRootStages(consumer_stage);

    llvm::SmallVector<mlir::Value, 8> signal_units;
    auto collect_units = [&](mlir::ktdf::StageOp stage) {
      auto it = stage_to_units.mapping.find(stage.getOperation());
      if (it != stage_to_units.mapping.end()) {
        for (auto unit : it->second) {
          signal_units.push_back(unit);
        }
      }
    };

    for (auto s : signal_producers) collect_units(s);
    for (auto s : signal_consumers) collect_units(s);

    if (!signal_units.empty()) {
      mlir::ktdf_lowering::SignalOp::create(builder, loc,
                                            mlir::ValueRange(signal_units));
    }
  }

  LDBG(1) << "  Signal insertion complete";
  return mlir::success();
}
