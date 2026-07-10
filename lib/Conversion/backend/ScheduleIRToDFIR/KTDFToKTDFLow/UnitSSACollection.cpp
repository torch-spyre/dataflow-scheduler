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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UnitSSACollection.h"

#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "phase2-analysis"

using namespace scheduler;
using ResourceType = mlir::Attribute;

mlir::LogicalResult collectUnitSSAValues(mlir::func::FuncOp func,
                                         StageToUnitsMap& stage_to_units,
                                         mlir::OpBuilder& builder) {
  LDBG(1) << "Step 1: Collect unit SSA values";

  llvm::SmallVector<mlir::ktdf::StageOp, 8> stages;
  mlir::ktdf::collectStages(func, stages);

  if (stages.empty()) {
    return func.emitError("No stages found in pipeline");
  }

  LDBG(1) << "Found " << stages.size() << " stages";

  llvm::SmallVector<mlir::Operation*, 8> query_ops;
  mlir::ktdf::collectQueriedUnits(func, query_ops);

  if (query_ops.empty()) {
    return func.emitError(
        "No unit queries found - Phase 1.5 may be incomplete");
  }

  LDBG(1) << "Found " << query_ops.size() << " query_map ops";

  llvm::DenseMap<ResourceType, llvm::SmallVector<mlir::Operation*, 4>>
      component_to_queries;

  for (auto query_op : query_ops) {
    auto query_map = mlir::dyn_cast<mlir::uniform::QueryMapOp>(query_op);
    if (!query_map) {
      LDBG(1) << "  Query is not QueryMapOp";
      continue;
    }

    // Use the utility function to extract unit type
    auto type_str = scheduler::getUnitTypeFromQueryMap(query_map.getResult());
    assert(!type_str.empty() && "getUnitTypeFromQueryMap failed");

    ResourceType component =
        mlir::StringAttr::get(builder.getContext(), type_str);
    component_to_queries[component].push_back(query_op);

    LDBG(1) << "  Mapped query to component " << type_str
            << " (total: " << component_to_queries[component].size() << ")";
  }

  LDBG(1) << "  Component to queries map has " << component_to_queries.size()
          << " entries";

  int wired_queries = 0;
  for (auto stage : stages) {
    auto applicable_units = stage.getApplicableUnitsAttr();
    if (!applicable_units) continue;

    for (auto component : applicable_units) {
      if (auto queries = component_to_queries.find(component);
          queries != component_to_queries.end()) {
        for (auto query_op : queries->second) {
          stage_to_units.mapping[stage.getOperation()].push_back(
              query_op->getResult(0));
          wired_queries++;
          LDBG(1) << "  Wired query to stage for component";
        }
      }
    }
  }

  LDBG(1) << "  Wired " << wired_queries << " query-stage pairs";

  if (wired_queries == 0) {
    LDBG(1) << "  No queries wired to stages - Phase 2 cannot proceed";
    return mlir::failure();
  }

  LDBG(1) << "Step 1 complete: collected " << stage_to_units.mapping.size()
          << " stages with units";

  return mlir::success();
}
