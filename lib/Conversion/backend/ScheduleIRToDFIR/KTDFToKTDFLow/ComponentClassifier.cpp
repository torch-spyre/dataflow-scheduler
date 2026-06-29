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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/ComponentClassifier.h"

#include "dataflow-scheduler/Dialect/KTDF/Analysis/Utils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ktdf-to-operand-lowering"

using namespace scheduler;

mlir::LogicalResult ComponentClassifier::classify(
    const llvm::SmallVector<mlir::ktdf::StageOp, 8>& stages,
    ComponentClassification& result) {
  LLVM_DEBUG(llvm::dbgs() << "Step 1: Classify components\n");

  llvm::SetVector<ResourceType> all_parallel_components;
  llvm::SetVector<ResourceType> temp_non_parallel_components;

  // Process provided stages and collect components
  for (mlir::ktdf::StageOp stage : stages) {
    auto units_attr = stage.getApplicableUnitsAttr();
    assert(units_attr && "Stage should be annotated with applicable units");

    // Check once per stage if it's inside any ktdf.parallel
    mlir::Operation* parallel_parent = mlir::ktdf::findParallelParent(stage);
    bool in_parallel = (parallel_parent != nullptr);

    // Process all components for this stage with cached parallel info
    for (mlir::Attribute component : units_attr.getValue()) {
      if (in_parallel) {
        // Record component in this parallel operation
        assert(parallel_parent && "Parallel components must exist");
        auto& comps = result.parallel_components_map[parallel_parent];
        comps.insert(component);
        all_parallel_components.insert(component);

        auto str_attr = mlir::dyn_cast<mlir::StringAttr>(component);
        LLVM_DEBUG(llvm::dbgs() << "  Component "
                                << (str_attr ? str_attr.getValue() : "unknown")
                                << " is parallel\n");
      } else {
        // Candidate for non-parallel (may be overridden if also in parallel)
        temp_non_parallel_components.insert(component);
      }
    }
  }

  // Final: non-parallel components = candidates - overlaps
  // If a component appears in any parallel region,
  // exclude it from non_parallel_components.
  for (auto comp : temp_non_parallel_components) {
    if (!all_parallel_components.contains(comp)) {
      result.non_parallel_components.insert(comp);

      auto str_attr = mlir::dyn_cast<mlir::StringAttr>(comp);
      LLVM_DEBUG(llvm::dbgs() << "  Component "
                              << (str_attr ? str_attr.getValue() : "unknown")
                              << " is non-parallel\n");
    } else {
      auto str_attr = mlir::dyn_cast<mlir::StringAttr>(comp);
      LLVM_DEBUG(
          llvm::dbgs()
          << "  Component " << (str_attr ? str_attr.getValue() : "unknown")
          << " excluded from non-parallel (appears in parallel region)\n");
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Component classification complete:\n";
    llvm::dbgs() << "  Non-parallel: " << result.non_parallel_components.size()
                 << "\n";
    llvm::dbgs() << "  Parallel regions: "
                 << result.parallel_components_map.size() << "\n";
  });

  return mlir::success();
}
