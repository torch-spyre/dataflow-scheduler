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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/ScratchpadConflicts.h"

#include <llvm/ADT/DenseSet.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>

#include <set>

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Links.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "phase2-analysis"

using namespace scheduler;

namespace {

void getResourceAccesses(mlir::ktdf_arch::Resource resource,
                         llvm::SmallDenseSet<mlir::Attribute>& read_kinds,
                         llvm::SmallDenseSet<mlir::Attribute>& write_kinds) {
  const auto visit = [](mlir::Value value,
                        llvm::SmallDenseSet<mlir::Attribute>& result) {
    auto resource = mlir::ktdf_arch::getResource(value);
    if (!resource || !resource.getKind()) {
      return;
    }

    if (!llvm::isa<mlir::ktdf_arch::MemoryOp>(resource)) {
      return;
    }

    result.insert(resource.getKind());
  };

  mlir::ktdf_arch::visitLinks(
      resource,
      [&](mlir::ktdf_arch::Link link,
          mlir::ktdf_arch::LinkDirection direction) -> bool {
        if (!mlir::ktdf_arch::isIncoming(direction)) {
          return true;
        }
        for (auto source : link.getSources()) {
          visit(source, read_kinds);
        }
        return true;
      });

  mlir::ktdf_arch::visitLinks(
      resource,
      [&](mlir::ktdf_arch::Link link,
          mlir::ktdf_arch::LinkDirection direction) -> bool {
        if (!mlir::ktdf_arch::isOutgoing(direction)) {
          return true;
        }
        for (auto source : link.getTargets()) {
          visit(source, write_kinds);
        }
        return true;
      });
}

}  // namespace

mlir::LogicalResult scheduler::computeScratchpadConflicts(
    const StageToUnitsMap& stage_to_units,
    const mlir::ktdf::StageDependencyDAG& dag,
    const arch_view::ResourceKinds& resource_kinds,
    std::map<std::pair<mlir::Operation*, mlir::Operation*>,
             llvm::SmallVector<scheduler::ResourceType, 2>>& conflicts) {
  LDBG(1) << "Step 3: Compute scratchpad conflicts";

  for (const auto& [producer_op, successors] : dag.successors) {
    assert(llvm::isa<mlir::ktdf::StageOp>(producer_op) &&
           "producer op has to be a stage operation");

    auto producer_it = stage_to_units.mapping.find(producer_op);
    assert(producer_it != stage_to_units.mapping.end() &&
           "producer op should be found in stage_to_units mapping");

    const auto& producer_units = producer_it->second;

    for (auto consumer_op : successors) {
      assert(llvm::isa<mlir::ktdf::StageOp>(consumer_op) &&
             "consumer op has to be a stage operation");

      auto consumer_it = stage_to_units.mapping.find(consumer_op);
      assert(consumer_it != stage_to_units.mapping.end() &&
             "consumer op should be found in stage_to_units mapping");

      const auto& consumer_units = consumer_it->second;

      llvm::SmallVector<scheduler::ResourceType, 2> conflicting_units;

      for (auto producer_unit_val : producer_units) {
        auto producer_comp_opt =
            scheduler::getUnitResourceType(producer_unit_val);
        if (!producer_comp_opt.has_value()) continue;
        scheduler::ResourceType producer_comp = producer_comp_opt.value();
        auto producer = resource_kinds.getResource(producer_comp);
        if (!producer) {
          return llvm::failure();
        }

        llvm::SmallDenseSet<mlir::Attribute> producer_writes;
        llvm::SmallDenseSet<mlir::Attribute> producer_reads_unused;
        getResourceAccesses(producer, producer_reads_unused, producer_writes);

        for (auto consumer_unit_val : consumer_units) {
          auto consumer_comp_opt =
              scheduler::getUnitResourceType(consumer_unit_val);
          assert(consumer_comp_opt.has_value() &&
                 "Consumer unit should have a component");

          scheduler::ResourceType consumer_comp = consumer_comp_opt.value();
          auto consumer = resource_kinds.getResource(consumer_comp);
          if (!consumer) {
            return llvm::failure();
          }

          llvm::SmallDenseSet<mlir::Attribute> consumer_reads;
          llvm::SmallDenseSet<mlir::Attribute> consumer_writes_unused;
          getResourceAccesses(consumer, consumer_reads, consumer_writes_unused);

          bool has_conflict = false;
          for (const auto& written : producer_writes) {
            if (consumer_reads.contains(written)) {
              LDBG(1) << "  Found conflict between " << producer_comp << " and "
                      << consumer_comp << " in " << written;
              has_conflict = true;
              break;
            }
          }

          if (has_conflict) {
            conflicting_units.push_back(producer_comp);
            conflicting_units.push_back(consumer_comp);
          }
        }
      }

      if (!conflicting_units.empty()) {
        conflicts[{producer_op, consumer_op}] = conflicting_units;
        LDBG(1) << "  Conflict between stages: " << producer_op << " -> "
                << consumer_op;
      }
    }
  }

  return mlir::success();
}
