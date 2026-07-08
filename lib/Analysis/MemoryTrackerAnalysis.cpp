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

#include "dataflow-scheduler/Analysis/MemoryTrackerAnalysis.h"

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "memory-tracker-analysis"

using namespace scheduler;

MemoryTrackerAnalysis::MemoryTrackerAnalysis(mlir::Operation* /*op*/,
                                             mlir::AnalysisManager& am)
    : memory_tree_([&]() -> const arch_view::MemoryTree& {
        auto& device_manager = am.getAnalysis<mlir::ktdf_arch::DeviceManager>();
        auto* const device = device_manager.getOrImportDevice();
        if (!device) {
          llvm::report_fatal_error(
              "MemoryTrackerAnalysis: No ktdf_arch::DeviceOp found in module. "
              "This could happen if the device spec file is empty or contains "
              "multiple devices");
        }
        return am.getChildAnalysis<arch_view::MemoryTree>(
            device->getDeclaration());
      }()),
      tracker_(memory_tree_) {
  LLVM_DEBUG({
    llvm::dbgs() << "MemoryTrackerAnalysis initialized\n";
    if (!memory_tree_.getAllNodeIds().empty()) {
      llvm::dbgs() << "Memory tree:\n";
      memory_tree_.dump();
    }
  });
}

llvm::Expected<size_t> MemoryTrackerAnalysis::allocate(
    ResourceType memory_resource, size_t size_in_bytes, size_t alignment) {
  return tracker_.allocate(memory_resource, size_in_bytes, alignment);
}

size_t MemoryTrackerAnalysis::getNextAvailableAddress(
    ResourceType memory_resource) const {
  return tracker_.getNextAvailableAddress(memory_resource);
}

size_t MemoryTrackerAnalysis::getTotalAllocated(
    ResourceType memory_resource) const {
  return tracker_.getTotalAllocated(memory_resource);
}

void MemoryTrackerAnalysis::print(llvm::raw_ostream& os) const {
  os << "MemoryTrackerAnalysis State:\n";

  for (auto node_id : memory_tree_.getAllNodeIds()) {
    auto node = memory_tree_.getNode(node_id);
    if (!node) continue;

    os << "  Resource: ";
    node->memory_resource.print(os);

    size_t allocated = getTotalAllocated(node->memory_resource);
    os << "\n    Allocated: " << allocated << " bytes";

    if (node->capacity_in_bytes) {
      os << "\n    Capacity: " << *node->capacity_in_bytes << " bytes";
      size_t available = *node->capacity_in_bytes - allocated;
      if (node->reserved_in_bytes) {
        available -= *node->reserved_in_bytes;
      }
      os << "\n    Available: " << available << " bytes";
    }
    os << "\n";
  }
}

void MemoryTrackerAnalysis::dump() const { print(llvm::dbgs()); }

// Made with Bob