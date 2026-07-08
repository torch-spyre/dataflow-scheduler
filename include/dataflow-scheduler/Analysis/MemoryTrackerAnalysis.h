//===-- MemoryTrackerAnalysis.h ---------------------------------*- c++ -*-===//
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
// Memory Tracker Analysis
//
// This file defines a stateful MLIR analysis that tracks memory allocations
// across the pass pipeline. Unlike typical analyses that derive read-only
// information from IR, this analysis maintains mutable state that accumulates
// as passes allocate memory.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_MEMORYTRACKERANALYSIS_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_MEMORYTRACKERANALYSIS_H_

#include <cstddef>

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "dataflow-scheduler/Analysis/MemoryTracker.h"
#include "llvm/Support/Error.h"
#include "mlir/Pass/AnalysisManager.h"

namespace scheduler {

/// Stateful analysis that tracks memory allocations across the pass pipeline.
///
/// This analysis maintains a MemoryTracker instance initialized from the
/// device's memory hierarchy (MemoryTree). It is preserved throughout the
/// pipeline and accumulates allocation state as passes request memory.
///
/// Key Characteristics:
/// - Lifetime: Created once per module, persists across all passes
/// - Initialization: Queries DeviceManager to build MemoryTree
/// - Preservation: Always preserved (never invalidated)
/// - Thread Safety: Not thread-safe; assumes sequential pass execution
///
/// Assumptions:
/// 1. ktdf_arch::DeviceOp is immutable throughout the pipeline
/// 2. Memory allocations are monotonic (only grow, never shrink)
/// 3. Passes are executed in a controlled order
/// 4. No concurrent pass execution on the same module
///
/// Usage:
///   auto& tracker = getAnalysis<MemoryTrackerAnalysis>();
///   auto addr = tracker.allocate(resource, size, alignment);
class MemoryTrackerAnalysis {
 public:
  /// Construct from operation and analysis manager, initializing from
  /// DeviceManager
  explicit MemoryTrackerAnalysis(mlir::Operation* op,
                                 mlir::AnalysisManager& am);

  /// Analysis is never invalidated (DeviceOp is immutable)
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses& /*pa*/) {
    return false;
  }

  /// Allocate memory and return the assigned address
  /// @param memory_resource The memory resource attribute (e.g., L1)
  /// @param size_in_bytes Size of the allocation in bytes
  /// @param alignment Byte alignment requirement (default: 1)
  /// @return The allocated address on success, or an error if allocation fails
  llvm::Expected<size_t> allocate(ResourceType memory_resource,
                                  size_t size_in_bytes, size_t alignment = 1);

  /// Get the current next available address for a memory resource
  size_t getNextAvailableAddress(ResourceType memory_resource) const;

  /// Get the total allocated size for a memory resource
  size_t getTotalAllocated(ResourceType memory_resource) const;

  /// Get the underlying MemoryTracker (for advanced use cases)
  MemoryTracker& getTracker() { return tracker_; }
  const MemoryTracker& getTracker() const { return tracker_; }

  /// Get the memory tree used for initialization
  const arch_view::MemoryTree& getMemoryTree() const { return memory_tree_; }

  /// Print current allocation state for debugging
  void dump() const;
  void print(llvm::raw_ostream& os) const;

 private:
  const arch_view::MemoryTree& memory_tree_;
  MemoryTracker tracker_;
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_MEMORYTRACKERANALYSIS_H_

// Made with Bob