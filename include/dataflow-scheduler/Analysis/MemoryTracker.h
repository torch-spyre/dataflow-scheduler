//===------------------------------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_MEMORYTRACKER_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_MEMORYTRACKER_H_

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Error.h>

#include <cstddef>

#include "mlir/IR/Attributes.h"

namespace scheduler {

// Forward declaration
namespace arch_view {
class MemoryTree;
}

// Type alias for resource representation
using ResourceType = mlir::Attribute;

/// @brief MemoryTracker manages buffer allocations for different memory spaces.
/// It tracks the next available address for each memory space and validates
/// that allocations fit within the available capacity.
///
/// The tracker is initialized from a MemoryTree view which provides capacity
/// information for each memory resource.
class MemoryTracker {
 public:
  /// First available address for allocations (starts at 0)
  static constexpr size_t kFirstAvailableAddress = 0;

  /// @brief Construct a MemoryTracker from a MemoryTree
  /// @param memory_tree The memory hierarchy view containing capacity info
  explicit MemoryTracker(const arch_view::MemoryTree& memory_tree);

  /// @brief Allocate memory and return the assigned address
  /// @param memory_resource The memory resource attribute (e.g., L1)
  /// @param size_in_bytes Size of the allocation in bytes
  /// @param alignment Byte alignment requirement (default: 1)
  /// @return The allocated address on success, or an error if allocation fails
  llvm::Expected<size_t> allocate(ResourceType memory_resource,
                                  size_t size_in_bytes, size_t alignment = 1);

  /// @brief Get the current next available address for a memory resource
  /// @param memory_resource The memory resource attribute to query
  /// @return The next available address, or 0 if not yet allocated
  size_t getNextAvailableAddress(ResourceType memory_resource) const;

  /// @brief Get the total allocated size for a memory resource
  /// @param memory_resource The memory resource attribute to query
  /// @return Total bytes allocated in this memory resource
  size_t getTotalAllocated(ResourceType memory_resource) const;

  /// @brief Puts every resource's next address back to the first one
  void reset();

 private:
  /// Memory resource capacities (available bytes per resource)
  llvm::DenseMap<ResourceType, size_t> capacities_;

  /// Track next available address per memory resource attribute
  llvm::DenseMap<ResourceType, size_t> next_address_;

  /// Helper to align an address to the specified alignment
  static size_t alignAddress(size_t address, size_t alignment);
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_MEMORYTRACKER_H_

// Made with Bob
