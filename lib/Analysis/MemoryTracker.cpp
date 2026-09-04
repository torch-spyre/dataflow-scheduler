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

#include "dataflow-scheduler/Analysis/MemoryTracker.h"

#include "dataflow-scheduler/Analysis/ArchViews/MemoryTree.h"
#include "llvm/Support/ErrorHandling.h"

using namespace scheduler;

MemoryTracker::MemoryTracker(const arch_view::MemoryTree& memory_tree) {
  // Extract capacities from the memory tree
  for (auto node_id : memory_tree.getAllNodeIds()) {
    auto node = memory_tree.getNode(node_id);
    if (node && node->capacity_in_bytes) {
      // Use available capacity (capacity - reserved)
      size_t available = *node->capacity_in_bytes;
      if (node->reserved_in_bytes) {
        available -= *node->reserved_in_bytes;
      }
      capacities_[node->memory_resource] = available;
      next_address_[node->memory_resource] = kFirstAvailableAddress;
    }
  }
}

size_t MemoryTracker::alignAddress(size_t address, size_t alignment) {
  if (alignment <= 1) return address;

  // Round up to next multiple of alignment
  size_t remainder = address % alignment;
  if (remainder == 0) return address;

  return address + (alignment - remainder);
}

llvm::Expected<size_t> MemoryTracker::allocate(ResourceType memory_resource,
                                               size_t size_in_bytes,
                                               size_t alignment) {
  assert(next_address_.find(memory_resource) != next_address_.end() &&
         "Memory resource not found in capacity map");
  // Get current address for this memory resource
  size_t current_address = next_address_[memory_resource];

  // Align the address
  size_t aligned_address = alignAddress(current_address, alignment);

  // Calculate the new next address after this allocation
  size_t new_next_address = aligned_address + size_in_bytes;

  // Query the available capacity
  auto capacity_it = capacities_.find(memory_resource);
  if (capacity_it == capacities_.end()) {
    std::string resource_str;
    llvm::raw_string_ostream ss(resource_str);
    if (memory_resource) {
      memory_resource.print(ss);
    } else {
      ss << "<null>";
    }
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Memory resource %s does not have capacity information",
        ss.str().c_str());
  }

  size_t available_capacity = capacity_it->second;

  // Check if allocation fits
  if (new_next_address > available_capacity) {
    std::string resource_str;
    llvm::raw_string_ostream ss(resource_str);
    if (memory_resource) {
      memory_resource.print(ss);
    } else {
      ss << "<null>";
    }
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Allocation of %zu bytes (aligned to %zu) exceeds %s capacity "
        "(%zu bytes available, %zu bytes already allocated)",
        size_in_bytes, alignment, ss.str().c_str(), available_capacity,
        current_address);
  }

  // Update the next address
  next_address_[memory_resource] = new_next_address;

  return aligned_address;
}

size_t MemoryTracker::getNextAvailableAddress(
    ResourceType memory_resource) const {
  auto it = next_address_.find(memory_resource);
  if (it == next_address_.end()) {
    return kFirstAvailableAddress;
  }
  return it->second;
}

size_t MemoryTracker::getTotalAllocated(ResourceType memory_resource) const {
  // Total allocated is the current next address minus the first address
  return getNextAvailableAddress(memory_resource) - kFirstAvailableAddress;
}

void MemoryTracker::reset(llvm::ArrayRef<ResourceType> memory_resources) {
  for (ResourceType resource : memory_resources) {
    auto it = next_address_.find(resource);
    if (it != next_address_.end()) it->second = kFirstAvailableAddress;
  }
}

// Made with Bob
