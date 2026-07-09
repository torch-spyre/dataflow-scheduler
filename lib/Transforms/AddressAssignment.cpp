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
//
// AddressAssignment: Analyze memory allocations and compute sizes for
// address assignment.
//
//===----------------------------------------------------------------------===//

#include <memory>
#include <optional>

#include "dataflow-scheduler/Analysis/MemoryTrackerAnalysis.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "address-assignment"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableAddressAssignmentPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Address Assignment pass"), llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_ADDRESSASSIGNMENTPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Compute the upper bound for a dynamic dimension.
/// Returns -1 if the bound cannot be computed.
int64_t computeDynamicDimensionBound(mlir::Value dynamic_size, size_t dim_idx) {
  LLVM_DEBUG({
    llvm::dbgs() << "  Dynamic dim " << dim_idx << ": SSA value "
                 << dynamic_size << "\n";
  });

  llvm::FailureOr<int64_t> upper_bound =
      mlir::ValueBoundsConstraintSet::computeConstantBound(
          mlir::presburger::BoundType::UB, dynamic_size,
          /*stopCondition=*/nullptr, /*closedUB=*/true);

  if (mlir::succeeded(upper_bound)) {
    LLVM_DEBUG(llvm::dbgs()
               << "    Computed upper bound: " << *upper_bound << "\n");
    return *upper_bound;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "    Failed to compute constant upper bound\n";
    llvm::dbgs() << "    Note: ValueBoundsConstraintSet could not "
                 << "resolve the bound to a constant value.\n";
    llvm::dbgs() << "    This may require more sophisticated "
                 << "analysis or manual annotation.\n";
  });
  return -1;
}

/// Compute total number of size of a memref.alloc in terms of the number of
/// elements. Returns -1 if any dynamic dimension cannot be resolved.
int64_t computeTotalElements(mlir::memref::AllocOp alloc) {
  auto shape = alloc.getType().getShape();
  int64_t total_elements = 1;
  unsigned dynamic_dim_idx = 0;

  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == mlir::ShapedType::kDynamic) {
      mlir::Value dynamic_size = alloc.getDynamicSizes()[dynamic_dim_idx++];
      int64_t bound = computeDynamicDimensionBound(dynamic_size, i);
      if (bound < 0) {
        return -1;
      }
      total_elements *= bound;
    } else {
      total_elements *= shape[i];
    }
  }

  return total_elements;
}

/// Compute the total allocation size in bytes for a memref.alloc operation.
/// For dynamic dimensions, attempts to compute upper bounds using
/// ValueBoundsConstraintSet. Returns -1 if any dimension cannot be resolved.
int64_t computeAllocationSize(mlir::memref::AllocOp alloc) {
  auto memref_type = alloc.getType();

  LLVM_DEBUG({
    llvm::dbgs() << "[" << PASS_NAME << "] Analyzing allocation: " << alloc
                 << "\n";
    llvm::dbgs() << "  Type: " << memref_type << "\n";
  });

  int64_t total_elements = computeTotalElements(alloc);
  if (total_elements < 0) {
    return -1;
  }

  int64_t element_size_bytes =
      getElementSizeBytes(memref_type.getElementType());
  int64_t total_bytes = total_elements * element_size_bytes;

  LLVM_DEBUG(llvm::dbgs() << "  Total size: " << total_bytes << " bytes ("
                          << total_elements << " elements * "
                          << element_size_bytes << " bytes/element)\n");

  return total_bytes;
}

/// Collect all memref.alloc operations that have a memory space attribute.
llvm::SmallVector<mlir::memref::AllocOp> collectAllocations(
    mlir::ModuleOp module) {
  llvm::SmallVector<mlir::memref::AllocOp> allocs;

  module.walk([&](mlir::memref::AllocOp alloc) {
    auto memref_type = alloc.getType();
    if (memref_type.getMemorySpace()) {
      allocs.push_back(alloc);
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skipping allocation with missing memory space: " << alloc
                 << "\n");
    }
    return mlir::WalkResult::advance();
  });

  return allocs;
}

/// Log allocation statistics.
void logAllocationStats(
    const llvm::SmallVector<mlir::memref::AllocOp>& allocs) {
  LLVM_DEBUG({
    llvm::dbgs() << "[" << PASS_NAME << "] Found " << allocs.size()
                 << " allocations with memory space attributes\n";
  });
}

/// Result of processing a single allocation.
struct AllocationResult {
  bool success;
  size_t assigned_address;
  int64_t size;
};

/// Replace memref.alloc with unrealized_conversion_cast using the assigned
/// address, and remove any corresponding memref.dealloc operations.
void materializeAddressAssignment(mlir::memref::AllocOp alloc,
                                  size_t assigned_address,
                                  mlir::OpBuilder& builder) {
  builder.setInsertionPoint(alloc);

  auto alloc_type = alloc.getType();

  // Create constant index for the assigned address
  mlir::Value addr_value = mlir::arith::ConstantIndexOp::create(
      builder, alloc.getLoc(), assigned_address);

  // Create unrealized_conversion_cast from index to memref
  auto cast = mlir::UnrealizedConversionCastOp::create(builder, alloc.getLoc(),
                                                       alloc_type, addr_value);

  LLVM_DEBUG({
    llvm::dbgs() << "  Materialized address assignment:\n";
    llvm::dbgs() << "    Original alloc: " << alloc << "\n";
    llvm::dbgs() << "    Replaced with: " << cast << "\n";
  });

  // Collect all dealloc operations that use this allocation
  llvm::SmallVector<mlir::memref::DeallocOp> deallocs_to_remove;
  for (mlir::Operation* user : alloc->getUsers()) {
    if (auto dealloc = mlir::dyn_cast<mlir::memref::DeallocOp>(user)) {
      deallocs_to_remove.push_back(dealloc);
    }
  }

  // Replace all uses of the alloc with the cast
  alloc.replaceAllUsesWith(cast.getResult(0));

  // Remove the dealloc operations
  for (auto dealloc : deallocs_to_remove) {
    LLVM_DEBUG(
        { llvm::dbgs() << "    Removing dealloc: " << dealloc << "\n"; });
    dealloc.erase();
  }

  // Erase the original alloc
  alloc.erase();
}

/// Process a single allocation and assign an address.
std::optional<AllocationResult> processAllocation(
    mlir::memref::AllocOp alloc, MemoryTrackerAnalysis& tracker,
    mlir::OpBuilder& builder) {
  int64_t size = computeAllocationSize(alloc);
  if (size < 0) {
    LLVM_DEBUG(llvm::dbgs()
               << "  Skipping allocation with unknown size: " << alloc << "\n");
    return std::nullopt;
  }

  // Get the memory space attribute from the memref type
  auto memref_type = alloc.getType();
  mlir::Attribute memory_space_attr = memref_type.getMemorySpace();
  assert(memory_space_attr);

  const size_t alignment =
      1;  // TODO: revisit in case we need to compute alignment
  auto address_result =
      tracker.allocate(memory_space_attr, static_cast<size_t>(size), alignment);

  if (!address_result) {
    llvm::Error err = address_result.takeError();
    std::string error_msg;
    llvm::handleAllErrors(std::move(err), [&](const llvm::ErrorInfoBase& ei) {
      error_msg = ei.message();
    });

    alloc.emitError() << "Failed to allocate " << memory_space_attr
                      << " memory: " << error_msg;
    return std::nullopt;
  }

  size_t assigned_address = *address_result;

  LLVM_DEBUG({
    llvm::dbgs() << "  Assigned address " << assigned_address
                 << " to allocation: " << alloc << "\n";
    llvm::dbgs() << "    Size: " << size << " bytes, Alignment: " << alignment
                 << " bytes\n";
  });

  // Materialize the address assignment in the IR
  materializeAddressAssignment(alloc, assigned_address, builder);

  return AllocationResult{true, assigned_address, size};
}

/// Process all allocations and assign addresses.
/// Returns the number of successful and failed assignments.
std::pair<int, int> processAllocations(
    const llvm::SmallVector<mlir::memref::AllocOp>& allocs,
    MemoryTrackerAnalysis& tracker, mlir::MLIRContext* context) {
  if (allocs.empty()) {
    return {0, 0};
  }

  LLVM_DEBUG(llvm::dbgs() << "\n[" << PASS_NAME
                          << "] Processing allocations with MemoryTracker:\n");

  mlir::OpBuilder builder(context);
  int successful_assignments = 0;
  int failed_assignments = 0;

  for (auto alloc : allocs) {
    auto result = processAllocation(alloc, tracker, builder);
    if (result) {
      successful_assignments++;
    } else {
      failed_assignments++;
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << "\n[" << PASS_NAME << "] Allocation summary:\n";
    llvm::dbgs() << "  Total allocations: " << allocs.size() << "\n";
    llvm::dbgs() << "  Successfully assigned: " << successful_assignments
                 << "\n";
    llvm::dbgs() << "  Failed assignments: " << failed_assignments << "\n";
    // TODO: Use memory hierarchy view to diagnose memref.alloc in namespaces
    // that are not supposed to be address assigned by the scheduler
  });

  return {successful_assignments, failed_assignments};
}

struct AddressAssignmentPass
    : public impl::AddressAssignmentPassBase<AddressAssignmentPass> {
  AddressAssignmentPass()
      : scheduler_ctx_(SchedulerExtContext::dummyContext()) {}
  explicit AddressAssignmentPass(const SchedulerExtContext& ctx)
      : scheduler_ctx_(ctx) {}

  void runOnOperation() override {
    if (DisableAddressAssignmentPass) return;

    mlir::ModuleOp module = getOperation();

    LLVM_DEBUG(llvm::dbgs() << "[" << PASS_NAME
                            << "] Starting address assignment analysis\n");

    // Collect all allocations with memory space attributes
    auto allocs = collectAllocations(module);
    logAllocationStats(allocs);

    // Get the MemoryTrackerAnalysis (automatically initialized from
    // DeviceManager)
    auto& tracker = getAnalysis<MemoryTrackerAnalysis>();

    // Process all allocations
    auto [successful, failed] =
        processAllocations(allocs, tracker, &getContext());

    if (failed > 0) {
      signalPassFailure();
    }
  }

 private:
  const SchedulerExtContext& scheduler_ctx_;
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createAddressAssignmentPass() {
  return std::make_unique<AddressAssignmentPass>();
}

std::unique_ptr<mlir::Pass> scheduler::createAddressAssignmentPass(
    const SchedulerExtContext& scheduler_ctx) {
  return std::make_unique<AddressAssignmentPass>(scheduler_ctx);
}

// Made with Bob
