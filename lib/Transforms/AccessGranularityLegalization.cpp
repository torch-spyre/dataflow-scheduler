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
// AccessGranularityLegalization: enforce hardware physical access_granularity
// constraints across all ktdf.data_transfer operations.
//
// Runs immediately after path-expansion and before splat-legalization, once
// all physical execution units and intermediate memory tiers are fully
// instantiated and concrete.
//
// Problem this pass addresses: hardware load/store units may only be able to
// issue transfers whose size is a multiple of some minimum bus granularity.
// For example, an MNILU can only read 64-element chunks from DDR, so a load
// transfer of [1, 1, 1, 1] must be widened to [1, 1, 1, 64] -- both source and
// destination sizes -- and the destination buffer resized to match. Similarly,
// store units must satisfy physical store granularity constraints.
//
// This pass's responsibilities are:
//   1. For load transfers (source is a memref): query the load unit's
//      access_granularity for the source memory space, compute the smallest
//      fitting legal transfer size, annotate
//      `dataflow_scheduler.access_granularity`, and widen innermost dimensions
//      and destination buffers/slots as needed.
//   2. For store transfers (destination is a memref): query the store unit's
//      access_granularity for the destination memory space, annotate
//      `dataflow_scheduler.access_granularity`, and verify the store satisfies
//      the hardware granularity without requiring write-widening (or emit a
//      diagnostic error).
//   3. Skips transfers with dynamic sizes, non-memref endpoints, and stages
//      with no unambiguous unit (no applicable_units).
//   4. Skips splat widening. A splat's source size indicates how many elements
//      the program requested, which splat-legalization needs to determine
//      hardware load overfetch. Splat-legalization handles widening splats
//      itself.
//
// Notes on scope:
//   - Only load (read) widening is handled here; write widening requires
//     hardware per-byte write-mask support and is diagnosed as an error.
//   - Source buffer bounds checking (verifying the source memref is large
//     enough to support the widened physical read) is a TODO.
//   - Integration with ZeroPadFront / ZeroPadBack for boundary tiles is a TODO.
//
//===----------------------------------------------------------------------===//

#include <functional>
#include <memory>
#include <numeric>
#include <optional>

#include "dataflow-scheduler/Analysis/Mapping.h"
#include "dataflow-scheduler/Analysis/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/DataTransfers.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/ResourceKinds.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define PASS_NAME "access-granularity-legalization"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> DisableThisPass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable the Access Granularity Legalization pass"),
    llvm::cl::init(false));

namespace scheduler {
#define GEN_PASS_DEF_ACCESSGRANULARITYLEGALIZATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Resolve the unit kind for a transfer that lives inside a ktdf.stage.
/// Returns nullptr when the stage has no unambiguous single applicable unit.
static scheduler::ResourceType resolveUnitKind(
    mlir::ktdf::DataTransferOp transfer) {
  auto stage = transfer->getParentOfType<mlir::ktdf::StageOp>();
  if (!stage) return nullptr;
  auto units = stage.getApplicableUnits();
  if (!units || units->size() != 1) return nullptr;
  return mlir::dyn_cast<scheduler::ResourceType>(units->getValue().front());
}

/// Compute the smallest legal transfer granularity in elements that covers
/// @p elements of @p elem_type from/to @p memory_space using @p feature .
template <typename FeatureT>
static int64_t computeGranularity(int64_t elements, mlir::Type elem_type,
                                  mlir::Attribute memory_space,
                                  FeatureT feature) {
  if (!feature || !memory_space) return elements;

  auto granularity_list = feature.getAccessGranularity(memory_space);
  if (!granularity_list) return elements;

  const size_t word_size = feature.getWordSize(memory_space);
  if (word_size == 0) return elements;

  const size_t elem_bytes =
      (static_cast<size_t>(elem_type.getIntOrFloatBitWidth()) + 7) / 8;
  if (elem_bytes == 0) return elements;

  // How many bus words are needed to cover all requested bytes?
  const size_t bytes_requested = static_cast<size_t>(elements) * elem_bytes;
  const size_t words_requested = (bytes_requested + word_size - 1) / word_size;

  auto best = granularity_list.fitAccess(words_requested);
  if (!best) return elements;

  const size_t granule_bytes = best.getSizeInWords() * word_size;
  const int64_t granule_elements =
      static_cast<int64_t>(granule_bytes / elem_bytes);
  return std::max(granule_elements, elements);
}

/// Widen the innermost source and destination sizes of @p transfer from
/// @p src_elements to @p required_elements.  For a memref destination,
/// resizes the backing alloc.  For a FIFO destination, resizes the slot.
/// Returns failure and emits a diagnostic if resizing is not possible.
static mlir::LogicalResult widenTransfer(mlir::RewriterBase& rewriter,
                                         mlir::ktdf::DataTransferOp transfer,
                                         int64_t required_elements) {
  LDBG(1) << "widening transfer to " << required_elements
          << " elements: " << transfer;

  // Widen the source sizes (innermost dimension carries the element count).
  mlir::ktdf::setInnermostStaticSourceSize(rewriter, transfer,
                                           required_elements);

  // Widen the destination sizes and resize the backing storage.
  auto dest_sizes_opt = transfer.getStaticDestSizesArray();
  if (dest_sizes_opt) {
    llvm::SmallVector<int64_t> new_dest(*dest_sizes_opt);
    new_dest.back() = required_elements;
    transfer.setStaticDestSizes(llvm::ArrayRef<int64_t>(new_dest));
  }

  const mlir::Value dest = transfer.getDestination();
  if (transfer.isDestFifo()) {
    if (!mlir::ktdf::resizePrivateFifoSlot(rewriter, dest, required_elements))
      return transfer.emitError(
          "access-granularity-legalization: cannot resize FIFO slot");
  } else {
    // MemRef destination: resize to the new innermost dimension.
    auto dest_type = mlir::cast<mlir::MemRefType>(dest.getType());
    llvm::SmallVector<int64_t> new_shape(dest_type.getShape());
    new_shape.back() = required_elements;
    if (!mlir::ktdf::resizePrivateBuffer(rewriter, dest, new_shape))
      return transfer.emitError(
          "access-granularity-legalization: cannot resize destination buffer");
  }
  return mlir::success();
}

struct MemRefTransferInfo {
  int64_t elements;
  int64_t required_elements;
};

/// Computes the requested and required elements for a memref operand given its
/// static transfer sizes and target unit feature.
template <typename FeatureT>
static std::optional<MemRefTransferInfo> getMemRefTransferGranularity(
    mlir::Value memref, std::optional<llvm::ArrayRef<int64_t>> static_sizes,
    FeatureT feature) {
  if (!static_sizes || static_sizes->empty()) return std::nullopt;

  auto memref_type = mlir::dyn_cast<mlir::MemRefType>(memref.getType());
  if (!memref_type) return std::nullopt;

  mlir::Attribute memory_space = memref_type.getMemorySpace();
  if (!memory_space) return std::nullopt;

  const int64_t elements =
      std::accumulate(static_sizes->begin(), static_sizes->end(), int64_t{1},
                      std::multiplies<>());
  const int64_t required_elements = computeGranularity(
      elements, memref_type.getElementType(), memory_space, feature);

  return MemRefTransferInfo{elements, required_elements};
}

/// Inspect @p transfer and set/widen its access granularity to the unit's
/// hardware constraints.
static mlir::LogicalResult legalizeTransfer(
    mlir::RewriterBase& rewriter, mlir::ktdf::DataTransferOp transfer,
    const mlir::ktdf_arch::ResourceKinds& resource_kinds) {
  scheduler::ResourceType unit_kind = resolveUnitKind(transfer);
  if (!unit_kind) return mlir::success();

  // Handle units with load feature (memref source).
  if (auto load_feature =
          resource_kinds.getFeature<mlir::ktdf_arch::feature::Load>(
              unit_kind)) {
    if (!transfer.isSourceMemRef()) return mlir::success();

    auto info = getMemRefTransferGranularity(
        transfer.getSource(), transfer.getStaticSourceSizesArray(),
        load_feature);
    if (!info) return mlir::success();

    // Set/refine the access granularity attribute on the transfer.
    scheduler::setAccessGranularity(transfer, info->required_elements);

    // Leave splat widening to splat-legalization.
    if (transfer.getTransferMode() == mlir::ktdf::TransferMode::Splat)
      return mlir::success();

    if (info->required_elements <= info->elements) return mlir::success();

    return widenTransfer(rewriter, transfer, info->required_elements);
  }

  // Handle units with store feature (memref destination).
  if (auto store_feature =
          resource_kinds.getFeature<mlir::ktdf_arch::feature::Store>(
              unit_kind)) {
    if (!transfer.isDestMemRef()) return mlir::success();

    auto info = getMemRefTransferGranularity(transfer.getDestination(),
                                             transfer.getStaticDestSizesArray(),
                                             store_feature);
    if (!info) return mlir::success();

    if (info->required_elements > info->elements) {
      return transfer.emitError()
             << "store transfer of " << info->elements
             << " elements cannot satisfy the hardware store access "
                "granularity of "
             << info->required_elements << " elements without write-widening";
    }

    // Set/refine the access granularity attribute on the store transfer.
    scheduler::setAccessGranularity(transfer, info->required_elements);
    return mlir::success();
  }

  return mlir::success();
}

struct AccessGranularityLegalizationPass
    : public scheduler::impl::AccessGranularityLegalizationPassBase<
          AccessGranularityLegalizationPass> {
  void runOnOperation() override {
    if (DisableThisPass) return;
    LDBG(1) << "========= " PASS_NAME " =========";

    mlir::ModuleOp module_op = getOperation();

    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    auto* const device = device_manager.getOrImportDevice();
    if (!device) {
      module_op->emitError(
          "Unable to import the device specification. This could happen if the "
          "device spec file is empty or contains multiple devices");
      signalPassFailure();
      return;
    }
    auto& resource_kinds =
        device_manager.getOrCreateView<mlir::ktdf_arch::ResourceKinds>(*device);

    // Collect first to avoid invalidating the walk while rewriting.
    llvm::SmallVector<mlir::ktdf::DataTransferOp> transfers;
    module_op.walk([&](mlir::ktdf::DataTransferOp transfer) {
      transfers.push_back(transfer);
    });

    mlir::IRRewriter rewriter(&getContext());
    for (auto transfer : transfers) {
      if (mlir::failed(legalizeTransfer(rewriter, transfer, resource_kinds))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass>
scheduler::createAccessGranularityLegalizationPass() {
  return std::make_unique<AccessGranularityLegalizationPass>();
}
