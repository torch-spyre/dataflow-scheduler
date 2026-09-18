//===-- FuseLinalg.cpp ------------------------------------------*- c++ -*-===//
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

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/CommandLine.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <utility>

#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchAttributes.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArchIntrinsics.h"

#define PASS_NAME "fuse-linalg"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable Fuse Linalg pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_FUSELINALGPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

/// Breaks a false dependency through an init operand via 'tensor.empty'.
///
/// This is a copy of the (hidden) upstream pattern.
struct RemoveOutsDependency : mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::linalg::GenericOp generic,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    const auto loc = generic.getLoc();

    rewriter.startOpModification(generic);
    bool changed = false;
    for (auto& init : generic.getDpsInitsMutable()) {
      const auto type =
          llvm::dyn_cast<mlir::RankedTensorType>(init.get().getType());
      if (!generic.payloadUsesValueFromOperand(&init) || !type ||
          type.getEncoding() ||
          init.get().getDefiningOp<mlir::tensor::EmptyOp>()) {
        continue;
      }

      llvm::SmallVector<mlir::Value> sizes;
      for (auto [index, size] : llvm::enumerate(type.getShape())) {
        if (mlir::ShapedType::isDynamic(size)) {
          sizes.push_back(
              mlir::tensor::DimOp::create(rewriter, loc, init.get(), index));
        }
      }
      init.set(mlir::tensor::EmptyOp::create(rewriter, loc, type, sizes));
      changed = true;
    }

    if (!changed) {
      rewriter.cancelOpModification(generic);
      return llvm::failure();
    }
    rewriter.finalizeOpModification(generic);
    return llvm::success();
  }
};

/// Determines whether @p lhs and @p rhs are subsets of one another.
[[nodiscard]] auto setEqual(
    llvm::MutableArrayRef<mlir::ktdf_arch::ResourceSpecAttr> lhs,
    llvm::MutableArrayRef<mlir::ktdf_arch::ResourceSpecAttr> rhs) {
  const auto compare = [](mlir::ktdf_arch::ResourceSpecAttr l,
                          mlir::ktdf_arch::ResourceSpecAttr r) -> bool {
    return l.getAsOpaquePointer() < r.getAsOpaquePointer();
  };

  llvm::sort(lhs, compare);
  lhs = {lhs.begin(), llvm::unique(lhs)};
  llvm::sort(rhs, compare);
  rhs = {rhs.begin(), llvm::unique(rhs)};

  return lhs == rhs;
}

/// Determines whether the mappings of @p lhs and @p rhs permit fusion.
[[nodiscard]] auto fuseMappings(mlir::ktdf_arch::MapsToAttr lhs,
                                mlir::ktdf_arch::MapsToAttr rhs)
    -> llvm::FailureOr<mlir::ktdf_arch::MapsToAttr> {
  if (lhs == rhs || !lhs || !rhs) {
    return lhs ? lhs : rhs;
  }

  // For now, only permit fusion of ops with the same mapping.
  llvm::SmallVector<mlir::ktdf_arch::ResourceSpecAttr> buffer;
  buffer.reserve(lhs.getValue().size() + rhs.getValue().size());
  llvm::append_range(buffer, lhs.getValue());
  const auto split = buffer.size();
  llvm::append_range(buffer, rhs.getValue());
  if (!setEqual({buffer.data(), buffer.data() + split},
                {buffer.data() + split, buffer.end()})) {
    return llvm::failure();
  }

  // For stability, return the lhs mapping.
  return lhs;
}

/// Fuses an elementwise producer into a 'linalg.generic' consumer.
///
/// This is a mapping-aware copy of the (hidden) upstream pattern.
struct FuseElementwiseOps : mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  FuseElementwiseOps(mlir::MLIRContext* ctx,
                     mlir::linalg::ControlFusionFn control_fusion)
      : OpRewritePattern(ctx), control_fusion_(std::move(control_fusion)) {}

  auto matchAndRewrite(mlir::linalg::GenericOp generic,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    for (auto& through : generic->getOpOperands()) {
      if (!mlir::linalg::areElementwiseOpsFusable(&through)) {
        continue;
      }
      auto* const producer = through.get().getDefiningOp();
      const auto fused_mapping =
          fuseMappings(mlir::ktdf_arch::Mappable::getMapsTo(generic),
                       mlir::ktdf_arch::Mappable::getMapsTo(producer));
      if (failed(fused_mapping) || !control_fusion_(&through)) {
        continue;
      }

      const auto fusion_result =
          mlir::linalg::fuseElementwiseOps(rewriter, &through);
      if (failed(fusion_result)) {
        return rewriter.notifyMatchFailure(generic, "fusion failed");
      }

      if (*fused_mapping) {
        std::ignore = mlir::ktdf_arch::Mappable::setMapsTo(
            fusion_result->fusedOp, *fused_mapping);
      }

      for (auto [from, to] : fusion_result->replacements) {
        rewriter.replaceUsesWithIf(
            from, to, [&](const mlir::OpOperand& use) -> bool {
              return use.get().getDefiningOp() != producer;
            });
      }
      rewriter.eraseOp(generic);
      return llvm::success();
    }

    return llvm::failure();
  }

 private:
  mlir::linalg::ControlFusionFn control_fusion_;
};

/// Detects the implied mapping of @p generic and explicitly sets it.
///
/// @returns  Succeeds when @p generic explicitly declares the correct mapping.
auto propagateImpliedMapping(mlir::linalg::GenericOp generic)
    -> llvm::FailureOr<mlir::ktdf_arch::MapsToAttr> {
  auto mappable =
      llvm::dyn_cast<mlir::ktdf_arch::Mappable>(generic.getOperation());
  if (!mappable) {
    return llvm::success(nullptr);
  }
  if (const auto maps_to = mappable.getMapsTo()) {
    return llvm::success(maps_to);
  }

  // Try to determine a mapping from the contents of the operation.
  llvm::SetVector<mlir::ktdf_arch::ResourceSpecAttr> mapping;
  for (auto& child : *generic.getBody()) {
    if (const auto child_mapping = mlir::ktdf_arch::Mappable::getMapsTo(&child);
        child_mapping) {
      mapping.insert_range(child_mapping.getValue());
    }
  }
  if (mapping.empty()) {
    return llvm::success(nullptr);
  }

  // There are mapping constraints on the contents of this op that will
  // have to be respected. Try to set them on the op.
  const auto maps_to = mlir::ktdf_arch::MapsToAttr::get(generic->getContext(),
                                                        mapping.getArrayRef());
  if (llvm::succeeded(mappable.setMapsTo(maps_to))) {
    return llvm::success(maps_to);
  }

  return llvm::failure();
}

struct FuseLinalgPass : public impl::FuseLinalgPassBase<FuseLinalgPass> {
  using FuseLinalgPassBase<FuseLinalgPass>::FuseLinalgPassBase;

  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    // Collect all fusable linalg.generic candidate operations.
    llvm::SetVector<mlir::Operation*> candidates;
    const auto collect = [&](mlir::linalg::GenericOp generic) {
      if (succeeded(propagateImpliedMapping(generic))) {
        candidates.insert(generic);
      }
    };
    getOperation()->walk(collect);

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RemoveOutsDependency>(patterns.getContext());
    patterns.add<FuseElementwiseOps>(
        patterns.getContext(), [&](mlir::OpOperand* through) -> bool {
          auto* const producer = through->get().getDefiningOp();
          return candidates.contains(producer);
        });

    // Fuse as many of the candidates as possible.
    auto changed = false;
    const auto rewrite_config =
        mlir::GreedyRewriteConfig()
            .setStrictness(mlir::GreedyRewriteStrictness::ExistingAndNewOps)
            .enableConstantCSE();
    if (mlir::failed(mlir::applyOpPatternsGreedily(candidates.getArrayRef(),
                                                   std::move(patterns),
                                                   rewrite_config, &changed))) {
      signalPassFailure();
    }
  }
};

}  // namespace
