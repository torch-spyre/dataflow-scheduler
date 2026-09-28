//===-- KTIRPipeline.cpp ----------------------------------------*- c++ -*-===//
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

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/SmallVectorExtras.h>
#include <llvm/Support/DebugLog.h>
#include <llvm/Support/LogicalResult.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "Utils.h"
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h"  // IWYU pragma: keep
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Transforms/PipelineBuilder.h"
#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/Mapping.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Dialect/KTDPLowering/KTDPLowering.h"

#define PASS_NAME "ktir-pipeline"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME, llvm::cl::desc("Disable KTIR Pipeline pass"),
    llvm::cl::init(false));

using namespace scheduler;

namespace scheduler {
#define GEN_PASS_DEF_KTIRPIPELINEPASS
#include "dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR/Passes.h.inc"
}  // namespace scheduler

namespace {

struct KTIRPipelinePass : public impl::KTIRPipelinePassBase<KTIRPipelinePass> {
  using KTIRPipelinePassBase<KTIRPipelinePass>::KTIRPipelinePassBase;

  void runOnOperation() override;
};

void findLoopNest(llvm::SmallVectorImpl<mlir::scf::ForOp>& loops) {
  if (loops.empty()) {
    return;
  }

  auto loop = loops.back();
  while (true) {
    const auto body = loop.getBody()->without_terminator();
    if (body.empty() || std::next(body.begin()) != body.end()) {
      return;
    }

    auto inner = llvm::cast<mlir::scf::ForOp>(&*body.begin());
    if (!inner) {
      return;
    }

    loop = loops.emplace_back(inner);
  }
}

auto createPipeline(mlir::RewriterBase& rewriter, mlir::scf::ForOp outermost,
                    mlir::DominanceInfo& dominance) -> mlir::ktdf::PipelineOp {
  mlir::scf::LoopVector loop_nest{outermost};
  findLoopNest(loop_nest);
  auto innermost = loop_nest.back();

  // Collect `ktdp_lowering.(load|store)` and `ktdf.via` operations.
  llvm::SmallVector<mlir::ktdp_lowering::LoadOp> loads;
  llvm::SmallVector<mlir::ktdp_lowering::StoreOp> stores;
  llvm::SmallVector<mlir::ktdf::ViaOp> vias;
  outermost.walk([&](mlir::Operation* op) {
    if (auto load = mlir::dyn_cast<mlir::ktdp_lowering::LoadOp>(op); load) {
      loads.push_back(load);
    } else if (auto store = mlir::dyn_cast<mlir::ktdp_lowering::StoreOp>(op);
               store) {
      stores.push_back(store);
    } else if (auto via = mlir::dyn_cast<mlir::ktdf::ViaOp>(op); via) {
      vias.push_back(via);
    }
  });

  // Create the pipeline.
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(innermost.getBody());
  mlir::ktdf::PipelineBuilder pipeline_builder(
      rewriter, rewriter.getFusedLoc(llvm::map_to_vector(
                    stores, [](mlir::Operation* op) { return op->getLoc(); })));

  // Populate the pipeline with operations.
  const auto get_mem_space = [&](mlir::Value value) -> mlir::Attribute {
    if (auto access_tile = llvm::dyn_cast<AccessTile>(value); access_tile) {
      return getMemorySpace(access_tile);
    }
    return getMemorySpace(llvm::cast<MemRef>(value));
  };
  const auto classify =
      [&](mlir::ktdf::PipelineBuilder& builder,
          mlir::Operation* op) -> mlir::ktdf::PipelineBuilder::Placement {
    if (auto store = llvm::dyn_cast<mlir::ktdp_lowering::StoreOp>(op); store) {
      auto stage = builder.createStage();
      stage.setApplicableUnitsAttr(mlir::ArrayAttr::get(
          builder.getContext(), {get_mem_space(store.getDest())}));
      return {stage, true};
    }
    if (auto load = llvm::dyn_cast<mlir::ktdp_lowering::LoadOp>(op); load) {
      return builder.tryPlacement(get_mem_space(load.getSource()));
    }
    if (auto via = llvm::dyn_cast<mlir::ktdf::ViaOp>(op); via) {
      if (via.getHops().size() > 1) {
        rewriter.setInsertionPoint(via);
        auto rest = mlir::ktdf::ViaOp::create(
            rewriter, via.getLoc(), via.getOperand(),
            rewriter.getArrayAttr(via.getHops().getValue().drop_back(1)));
        rewriter.modifyOpInPlace(via, [&]() {
          via.setOperand(rest);
          via.setHopsAttr(
              rewriter.getArrayAttr(via.getHops().getValue().back()));
        });
      }
      return builder.tryPlacement(via.getHopsAttr());
    }
    if (auto compute = llvm::dyn_cast<mlir::linalg::LinalgOp>(op); compute) {
      return builder.tryPlacement(
          mlir::ktdf_arch::Mappable::getMapsTo(compute));
    }
    if (!innermost->isAncestor(op)) {
      return nullptr;
    }

    return mlir::ktdf::PipelineBuilder::naturalPlacement(op);
  };
  pipeline_builder.insert(
      llvm::ArrayRef<mlir::Operation*>(
          reinterpret_cast<mlir::Operation* const*>(stores.data()),
          stores.size()),
      classify, dominance);

  auto result = pipeline_builder.finalize();
  LDBG() << "created " << result;
  return result;
}

[[nodiscard]] auto makeMap(mlir::OpBuilder& builder,
                           mlir::OffsetSizeAndStrideOpInterface iface)
    -> std::pair<mlir::AffineMap, mlir::ValueRange> {
  llvm::SmallVector<mlir::AffineExpr> results;

  auto offsets = iface.getMixedOffsets();
  auto strides = iface.getMixedStrides();
  unsigned dim_idx = 0;
  for (auto idx : llvm::iota_range<unsigned>(0, offsets.size(), false)) {
    if (const auto value = mlir::getConstantIntValue(offsets[idx]); value) {
      results.push_back(builder.getAffineConstantExpr(*value));
    } else {
      results.push_back(builder.getAffineDimExpr(dim_idx++));
    }

    if (const auto value = mlir::getConstantIntValue(strides[idx]); value) {
      results.back() = results.back() * builder.getAffineConstantExpr(*value);
    } else {
      llvm::report_fatal_error("dynamic strides are not supported");
    }
  }

  return {mlir::AffineMap::get(iface.getOffsets().size(), 0, results,
                               builder.getContext()),
          iface.getOffsets()};
}

struct LowerLoad : mlir::OpRewritePattern<mlir::ktdp_lowering::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdp_lowering::LoadOp load,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto* const use = getSingleUse(load);
    auto write =
        use ? llvm::dyn_cast<mlir::ktdf::WriteToFifoOp>(use->getOwner())
            : nullptr;
    if (!write) {
      return rewriter.notifyMatchFailure(load, "does not write to FIFO");
    }
    auto memref = mlir::dyn_cast<MemRef>(load.getSource());
    if (!memref) {
      return rewriter.notifyMatchFailure(load, "not bufferized");
    }

    auto [map, ivs] = makeMap(rewriter, load);
    auto transfer = mlir::ktdf::DataTransferOp::create(
        rewriter, load.getLoc(), memref, map, ivs, load.getMixedSizes(),
        write.getFifoSlot(), {}, {}, {});
    transfer->setDiscardableAttrs(load->getRawDictionaryAttrs());
    rewriter.eraseOp(write);
    rewriter.eraseOp(load);
    return llvm::success();
  }
};

struct LowerStore : mlir::OpRewritePattern<mlir::ktdp_lowering::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdp_lowering::StoreOp store,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    auto read = store.getSource().getDefiningOp<mlir::ktdf::ReadFromFifoOp>();
    if (!read) {
      return rewriter.notifyMatchFailure(store, "does not read from FIFO");
    }
    if (!read->hasOneUse()) {
      return read.emitError("has multiple uses");
    }
    auto memref = mlir::dyn_cast<MemRef>(store.getDest());
    if (!memref) {
      return rewriter.notifyMatchFailure(store, "not bufferized");
    }

    auto [map, ivs] = makeMap(rewriter, store);
    auto transfer = mlir::ktdf::DataTransferOp::create(
        rewriter, store.getLoc(), read.getFifoSlot(), {}, {}, {}, memref, map,
        ivs, store.getMixedSizes());
    transfer->setDiscardableAttrs(store->getRawDictionaryAttrs());
    rewriter.eraseOp(store);
    rewriter.eraseOp(read);
    return llvm::success();
  }
};

struct LowerVia : mlir::OpRewritePattern<mlir::ktdf::ViaOp> {
  using OpRewritePattern::OpRewritePattern;

  auto matchAndRewrite(mlir::ktdf::ViaOp via,
                       mlir::PatternRewriter& rewriter) const
      -> llvm::LogicalResult override {
    return mlir::ktdf::eliminateVia(rewriter, via);
  }
};

auto postProcess(mlir::RewriterBase& rewriter, mlir::ktdf::PipelineOp pipeline,
                 mlir::ktdf_arch::Mapping& mapping) -> llvm::LogicalResult {
  // Combine FIFO slots.
  // FIXME: Path expansion expects to have one alloc per unit? How does that
  //        work for different dynamic sizes?
  llvm::DenseMap<std::pair<mlir::Attribute, mlir::Attribute>, mlir::OpOperand*>
      fifo_map;
  for (auto& operand : pipeline.getPrivateOp().getYieldOp()->getOpOperands()) {
    auto alloc = operand.get().getDefiningOp<mlir::ktdf::FifoAllocateOp>();
    if (!alloc || alloc.getNumResults() != 1) {
      continue;
    }

    const auto type =
        llvm::cast<mlir::ktdf::FifoSlotType>(alloc->getResult(0).getType());
    const auto [it, added] =
        fifo_map.try_emplace({type.getSrc(), type.getDest()}, &operand);
    if (added) {
      continue;
    }

    auto widen = it->second->get().getDefiningOp<mlir::ktdf::FifoAllocateOp>();
    if (widen.getDynamicSizes() != alloc.getDynamicSizes()) {
      continue;
    }

    rewriter.setInsertionPoint(widen);
    llvm::SmallVector<mlir::Type> types(widen->getResultTypes());
    types.push_back(operand.get().getType());
    auto new_alloc = mlir::ktdf::FifoAllocateOp::create(
        rewriter, widen.getLoc(), types, widen.getDynamicSizes());
    rewriter.replaceOp(widen, new_alloc->getResults().drop_back(1));
    rewriter.replaceOp(alloc, new_alloc->getResults().back());
  }

  // Remove all non-execution-unit mappings on the stages.
  // FIXME: We'll need to improve this to provide path expansion with what it
  //        needs top handle vias.
  for (auto stage : pipeline.getStages()) {
    auto units_attr = llvm::dyn_cast_if_present<
        mlir::ktdf_arch::TypedArrayAttr<mlir::ktdf_arch::ResourceSpecAttr>>(
        stage.removeApplicableUnitsAttr());
    if (!units_attr) {
      continue;
    }

    llvm::SmallVector<mlir::Attribute> units;
    for (auto unit : units_attr) {
      if (auto resource =
              mapping.lookup<mlir::ktdf_arch::ExecutionUnitOp>(unit);
          resource) {
        units.push_back(unit);
      }
    }

    if (!units.empty()) {
      stage.setApplicableUnitsAttr(rewriter.getArrayAttr(units));
    }
  }

  return llvm::success();
}

}  // namespace

void KTIRPipelinePass::runOnOperation() {
  if (disable_this_pass) {
    return;
  }

  mlir::func::FuncOp func = getOperation();
  mlir::IRRewriter rewriter(func);
  mlir::DominanceInfo dominance;

  // Obtain the default device and start a mapping.
  auto& default_device = getAnalysis<mlir::ktdf_arch::DefaultDevice>();
  if (!default_device) {
    signalPassFailure();
    return;
  }
  mlir::ktdf_arch::Mapping mapping(default_device.getRef());

  // Create the pipelines.
  const auto pipelines = llvm::map_to_vector(
      func.getOps<mlir::scf::ForOp>(), [&](mlir::scf::ForOp outermost) {
        return createPipeline(rewriter, outermost, dominance);
      });

  // Legalize the pipelines.
  {
    mlir::ConversionTarget target(getContext());
    target.addIllegalOp<mlir::ktdp_lowering::LoadOp>();
    target.addIllegalOp<mlir::ktdp_lowering::StoreOp>();
    target.addIllegalOp<mlir::ktdf::ViaOp>();
    target.addLegalOp<mlir::ktdf::DataTransferOp>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerLoad, LowerStore, LowerVia>(patterns.getContext());

    const auto ops = llvm::ArrayRef<mlir::Operation*>(
        reinterpret_cast<mlir::Operation* const*>(pipelines.data()),
        pipelines.size());
    if (failed(
            mlir::applyPartialConversion(ops, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

  // FIXME: Get rid of this additional post-processing and put it into the
  //        patterns for legalization.
  for (auto pipeline : pipelines) {
    if (failed(postProcess(rewriter, pipeline, mapping))) {
      signalPassFailure();
      return;
    }
  }
}
