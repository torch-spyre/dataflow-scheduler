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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/BufferPhaseLowering.h"

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"
// Marker used only to locate buffer_phase_op inside the cloned loop tree.
// Set before cloning, removed immediately after the cloned op is found.
#define PROCESSING_MARKER_ATTR "__processing_marker__"

using namespace scheduler;

namespace {

/// Controls where the iter-arg toggle (subi) is placed in a double-buffered
/// loop. BufferFirst places it at the yield (most units); BufferLast places it
/// at the loop-body top (MNILU/MNISU).
enum class DoubleBufferingMode { BufferFirst, BufferLast };

[[nodiscard]]
auto getDoubleBufferingMode(mlir::ktdf_arch::Resource resource)
    -> DoubleBufferingMode {
  return resource->getAttr("dataflow_scheduler.double_buffer_last")
             ? DoubleBufferingMode::BufferLast
             : DoubleBufferingMode::BufferFirst;
}

bool areMemoryViewsMatchingExceptOffset(
    mlir::dataflow::GetLogicalMemoryViewOp view_0,
    mlir::dataflow::GetLogicalMemoryViewOp view_1) {
  return view_0.getResult().getType() == view_1.getResult().getType() &&
         view_0.getLayoutMap() == view_1.getLayoutMap() &&
         view_0.getFromUnit() == view_1.getFromUnit();
}

// Returns a map from original ForOp → cloned ForOp. The cloned chain has one
// additional iter arg cascaded from outermost to innermost. In BufferFirst mode
// the toggle subi is placed at yield; in BufferLast mode it is placed at the
// top of the innermost loop body.
llvm::DenseMap<mlir::scf::ForOp, mlir::scf::ForOp>
cloneLoopChainWithDoubleBufferingIterArgs(
    llvm::ArrayRef<mlir::scf::ForOp> loops, mlir::Value initial_offset,
    mlir::Value offset_sum, DoubleBufferingMode mode) {
  llvm::DenseMap<mlir::scf::ForOp, mlir::scf::ForOp> cloned_loops;

  // Clone from innermost to outermost so each outer clone finds its inner
  // already in cloned_loops.
  for (int i = static_cast<int>(loops.size()) - 1; i >= 0; --i) {
    auto loop = loops[i];
    bool is_innermost = (i == static_cast<int>(loops.size()) - 1);

    mlir::IRMapping ir_map;
    auto new_loop = createForOpWithAdditionalIterArgs(
        loop, /*n_values=*/1, ir_map, /*delete_op=*/true);
    cloned_loops[loop] = new_loop;

    auto yield_op =
        mlir::cast<mlir::scf::YieldOp>(new_loop.getBody()->getTerminator());
    mlir::OpBuilder yield_builder(yield_op);
    auto new_iter_arg = new_loop.getRegionIterArgs().back();

    mlir::Value yield_value;
    if (is_innermost) {
      if (mode == DoubleBufferingMode::BufferFirst) {
        // Mode A: toggle subi at yield
        yield_value = mlir::arith::SubIOp::create(
            yield_builder, yield_op.getLoc(), offset_sum, new_iter_arg);
      } else {
        // Mode B: insert toggle subi at loop body top, yield its result
        mlir::OpBuilder body_builder(&new_loop.getBody()->front());
        yield_value = mlir::arith::SubIOp::create(
            body_builder, yield_op.getLoc(), offset_sum, new_iter_arg);
      }
    } else {
      // Outer loop: the previously-cloned inner loop was re-cloned into
      // new_loop's body by createForOpWithAdditionalIterArgs (and the
      // pre-clone was destroyed when its containing outer loop was erased).
      // Use ir_map to find the live re-clone inside new_loop's body.
      auto pre_clone_inner = cloned_loops[loops[i + 1]];
      auto* re_cloned_op = ir_map.lookupOrNull(pre_clone_inner.getOperation());
      assert(re_cloned_op &&
             "inner loop should have been re-cloned into outer loop body");
      auto inner_re_cloned = mlir::cast<mlir::scf::ForOp>(re_cloned_op);
      // Update cloned_loops to the live clone so the next outer iteration
      // (if any) can find it.
      cloned_loops[loops[i + 1]] = inner_re_cloned;
      yield_value = inner_re_cloned.getResults().back();

      // Cascade: inner loop's new iter arg init = outer loop's new iter arg
      inner_re_cloned
          .getInitsMutable()[inner_re_cloned.getNumRegionIterArgs() - 1]
          .assign(new_iter_arg);
    }

    llvm::SmallVector<mlir::Value> new_yield_operands(
        yield_op.getOperands().begin(), yield_op.getOperands().end());
    new_yield_operands.back() = yield_value;
    yield_op->setOperands(new_yield_operands);
  }

  // Set the initial value on the outermost loop's new iter arg
  if (!loops.empty()) {
    auto outermost = cloned_loops[loops.front()];
    outermost.getInitsMutable()[outermost.getNumRegionIterArgs() - 1].assign(
        initial_offset);
  }

  return cloned_loops;
}

// Replace cloned_buffer_phase and cloned_select with a new
// get_logical_memory_view using mem_view_offset as the offset operand.
mlir::LogicalResult replaceBufferPhaseAndSelectInClonedLoops(
    mlir::ktdf::BufferPhaseOp cloned_buffer_phase,
    mlir::ktdf::SelectMemrefOp cloned_select, mlir::MemRefType memref_ty,
    mlir::Value unit, mlir::AffineMapAttr layout_map_attr,
    mlir::Value mem_view_offset) {
  mlir::OpBuilder builder(cloned_select);
  auto new_view = mlir::dataflow::GetLogicalMemoryViewOp::create(
      builder, cloned_select.getLoc(), memref_ty, unit, mem_view_offset,
      layout_map_attr);
  cloned_select.getResult().replaceAllUsesWith(new_view.getResult());
  return mlir::success();
}

mlir::LogicalResult processOneBufferPhasePair(
    mlir::ktdf::BufferPhaseOp buffer_phase_op,
    const ResourceToUnits& components,
    arch_view::ResourceKinds& resource_kinds) {
  // Validate num_phases == 2
  auto num_phases_attr = buffer_phase_op.getNumPhasesAttr();
  if (!num_phases_attr || num_phases_attr.getInt() != 2) {
    buffer_phase_op.emitError(
        "buffer_phase must have num_phases = 2 for double buffering");
    return mlir::failure();
  }

  // TODO: Handle case where buffer_phase_op has multiple users
  // (or check against it)
  auto select_op = mlir::dyn_cast<mlir::ktdf::SelectMemrefOp>(
      *buffer_phase_op.getResult().getUsers().begin());
  if (!select_op) {
    buffer_phase_op.emitError(
        "buffer_phase's sole user must be a select_memref operation");
    return mlir::failure();
  }

  auto candidates = select_op.getCandidates();
  if (candidates.size() != 2) {
    select_op.emitError("select_memref must have exactly 2 candidate memrefs");
    return mlir::failure();
  }

  auto get_mem_view_0 =
      candidates[0].getDefiningOp<mlir::dataflow::GetLogicalMemoryViewOp>();
  auto get_mem_view_1 =
      candidates[1].getDefiningOp<mlir::dataflow::GetLogicalMemoryViewOp>();
  if (!get_mem_view_0 || !get_mem_view_1) {
    select_op.emitError(
        "select_memref operands must be get_logical_memory_view operations");
    return mlir::failure();
  }
  if (!areMemoryViewsMatchingExceptOffset(get_mem_view_0, get_mem_view_1)) {
    select_op.emitError(
        "get_logical_memory_view operations must match except for offset");
    return mlir::failure();
  }

  auto memref_ty =
      llvm::cast<mlir::MemRefType>(get_mem_view_0.getResult().getType());
  mlir::Value unit = get_mem_view_0.getFromUnit();
  mlir::Value offset_0 = get_mem_view_0.getStartAddress();
  mlir::Value offset_1 = get_mem_view_1.getStartAddress();

  // Resolve mode from enclosing program_unit's resource type
  mlir::ktdf_arch::Resource resource;
  auto resource_opt =
      getEnclosingProgramUnitResourceType(buffer_phase_op.getOperation());
  if (resource_opt) {
    resource = resource_kinds.getResource(*resource_opt);
  }
  if (!resource) {
    buffer_phase_op.emitError(
        "lowerDoubleBuffering: could not resolve component type from "
        "enclosing program_unit");
    return mlir::failure();
  }

  auto mode = getDoubleBufferingMode(resource);

  // Collect loops from IV operands
  llvm::SmallVector<mlir::scf::ForOp> loops;
  for (auto iv : buffer_phase_op.getIvs()) {
    auto block_arg = mlir::cast<mlir::BlockArgument>(iv);
    loops.push_back(
        mlir::cast<mlir::scf::ForOp>(block_arg.getOwner()->getParentOp()));
  }

  // Compute offset_sum before the outermost loop
  mlir::OpBuilder builder(loops.front());
  auto offset_sum = mlir::arith::AddIOp::create(
      builder, buffer_phase_op.getLoc(), offset_0, offset_1);

  // Mark buffer_phase_op so we can find its clone after loop cloning
  buffer_phase_op->setAttr(PROCESSING_MARKER_ATTR,
                           mlir::UnitAttr::get(buffer_phase_op.getContext()));

  // Select initial offset: Mode A = buffer0, Mode B = buffer1
  mlir::Value initial_offset =
      (mode == DoubleBufferingMode::BufferFirst) ? offset_0 : offset_1;

  auto cloned_loops = cloneLoopChainWithDoubleBufferingIterArgs(
      loops, initial_offset, offset_sum, mode);

  LDBG(1) << "Loops cloned, finding marked buffer_phase_op";

  // Find the cloned buffer_phase by marker
  auto cloned_outermost = cloned_loops[loops.front()];
  mlir::ktdf::BufferPhaseOp cloned_buffer_phase = nullptr;
  mlir::ktdf::SelectMemrefOp cloned_select = nullptr;

  cloned_outermost.walk([&](mlir::ktdf::BufferPhaseOp bp) {
    if (bp->hasAttr(PROCESSING_MARKER_ATTR)) {
      cloned_buffer_phase = bp;
      bp->removeAttr(PROCESSING_MARKER_ATTR);
      cloned_select = mlir::dyn_cast<mlir::ktdf::SelectMemrefOp>(
          *bp.getResult().getUsers().begin());
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  if (!cloned_buffer_phase || !cloned_select) {
    cloned_outermost.emitError(
        "could not find cloned buffer_phase or select_memref after loop "
        "cloning");
    return mlir::failure();
  }

  LDBG(1) << "Found cloned buffer_phase and select_memref";

  // Compute the offset value for get_logical_memory_view.
  // Mode A: iter_arg (toggle is at yield).
  // Mode B: subi result inserted at loop body top.
  auto ivs = cloned_buffer_phase.getIvs();
  if (ivs.empty()) {
    cloned_buffer_phase.emitError("buffer_phase must have at least one IV");
    return mlir::failure();
  }
  auto innermost_iv = ivs.back();
  auto cloned_block_arg = mlir::cast<mlir::BlockArgument>(innermost_iv);
  auto innermost_loop =
      mlir::cast<mlir::scf::ForOp>(cloned_block_arg.getOwner()->getParentOp());
  mlir::Value iter_arg =
      innermost_loop
          .getRegionIterArgs()[innermost_loop.getNumRegionIterArgs() - 1];

  mlir::Value mem_view_offset = iter_arg;
  if (mode == DoubleBufferingMode::BufferLast) {
    auto* first_op = &innermost_loop.getBody()->front();
    auto subi = mlir::dyn_cast<mlir::arith::SubIOp>(first_op);
    assert(
        subi &&
        "Expected arith.subi at top of innermost loop body in BufferLast mode");
    mem_view_offset = subi.getResult();
  }

  if (mlir::failed(replaceBufferPhaseAndSelectInClonedLoops(
          cloned_buffer_phase, cloned_select, memref_ty, unit,
          get_mem_view_0.getLayoutMapAttr(), mem_view_offset))) {
    return mlir::failure();
  }

  // Erase cloned intermediates
  cloned_select.erase();
  if (cloned_buffer_phase.use_empty()) cloned_buffer_phase.erase();

  // Erase original get_logical_memory_view ops
  assert(get_mem_view_0.use_empty());
  get_mem_view_0.erase();
  assert(get_mem_view_1.use_empty());
  get_mem_view_1.erase();

  return mlir::success();
}

}  // namespace

mlir::LogicalResult scheduler::lowerDoubleBuffering(
    mlir::func::FuncOp func, const ResourceToUnits& components,
    arch_view::ResourceKinds& resource_kinds) {
  while (true) {
    mlir::ktdf::BufferPhaseOp found = nullptr;
    func.walk([&](mlir::ktdf::BufferPhaseOp bp) {
      found = bp;
      return mlir::WalkResult::interrupt();
    });
    if (!found) break;
    if (mlir::failed(
            processOneBufferPhasePair(found, components, resource_kinds)))
      return mlir::failure();
  }
  return mlir::success();
}
