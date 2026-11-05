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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/PatternMatch.h>

#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Dataflow/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDF/Utils/Utils.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "ktir/Dialect/KTDP/KTDP.h"

using namespace scheduler;

auto scheduler::getFlattenedVectorType(mlir::ShapedType type)
    -> mlir::VectorType {
  if (!type.hasStaticShape()) {
    return nullptr;
  }

  return mlir::VectorType::get({type.getNumElements()}, type.getElementType());
}

mlir::Value scheduler::createQueryMapForComponent(
    mlir::OpBuilder& builder, mlir::dataflow::ProgramUnitOp program_unit,
    const llvm::SmallVector<mlir::Value, 4>& target_units, mlir::Location loc) {
  mlir::ValueRange pu_operands = program_unit.getUnits();
  mlir::Block& body = program_unit.getRegion().front();
  mlir::Value iter_arg = body.getArgument(0);

  llvm::SmallVector<mlir::Value> keys;
  llvm::SmallVector<mlir::Value> values;

  for (mlir::Value pu_op : pu_operands) {
    auto get_unit = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
        pu_op.getDefiningOp());
    assert(get_unit &&
           "program_unit operand must be defined by dataflow.get_unit");
    auto core_attr = get_unit->getAttrOfType<mlir::IntegerAttr>("core");
    assert(core_attr && "dataflow.get_unit must have 'core' attribute");
    int core = static_cast<int>(core_attr.getInt());
    int corelet = mlir::dataflow::getCoreletId(get_unit);

    mlir::Value matching_target;
    for (mlir::Value target : target_units) {
      auto target_get_unit = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
          target.getDefiningOp());
      assert(target_get_unit &&
             "target unit must be defined by dataflow.get_unit");
      auto target_core_attr =
          target_get_unit->getAttrOfType<mlir::IntegerAttr>("core");
      assert(target_core_attr &&
             "target dataflow.get_unit must have 'core' attribute");
      if (static_cast<int>(target_core_attr.getInt()) != core) continue;
      // When the source unit has a corelet, the target must match it so that
      // each src corelet maps to its own dst corelet (not always corelet 0).
      int target_corelet = mlir::dataflow::getCoreletId(target_get_unit);
      if (corelet >= 0 && target_corelet != corelet) continue;
      matching_target = target;
      break;
    }

    assert(matching_target &&
           "must find matching target unit for each program_unit operand");
    keys.push_back(pu_op);
    values.push_back(matching_target);
  }

  assert(!keys.empty() && "must have at least one key-value pair");

  auto map_op = mlir::uniform::DefImmutableMappingOp::create(
      builder, loc, builder.getIndexType(), keys, values);
  auto query_op = mlir::uniform::QueryMapOp::create(
      builder, loc, builder.getIndexType(), map_op.getResult(), iter_arg);

  return query_op.getResult();
}

mlir::LogicalResult scheduler::replaceComputeTileIdWithCoreQuery(
    mlir::dataflow::ProgramUnitOp program_unit,
    llvm::DenseMap<int64_t, mlir::Value>& core_id_consts,
    mlir::OpBuilder& const_builder) {
  mlir::Region& region = program_unit.getRegion();
  mlir::Block& body = region.front();

  // Collect single-result get_compute_tile_id ops with a use inside this
  // program_unit's region. The op is typically defined outside the region (in
  // the enclosing function) and captured in. Walk the enclosing function body
  // (the program_unit's parent op) — this avoids needing ModuleOp and is the
  // scope where the captured tile-id is defined.
  llvm::SmallVector<mlir::ktdp::GetComputeTileIdOp> tile_ids;
  mlir::Operation* parent = program_unit->getParentOp();
  parent->walk([&](mlir::ktdp::GetComputeTileIdOp tid) {
    if (tid->getNumResults() != 1) return;  // single-result only
    for (mlir::OpOperand& use : tid->getResult(0).getUses()) {
      if (region.isAncestor(use.getOwner()->getParentRegion())) {
        tile_ids.push_back(tid);
        break;
      }
    }
  });
  if (tile_ids.empty()) return mlir::success();  // nothing to do for this unit

  // Build the core map: keys = program_unit operands, values = SHARED constant
  // flat core ids read from each operand get_unit's 'core' attribute. Constants
  // are materialized once at function scope (const_builder) and cached in
  // core_id_consts, then captured into this body (units are not
  // IsolatedFromAbove).
  mlir::OpBuilder builder(&body, body.begin());
  auto loc = program_unit.getLoc();
  llvm::SmallVector<mlir::Value> keys;
  llvm::SmallVector<mlir::Value> values;
  for (mlir::Value pu_op : program_unit.getUnits()) {
    auto get_unit = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
        pu_op.getDefiningOp());
    if (!get_unit) {
      return program_unit.emitError(
          "program_unit operand must be defined by dataflow.get_unit");
    }
    auto core_attr = get_unit->getAttrOfType<mlir::IntegerAttr>("core");
    if (!core_attr) {
      return program_unit.emitError(
          "dataflow.get_unit must have 'core' attribute");
    }
    int64_t core = core_attr.getInt();
    auto cached = core_id_consts.find(core);
    mlir::Value core_const;
    if (cached != core_id_consts.end()) {
      core_const = cached->second;
    } else {
      core_const =
          mlir::arith::ConstantIndexOp::create(const_builder, loc, core);
      core_id_consts[core] = core_const;
    }
    keys.push_back(pu_op);
    values.push_back(core_const);
  }

  auto map_op = mlir::uniform::DefImmutableMappingOp::create(
      builder, loc, builder.getIndexType(), keys, values);
  mlir::Value iter_arg = body.getArgument(0);
  auto query_op = mlir::uniform::QueryMapOp::create(
      builder, loc, builder.getIndexType(), map_op.getResult(), iter_arg);
  mlir::Value core_id = query_op.getResult();

  // NOTE: in-region uses include any NESTED program_unit bodies. This assumes
  // nested program_units share the same unit set (hence the same core) as
  // this outer unit, which holds in the current pipeline; a nested unit with
  // a different unit set would need its own per-unit core query instead.
  // Redirect every in-region use of each tile-id result to the query result.
  for (mlir::ktdp::GetComputeTileIdOp tid : tile_ids) {
    tid->getResult(0).replaceUsesWithIf(core_id, [&](mlir::OpOperand& use) {
      return region.isAncestor(use.getOwner()->getParentRegion());
    });
  }

  return mlir::success();
}

llvm::FailureOr<mlir::Value> scheduler::resolveUnitFromFifoAttr(
    mlir::Attribute fifo_attr, const ResourceToUnits& components,
    mlir::PatternRewriter& rewriter, mlir::dataflow::ProgramUnitOp program_unit,
    mlir::Location loc, mlir::Operation* op_for_errors) {
  // Step 1: cast the raw attribute to a StringAttr and upper-case it.
  auto str_attr = mlir::dyn_cast<mlir::StringAttr>(fifo_attr);
  if (!str_attr) {
    op_for_errors->emitError("unsupported FIFO endpoint attribute type");
    return mlir::failure();
  }
  scheduler::ResourceType component_type =
      mlir::StringAttr::get(str_attr.getContext(), str_attr.getValue().upper());

  // Step 2: look up the resource type in the components map.
  auto it = components.find(component_type);
  if (it == components.end()) {
    op_for_errors->emitError()
        << "no units found for FIFO component type: " << component_type;
    return mlir::failure();
  }

  // Step 3: build and return the query_map for that component.
  return createQueryMapForComponent(rewriter, program_unit, it->second, loc);
}

llvm::FailureOr<scheduler::DataTransferType> scheduler::getDataTransferType(
    bool src_is_fifo, bool dst_is_fifo) {
  // Case 1: Both source and destination are memrefs (memory to memory)
  if (!src_is_fifo && !dst_is_fifo) {
    return scheduler::DataTransferType::kLoadAndStore;
  }

  // Case 2: Source is memref, destination is FIFO slot
  if (!src_is_fifo && dst_is_fifo) {
    return scheduler::DataTransferType::kLoadAndSend;
  }

  // Case 3: Source is FIFO slot, destination is memref
  if (src_is_fifo && !dst_is_fifo) {
    return scheduler::DataTransferType::kReceiveAndStore;
  }

  // Both source and destination are FIFO slots - unsupported
  return llvm::failure();
}

namespace {

/// Gets whether \p map, an indexing map of a loop nest of \p ranges, takes the
/// loop iterations to the elements of its operand in the same row-major order:
/// a permutation that only moves loops of one iteration.
auto keepsRowMajorOrder(mlir::AffineMap map, llvm::ArrayRef<int64_t> ranges)
    -> bool {
  if (!map.isPermutation()) return false;
  int64_t last = -1;
  for (auto result : map.getResults()) {
    const auto dim = llvm::cast<mlir::AffineDimExpr>(result).getPosition();
    if (ranges[dim] == 1) continue;
    if (static_cast<int64_t>(dim) < last) return false;
    last = dim;
  }
  return true;
}

}  // namespace

auto scheduler::isFloatConversion(mlir::Operation* op) -> bool {
  auto generic = mlir::dyn_cast_if_present<mlir::linalg::GenericOp>(op);
  if (!generic || generic.getNumDpsInputs() != 1 ||
      generic.getNumDpsInits() != 1 || !generic.hasPureTensorSemantics() ||
      generic.getNumResults() != 1) {
    return false;
  }
  // Lane-wise: every lane of the input converts into the same lane of the
  // result, so the conversion is one of whole vectors.
  if (generic.getNumParallelLoops() != generic.getNumLoops() ||
      !llvm::all_of(generic->getOpOperands(), [&](mlir::OpOperand& operand) {
        return keepsRowMajorOrder(generic.getMatchingIndexingMap(&operand),
                                  generic.getStaticLoopRanges());
      })) {
    return false;
  }
  // The body converts its input and yields that, and does nothing else.
  mlir::Block& body = generic.getRegion().front();
  if (!llvm::hasSingleElement(body.without_terminator())) return false;
  mlir::Operation& converting = body.front();
  return isDeviceFloatConversion(&converting) &&
         converting.getOperand(0) == body.getArgument(0) &&
         body.getTerminator()->getOperands() ==
             mlir::ValueRange(converting.getResult(0));
}

auto scheduler::isWideningFloatConversion(mlir::Operation* op) -> bool {
  return isFloatConversion(op) &&
         mlir::isa<mlir::arith::ExtFOp>(mlir::cast<mlir::linalg::GenericOp>(op)
                                            .getRegion()
                                            .front()
                                            .front());
}

auto scheduler::markWideConversionSlots(mlir::func::FuncOp func)
    -> mlir::LogicalResult {
  // The FIFO slot on the wide side of each conversion. Each end of the slot is
  // lowered as two vectors: the conversion lowers its own end, and a transfer
  // to or from memory is marked so it lowers the other. Any other end would
  // move the slot as one vector, so it is rejected here, while both ends are
  // still in sight.
  bool failed = false;
  func.walk([&](mlir::linalg::GenericOp generic) {
    if (!isFloatConversion(generic)) {
      // A conversion that is not one of a whole tensor, as the device converts,
      // is not one this lowers.
      generic.getRegion().walk([&](mlir::Operation* inner) {
        if (isDeviceFloatConversion(inner)) {
          inner->emitError(
              "a conversion between f16 and f32 must be the only operation of "
              "a lane-wise linalg.generic on tensors, converting its one "
              "input");
          failed = true;
        }
      });
      return;
    }

    const bool widens = isWideningFloatConversion(generic);
    mlir::Value slot;
    if (widens) {
      auto result = generic->getResult(0);
      auto write =
          result.hasOneUse()
              ? mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(*result.user_begin())
              : nullptr;
      if (!write) {
        generic.emitError(
            "the widened result of a dl16tofp32 conversion must be written to "
            "a FIFO and nothing else");
        failed = true;
        return;
      }
      slot = write.getFifoSlot();
    } else {
      auto input = generic.getDpsInputs()[0];
      auto read = input.getDefiningOp<mlir::ktdf::ReadFromFifoOp>();
      if (!read || read.getSplat() || !input.hasOneUse()) {
        generic.emitError(
            "the input of a fp32todl16 conversion must be read from a FIFO and "
            "used by nothing else");
        failed = true;
        return;
      }
      slot = read.getFifoSlot();
    }

    for (auto* user : slot.getUsers()) {
      auto transfer = mlir::dyn_cast<mlir::ktdf::DataTransferOp>(user);
      if (!transfer) {
        // The conversion's own end, or the other end in another compute unit.
        if (auto write = mlir::dyn_cast<mlir::ktdf::WriteToFifoOp>(user);
            write &&
            isWideningFloatConversion(write.getData().getDefiningOp())) {
          continue;
        }
        if (auto read = mlir::dyn_cast<mlir::ktdf::ReadFromFifoOp>(user);
            read &&
            llvm::all_of(read->getUsers(),
                         [](mlir::Operation* use) {
                           return isFloatConversion(use) &&
                                  !isWideningFloatConversion(use);
                         }) &&
            !read->use_empty()) {
          continue;
        }
        user->emitError(
                "the f32 side of a conversion between f16 and f32 moves as two "
                "vectors, so the other end of its FIFO must be a transfer to "
                "or from memory or another such conversion")
                .attachNote(generic.getLoc())
            << "the conversion";
        failed = true;
        continue;
      }
      if (transfer.getTransferMode() != mlir::ktdf::TransferMode::Copy) {
        transfer.emitError(
                    "the f32 side of a conversion between f16 and f32 moves as "
                    "two vectors, which a splat or pad transfer does not")
                .attachNote(generic.getLoc())
            << "the conversion";
        failed = true;
        continue;
      }
      transfer->setAttr(kVectorsPerSlotAttrName,
                        mlir::IntegerAttr::get(
                            mlir::IntegerType::get(func.getContext(), 64), 2));
    }
  });
  return mlir::failure(failed);
}

auto scheduler::getFifoVectorType(mlir::Operation* transfer,
                                  mlir::VectorType vector_type)
    -> mlir::VectorType {
  const auto vectors =
      transfer->getAttrOfType<mlir::IntegerAttr>(kVectorsPerSlotAttrName);
  if (!vectors || vectors.getInt() <= 1 ||
      vector_type.getNumElements() % vectors.getInt() != 0) {
    return vector_type;
  }
  return mlir::VectorType::get(
      {vector_type.getNumElements() / vectors.getInt()},
      vector_type.getElementType());
}
