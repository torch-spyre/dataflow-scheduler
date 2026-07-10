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
/// Parallel operation lowering for KTDFLowToDFIR pass
///
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/ParallelLowering.h"

#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Agen/Utils.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Dataflow/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Transforms/Utils/Utils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "ktdflowering-parallel-lowering"

namespace scheduler {

namespace {

/// Gets the coefficient of loop_iv from a vector load/store operation
/// Extracts affine maps and uses constructIteratorCoeffDict to get the
/// coefficient
static llvm::FailureOr<int64_t> getLayoutMapCoeff(mlir::Operation* vector_op,
                                                  mlir::BlockArgument loop_iv) {
  // Get the memref operand and the map operands
  mlir::Value memref;
  mlir::AffineMap subscripts_map;
  mlir::AffineMap transfer_order;
  llvm::SmallVector<mlir::Value, 8> map_operands;

  if (auto load_op = mlir::dyn_cast<mlir::agen::VectorLoadOp>(vector_op)) {
    memref = load_op.getMemRef();
    subscripts_map = load_op.getAffineMap();
    transfer_order = load_op.getLoadOrder();
    map_operands.append(load_op.getMapOperands().begin(),
                        load_op.getMapOperands().end());
  } else if (auto store_op =
                 mlir::dyn_cast<mlir::agen::VectorStoreOp>(vector_op)) {
    memref = store_op.getMemRef();
    subscripts_map = store_op.getAffineMap();
    transfer_order = store_op.getStoreOrder();
    map_operands.append(store_op.getMapOperands().begin(),
                        store_op.getMapOperands().end());
  } else {
    return vector_op->emitError("Unexpected operation type");
  }

  // Find the get_logical_memory_view operation that defines the memref
  auto mem_view_op =
      memref.getDefiningOp<mlir::dataflow::GetLogicalMemoryViewOp>();
  if (!mem_view_op) {
    return vector_op->emitError(
        "Memref must be defined by dataflow.get_logical_memory_view");
  }

  // Get the layout_map from get_logical_memory_view
  mlir::AffineMap mem_view_layout_map = mem_view_op.getLayoutMap();

  // Call constructIteratorCoeffDict to get the coefficient dictionary
  auto coeff_dict = mlir::agen::constructIteratorCoeffDict(
      subscripts_map, transfer_order, mem_view_layout_map, map_operands);

  // Look up the coefficient for loop_iv
  auto it = coeff_dict.find(loop_iv);
  if (it == coeff_dict.end()) {
    return vector_op->emitError("loop_iv not found in coefficient dictionary");
  }

  return it->second;
}

/// Updates the start address of a vector load/store operation.
/// Gets the layout map coefficient of loop_iv, then replaces the start address
/// of the get_logical_memory_view with start_addr + offset, where offset is a
/// query_map that maps each unit to: 0 if corelet=0, or layout_map_coeff *
/// parallel_op_ub / num_instances if corelet=1
static mlir::LogicalResult updateStartAddress(mlir::Operation* vector_op,
                                              mlir::BlockArgument loop_iv,
                                              int64_t parallel_op_ub,
                                              int64_t num_instances) {
  // Get the coefficient of loop_iv from getLayoutMapCoeff
  auto coeff_result = getLayoutMapCoeff(vector_op, loop_iv);
  if (mlir::failed(coeff_result)) {
    return mlir::failure();
  }
  int64_t layout_map_coeff = *coeff_result;

  // Get the memref operand from the vector load/store
  mlir::Value memref;
  if (auto load_op = mlir::dyn_cast<mlir::agen::VectorLoadOp>(vector_op)) {
    memref = load_op.getMemRef();
  } else if (auto store_op =
                 mlir::dyn_cast<mlir::agen::VectorStoreOp>(vector_op)) {
    memref = store_op.getMemRef();
  } else {
    return vector_op->emitError("Expected VectorLoadOp or VectorStoreOp");
  }

  // Find the get_logical_memory_view operation that defines the memref
  auto mem_view_op =
      memref.getDefiningOp<mlir::dataflow::GetLogicalMemoryViewOp>();
  if (!mem_view_op) {
    return vector_op->emitError(
        "Memref must be defined by dataflow.get_logical_memory_view");
  }

  // Find the enclosing program_unit to get its units
  auto program_unit =
      vector_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
  if (!program_unit) {
    return vector_op->emitError(
        "vector operation must be inside a program_unit");
  }

  mlir::Value start_addr = mem_view_op.getStartAddress();

  // Create the offset to the get_logical_memory_view: 0 for corelet 0 units,
  // layout_map_coeff * (parallel_op_ub / num_instances) for corelet 1 units
  // Reason: For corelet 1 units, the loop will correspond to the second half of
  // the iteration space. Since we want the loop to be normalized (i.e. have
  // lower bound 0), we must adjust the get_logical_memory_view by the number of
  // iterations in the first half of the iteration space (namely parallel_op_ub
  // / num_instances) times the coefficient of the parallel loop IV in the
  // layout map of mem_view_op.
  mlir::OpBuilder builder(mem_view_op);
  mlir::Location loc = mem_view_op.getLoc();

  assert(parallel_op_ub % num_instances == 0 &&
         "parallel_op_ub must be evenly divisible by num_instances");
  int64_t offset_value = layout_map_coeff * (parallel_op_ub / num_instances);

  // Build the query_map: map each program_unit operand to 0 or offset_value
  // based on corelet
  llvm::SmallVector<mlir::Value> keys;
  llvm::SmallVector<mlir::Value> values;

  for (mlir::Value pu_operand : program_unit.getUnits()) {
    auto get_unit = mlir::dyn_cast_or_null<mlir::dataflow::GetUnitOp>(
        pu_operand.getDefiningOp());
    assert(get_unit &&
           "program_unit operand must be defined by dataflow.get_unit");

    int corelet_id = mlir::dataflow::getCoreletId(get_unit);

    // Map to 0 if corelet=0, to offset_value if corelet=1, error otherwise
    int64_t mapped_offset;
    if (corelet_id == 0) {
      mapped_offset = 0;
    } else if (corelet_id == 1) {
      mapped_offset = offset_value;
    } else {
      return vector_op->emitError("corelet ID must be 0 or 1, got ")
             << corelet_id;
    }

    mlir::Value offset_const =
        mlir::arith::ConstantIndexOp::create(builder, loc, mapped_offset);

    keys.push_back(pu_operand);
    values.push_back(offset_const);
  }

  assert(!keys.empty() && "must have at least one key-value pair");

  // Create def_immutable_mapping and query_map
  auto map_op = mlir::uniform::DefImmutableMappingOp::create(
      builder, loc, builder.getIndexType(), keys, values);
  mlir::Value iter_arg = program_unit.getRegion().front().getArgument(0);
  auto query_op = mlir::uniform::QueryMapOp::create(
      builder, loc, builder.getIndexType(), map_op.getResult(), iter_arg);

  mlir::Value offset = query_op.getResult();

  // Create arith.addi to compute start_addr + offset
  mlir::Value new_start_addr =
      mlir::arith::AddIOp::create(builder, loc, start_addr, offset);

  mem_view_op.getStartAddressMutable().assign(new_start_addr);
  return mlir::success();
}

/// Computes the number of iterations per hardware instance for a parallel op.
/// Gets the initial upper bound from parallel op, asserts it's constant,
/// and returns parallel_op_ub / num_instances.
/// Remainder must be 0.
static llvm::FailureOr<int64_t> getNumIterationsPerHardwareInstance(
    mlir::ktdf::ParallelOp parallel_op) {
  // Get num_instances attribute from parallel op
  int64_t num_instances = parallel_op.getNumInstances();

  // Get the upper bound from parallel op (should be a single upper bound for
  // 1D)
  auto upper_bounds = parallel_op.getUpperBounds();
  if (upper_bounds.size() != 1) {
    parallel_op.emitError(
        "Expected exactly 1 upper bound for 1D parallel loop");
    return mlir::failure();
  }
  mlir::Value parallel_op_ub = upper_bounds[0];

  // Assert that the upper bound is a constant
  auto upper_bound_const =
      parallel_op_ub.getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (!upper_bound_const) {
    parallel_op.emitError("Upper bound must be a constant for lowering");
    return mlir::failure();
  }

  int64_t parallel_op_ub_val = upper_bound_const.value();

  // Check that parallel_op_ub divides evenly by num_instances
  if (parallel_op_ub_val % num_instances != 0) {
    parallel_op.emitError(
        "Upper bound must be evenly divisible by num_instances");
    return mlir::failure();
  }

  // Calculate and return iterations per instance
  return parallel_op_ub_val / num_instances;
}

}  // namespace

mlir::LogicalResult lowerParallelOps(mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::ktdf::ParallelOp> parallel_ops;
  // Collect all ktdf.parallel operations
  func.walk([&](mlir::ktdf::ParallelOp parallel_op) {
    parallel_ops.push_back(parallel_op);
  });

  if (parallel_ops.empty()) return mlir::success();
  int num_cores;
  if (mlir::failed(extractGridSize(func, num_cores))) {
    return mlir::failure();
  }

  // Lower each parallel operation
  for (auto parallel_op : parallel_ops) {
    mlir::OpBuilder builder(parallel_op);
    auto loc = parallel_op.getLoc();
    // Get the parallel operation's body and induction variables
    mlir::Region& parallel_region = parallel_op.getRegion();
    mlir::Block& parallel_body = parallel_region.front();

    // Assert that ktdf.parallel has exactly 2 block arguments (1D loop)
    // First arg is the iteration number, second is the instance ID
    if (parallel_body.getNumArguments() != 2) {
      return parallel_op.emitError(
          "Expect 1 dimensional ktdf.parallel (with exactly 2 block "
          "arguments)");
    }

    mlir::BlockArgument loop_iv = parallel_body.getArgument(0);
    mlir::BlockArgument instance_id = parallel_body.getArgument(1);
    assert(instance_id.use_empty() && "Instance ID should not have uses");

    if (!instance_id.use_empty()) {
      return parallel_op.emitError(
          "instance_id expected to have no uses for lowering "
          "to scf.for");
    }

    for (auto* user : loop_iv.getUsers()) {
      if (!mlir::isa<mlir::agen::VectorLoadOp, mlir::agen::VectorStoreOp>(
              user)) {
        return parallel_op.emitError(
            "loop_iv can only be used in agen.vector_load/store operations");
      }
    }

    // Get num_instances from parallel op
    int64_t num_instances = parallel_op.getNumInstances();

    // Get the initial upper bound value
    auto upper_bounds = parallel_op.getUpperBounds();
    assert(upper_bounds.size() == 1 &&
           "Expected exactly 1 upper bound for 1D parallel loop");
    mlir::Value parallel_op_ub_value = upper_bounds[0];
    auto upper_bound_const =
        parallel_op_ub_value.getDefiningOp<mlir::arith::ConstantIndexOp>();
    assert(upper_bound_const && "Upper bound must be a constant");
    int64_t parallel_op_ub = upper_bound_const.value();

    // Get the number of iterations per hardware instance
    // (parallel_op_ub / num_instances)
    auto num_iters_result = getNumIterationsPerHardwareInstance(parallel_op);
    if (mlir::failed(num_iters_result)) {
      return mlir::failure();
    }
    int64_t num_iters_per_instance = *num_iters_result;
    LDBG(1) << "ktdf.parallel " << parallel_op << " replaced with scf.for with "
            << num_iters_per_instance << " iterations";

    // Create constants for loop bounds: 0 to num_iters_per_instance with step 1
    mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
    mlir::Value c1 = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
    mlir::Value upper_bound = mlir::arith::ConstantIndexOp::create(
        builder, loc, num_iters_per_instance);

    // Create scf.for loop with bounds [0, num_iters_per_instance)
    auto for_op =
        mlir::scf::ForOp::create(builder, loc, c0, upper_bound, c1, {});
    mlir::Block& for_body = for_op.getRegion().front();
    mlir::BlockArgument for_iv = for_body.getArgument(0);

    // Replace all uses of the parallel loop_iv with the for loop IV
    loop_iv.replaceAllUsesWith(for_iv);

    // Move all operations except the terminator from parallel body to for body
    auto& parallel_ops_list = parallel_body.getOperations();
    auto& for_ops_list = for_body.getOperations();

    // Insert before the scf.yield terminator in the for loop
    for_ops_list.splice(for_body.getTerminator()->getIterator(),
                        parallel_ops_list, parallel_ops_list.begin(),
                        std::prev(parallel_ops_list.end()));

    // Update start addresses for all vector load/store operations
    for (auto* user : for_iv.getUsers()) {
      assert(
          (mlir::isa<mlir::agen::VectorLoadOp, mlir::agen::VectorStoreOp>(
              user)) &&
          "loop_iv should only be used in agen.vector_load/store operations");
      if (mlir::failed(updateStartAddress(user, for_iv, parallel_op_ub,
                                          num_instances))) {
        return mlir::failure();
      }
    }

    // Erase the parallel operation
    parallel_op.erase();
  }

  return mlir::success();
}

}  // namespace scheduler

// Made with Bob