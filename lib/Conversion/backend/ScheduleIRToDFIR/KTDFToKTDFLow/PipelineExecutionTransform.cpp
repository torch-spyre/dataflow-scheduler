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

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/PipelineExecutionTransform.h"

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Dominance.h"

#define DEBUG_TYPE "phase2-transformation"

using namespace scheduler;

mlir::LogicalResult scheduler::transformStagesToExecuteOn(
    mlir::ktdf::PipelineOp pipeline,
    const llvm::SmallVector<mlir::ktdf::StageOp, 8>& sorted_stages,
    const StageToUnitsMap& stage_to_units, mlir::OpBuilder& builder) {
  LLVM_DEBUG(llvm::dbgs() << "Step 9: Transform stages to execute_on\n");

  for (auto stage : sorted_stages) {
    auto units_it = stage_to_units.mapping.find(stage.getOperation());
    if (units_it == stage_to_units.mapping.end()) {
      return stage.emitError("No units found for stage");
    }

    auto stage_units = units_it->second;
    if (stage_units.empty()) {
      return stage.emitError("Stage has empty unit list");
    }

    LLVM_DEBUG(llvm::dbgs() << "  Creating execute_on for stage with "
                            << stage_units.size() << " units\n");

    // Create execute_on before the stage
    builder.setInsertionPoint(stage);
    auto stage_execute_on = mlir::ktdf_lowering::ExecuteOnOp::create(
        builder, stage.getLoc(), mlir::ValueRange(stage_units));

    // Ensure body block exists
    if (stage_execute_on.getBody().empty()) {
      builder.createBlock(&stage_execute_on.getBody());
    }

    // Move stage body into execute_on body
    mlir::Block* stage_body = stage_execute_on.getBodyBlock();
    mlir::Block* original_stage_body = stage.getBody();

    size_t num_ops = original_stage_body->getOperations().size();
    if (num_ops > 0) {
      stage_body->getOperations().splice(stage_body->end(),
                                         original_stage_body->getOperations());
    }

    LLVM_DEBUG(llvm::dbgs()
               << "  Moved " << num_ops << " operations into execute_on\n");

    // Replace stage with execute_on by moving execute_on before stage
    stage_execute_on->moveBefore(stage.getOperation());

    LLVM_DEBUG(llvm::dbgs()
               << "  Transformed stage to execute_on and replaced\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "  All " << sorted_stages.size()
                          << " stages transformed\n");
  return mlir::success();
}

mlir::LogicalResult scheduler::transformPipelineToExecuteOn(
    mlir::ktdf::PipelineOp pipeline,
    const llvm::SmallVector<mlir::ktdf::StageOp, 8>& sorted_stages,
    const StageToUnitsMap& stage_to_units, mlir::OpBuilder& builder) {
  LLVM_DEBUG(llvm::dbgs() << "Step 11: Transform pipeline to execute_on\n");

  llvm::SmallVector<mlir::Value, 16> all_units;
  llvm::DenseSet<mlir::Value> seen_units;

  // Collect all unique units from stages
  for (auto stage : sorted_stages) {
    auto units_it = stage_to_units.mapping.find(stage.getOperation());
    if (units_it != stage_to_units.mapping.end()) {
      for (auto unit : units_it->second) {
        if (!seen_units.count(unit)) {
          all_units.push_back(unit);
          seen_units.insert(unit);
        }
      }
    }
  }

  if (all_units.empty()) {
    return pipeline.emitError("No units found for pipeline execute_on");
  }

  LLVM_DEBUG(llvm::dbgs() << "  Creating pipeline execute_on with "
                          << all_units.size() << " units\n");

  // Create pipeline execute_on before the pipeline
  mlir::OpBuilder::InsertPoint insert_pt = builder.saveInsertionPoint();
  builder.setInsertionPoint(pipeline);

  auto pipeline_execute_on = mlir::ktdf_lowering::ExecuteOnOp::create(
      builder, pipeline.getLoc(), mlir::ValueRange(all_units));

  // Ensure body block exists
  if (pipeline_execute_on.getBody().empty()) {
    builder.createBlock(&pipeline_execute_on.getBody());
  }

  // Move all operations from pipeline body into execute_on body
  mlir::Block* pipeline_body = pipeline.getBody();
  mlir::Block* exec_body = pipeline_execute_on.getBodyBlock();

  llvm::SmallVector<mlir::ktdf::PrivateOp, 4> private_ops;
  for (auto& op : llvm::make_early_inc_range(pipeline_body->getOperations())) {
    if (auto private_op = mlir::dyn_cast<mlir::ktdf::PrivateOp>(&op)) {
      private_ops.push_back(private_op);
    }
  }

  // Inline ktdf.private operations and move their non-terminator ops
  for (auto private_op : private_ops) {
    LLVM_DEBUG(llvm::dbgs() << "  Inlining ktdf.private operations\n");

    mlir::Block* private_body = private_op.getBody();
    auto& ops = private_body->getOperations();
    if (!ops.empty()) {
      exec_body->getOperations().splice(
          exec_body->end(), ops, ops.begin(),
          std::prev(ops.end()));  // Move all except terminator
    }

    // Replace uses of private_op results with the values yielded by
    // private_yield
    auto terminator = private_body->getTerminator();
    if (auto yield_op = mlir::dyn_cast<mlir::ktdf::PrivateYieldOp>(terminator)) {
      for (size_t i = 0; i < private_op.getNumResults(); ++i) {
        private_op.getResult(i).replaceAllUsesWith(yield_op.getOperand(i));
      }
    }

    private_op.erase();
  }

  // Splice remaining (non-private) operations
  if (!pipeline_body->empty()) {
    exec_body->getOperations().splice(exec_body->end(),
                                      pipeline_body->getOperations());
  }

  // Replace pipeline with execute_on
  pipeline_execute_on->moveBefore(pipeline.getOperation());
  pipeline.erase();

  builder.restoreInsertionPoint(insert_pt);

  LLVM_DEBUG(llvm::dbgs() << "  Pipeline transformed to execute_on\n");
  return mlir::success();
}
