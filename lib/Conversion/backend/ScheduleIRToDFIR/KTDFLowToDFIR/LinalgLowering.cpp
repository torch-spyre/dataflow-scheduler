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
// Lowering compute operations (linalg.generic, arith, math) into DFIR.
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/LinalgLowering.h"

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Dialect/VectorChain/VectorChain.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"

using namespace scheduler;

namespace {

/// Pattern to lower linalg.generic compute operations
struct LowerLinalgGenericPattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  LowerLinalgGenericPattern(mlir::MLIRContext* context,
                            arch_view::ResourceKinds& resource_kinds)
      : OpRewritePattern(context), resource_kinds_(resource_kinds) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::linalg::GenericOp generic_op,
      mlir::PatternRewriter& rewriter) const override {
    if (!generic_op.hasPureTensorSemantics() ||
        generic_op.getNumResults() != 1) {
      return mlir::failure();
    }

    mlir::Block& body = generic_op.getRegion().front();
    auto yield_op = mlir::dyn_cast<mlir::linalg::YieldOp>(body.getTerminator());
    if (!yield_op || yield_op.getNumOperands() != 1) {
      return mlir::failure();
    }

    // Replace block arguments with generic inputs
    unsigned num_inputs = generic_op.getNumDpsInputs();
    for (auto [block_arg, input] :
         llvm::zip(body.getArguments().take_front(num_inputs),
                   generic_op.getDpsInputs())) {
      block_arg.replaceAllUsesWith(input);
    }

    // Identity affine map used as op_specific_map for binary ops.
    mlir::AffineMap identity_map =
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext());

    // Collect compute operations to replace
    llvm::SmallVector<mlir::Operation*> ops_to_lower;
    for (mlir::Operation& op : body.without_terminator()) {
      ops_to_lower.push_back(&op);
    }

    // Process and lower compute operations via visitors
    rewriter.setInsertionPoint(generic_op);
    for (mlir::Operation* op : ops_to_lower) {
      mlir::LogicalResult result =
          mlir::TypeSwitch<mlir::Operation*, mlir::LogicalResult>(op)
              // arith.mulf %lhs, %rhs -> vectorchain.binary {binary_op = mul}
              .Case<mlir::arith::MulFOp>([&](mlir::arith::MulFOp mulf_op) {
                return lowerMulFOp(mulf_op, rewriter, identity_map);
              })
              // arith.addf %lhs, %rhs -> vectorchain.binary {binary_op = add}
              .Case<mlir::arith::AddFOp>([&](mlir::arith::AddFOp addf_op) {
                return lowerAddFOp(addf_op, rewriter, identity_map);
              })
              // arith.subf %lhs, %rhs -> vectorchain.binary {binary_op = sub}
              .Case<mlir::arith::SubFOp>([&](mlir::arith::SubFOp subf_op) {
                return lowerSubFOp(subf_op, rewriter, identity_map);
              })
              .Default([](mlir::Operation* unknown_op) {
                return unknown_op->emitError(
                    "unsupported operation type in linalg.generic body");
              });

      if (mlir::failed(result)) return mlir::failure();
    }

    // Replace the generic op with the yield operand
    rewriter.replaceOp(generic_op, yield_op.getOperand(0));
    return mlir::success();
  }

 private:
  arch_view::ResourceKinds& resource_kinds_;

  // arith.mulf visitor: lowers to vectorchain.binary {binary_op = mul}
  mlir::LogicalResult lowerMulFOp(mlir::arith::MulFOp mulf_op,
                                  mlir::PatternRewriter& rewriter,
                                  mlir::AffineMap identity_map) const {
    auto vector_type =
        getFlattenedVectorType(mulf_op.getLhs().getType(), resource_kinds_);
    if (!vector_type) return mlir::failure();

    auto binary_op = mlir::vectorchain::BinaryOp::create(
        rewriter, mulf_op.getLoc(), vector_type, mulf_op.getLhs(),
        mulf_op.getRhs(),
        /*mask=*/nullptr, /*dbgName=*/nullptr,
        mlir::vectorchain::VectorChainBinaryOperator::mul, identity_map);

    rewriter.replaceOp(mulf_op, binary_op.getData());
    return mlir::success();
  }

  // arith.addf visitor: lowers to vectorchain.binary {binary_op = add}
  mlir::LogicalResult lowerAddFOp(mlir::arith::AddFOp addf_op,
                                  mlir::PatternRewriter& rewriter,
                                  mlir::AffineMap identity_map) const {
    auto vector_type =
        getFlattenedVectorType(addf_op.getLhs().getType(), resource_kinds_);
    if (!vector_type) return mlir::failure();

    auto binary_op = mlir::vectorchain::BinaryOp::create(
        rewriter, addf_op.getLoc(), vector_type, addf_op.getLhs(),
        addf_op.getRhs(),
        /*mask=*/nullptr, /*dbgName=*/nullptr,
        mlir::vectorchain::VectorChainBinaryOperator::add, identity_map);

    rewriter.replaceOp(addf_op, binary_op.getData());
    return mlir::success();
  }

  // arith.subf visitor: lowers to vectorchain.binary {binary_op = sub}
  mlir::LogicalResult lowerSubFOp(mlir::arith::SubFOp subf_op,
                                  mlir::PatternRewriter& rewriter,
                                  mlir::AffineMap identity_map) const {
    auto vector_type =
        getFlattenedVectorType(subf_op.getLhs().getType(), resource_kinds_);
    if (!vector_type) return mlir::failure();

    auto binary_op = mlir::vectorchain::BinaryOp::create(
        rewriter, subf_op.getLoc(), vector_type, subf_op.getLhs(),
        subf_op.getRhs(),
        /*mask=*/nullptr, /*dbgName=*/nullptr,
        mlir::vectorchain::VectorChainBinaryOperator::sub, identity_map);

    rewriter.replaceOp(subf_op, binary_op.getData());
    return mlir::success();
  }
};

}  // namespace

void scheduler::populateLinalgLoweringPatterns(
    mlir::RewritePatternSet& patterns,
    arch_view::ResourceKinds& resource_kinds) {
  patterns.add<LowerLinalgGenericPattern>(patterns.getContext(),
                                          resource_kinds);
}
