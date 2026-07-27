//===-- Passes.td ------------------------------------------*- tablegen -*-===//
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
// This file declares all the ktdf dialect passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PASSES_TD
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def BroadcastPromotionPass : Pass<"broadcast-promotion", "mlir::ModuleOp"> {
  let summary = "Hoist invariant ktdf.data_transfer ops as sibling pipelines";
  let description = [{
    This pass identifies ktdf.data_transfer ops inside ktdf.stage bodies
    whose source and destination indices/sizes are invariant w.r.t. at
    least one enclosing scf.for induction variable, and hoists them
    outward as a sibling ktdf.pipeline placed before the invariant loop.

    Only broadcast-style hoists (single-transfer, hoist-up) are handled.
    Reduction hoists (paired pre-load/post-store) and FIFO-destination
    hoists (requiring co-hoisting of ktdf.fifo.allocate and consumer-side
    register-promotion rewrites) are handled by separate future passes.

    The pass runs a fixed-point loop: one hoist per iteration, re-scan
    after each hoist, exit when no candidate remains.
  }];
  let constructor = "::mlir::ktdf::createBroadcastPromotionPass()";
  let dependentDialects = ["mlir::ktdf::KTDFDialect", "mlir::scf::SCFDialect"];
}

def StageCoarseningPass : Pass<"stage-coarsening", "mlir::ModuleOp"> {
  let summary = "Perform stage coarsening transformation";
  let description = [{
    This pass implements a 4-stage stage coarsening algorithm to create
    hierarchical pipeline structures by sinking loops into stages.

    Following identification of stage groups (using loop distribution legality rules):
    - Stage 1a: Distribute and sink loops into pipeline
    - Stage 1b: Further sink into stages with single parent loop (expand buffers)
    - Stage 2: Discover and correct illegal structures iteratively
    - Stage 3: Correct stage dependencies that cross pipeline boundaries

    The transformation creates nested pipelines where:
    - Outer pipelines operate at tile granularity
    - Inner pipelines operate at hardware execution granularity

    This enables better hardware mapping and reduced synchronization overhead.
  }];
  let constructor = "::mlir::ktdf::createStageCoarseningPass()";
  let options = [
    Option<"loopNestDepth", "loop-nest-depth", "unsigned",
           /*default=*/"2",
           "Number of perfectly nested loops to consider for sinking">
  ];
  let dependentDialects = ["mlir::ktdf::KTDFDialect"];
}

def SubsumeLinearizeIndexPass : Pass<"subsume-linearize-index", "mlir::ModuleOp"> {
  let summary = "Fold ktdf.tiling.linearize_index into ktdf.data_transfer affine maps";
  let description = [{
    For each ktdf.data_transfer, any index operand that is produced by a
    ktdf.tiling.linearize_index with all-constant strides is replaced by an
    affine.apply encoding the weighted-sum expression. The affine.apply is
    then composed into the transfer's source_map/dest_map via
    fullyComposeAffineMapAndOperands. Dead tiling.linearize_index ops are erased.
  }];
  let constructor = "::mlir::ktdf::createSubsumeLinearizeIndexPass()";
  let dependentDialects = [
      "mlir::ktdf::KTDFDialect", 
      "mlir::affine::AffineDialect",
      "mlir::arith::ArithDialect"];
}

def TileSizeSelectionPass : Pass<"tile-size-selection", "mlir::ModuleOp"> {
  let summary = "Resolve ktdf.tiling.reserve_size placeholders to concrete constants";
  let description = [{
    This pass analyzes each ktdf.tiling.reserve_size op to discover which scf.for
    loops it is used to tile and uses the loop trip count and other information
    (eg. in future L1 usage) to replace the placeholder with an actual tile size.

    This pass requires a SchedulerExtContext for agent-driven tile size optimization.
  }];
  let dependentDialects = [
      "mlir::ktdf::KTDFDialect",
      "mlir::scf::SCFDialect",
      "mlir::arith::ArithDialect"];
}

#endif // DATAFLOW_SCHEDULER_DIALECT_KTDF_TRANSFORMS_PASSES_TD
