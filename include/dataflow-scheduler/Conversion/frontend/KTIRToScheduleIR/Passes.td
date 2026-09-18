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
// This file declares all the scheduler passes.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_CONVERSION_KTIRTOSCHEDULEIR_PASSES_TD
#define DATAFLOW_SCHEDULER_CONVERSION_KTIRTOSCHEDULEIR_PASSES_TD

include "mlir/Pass/PassBase.td"

def ComputeGroupExtractionPass : Pass<"compute-group-extraction", "mlir::ModuleOp"> {
  let summary = "Extract main and compute group modules from a module";
  let description = [{
    This pass extracts the main and compute groups from the initial module.
  }];
  let constructor = "::scheduler::createComputeGroupExtractionPass()";
}

def ConstructThreeStagePipelinePass : Pass<"construct-three-stage-pipeline", "mlir::ModuleOp"> {
  let summary = "Construct a three-stage pipeline";
  let description = [{
    This pass constructs a three-stage pipeline.
  }];
  let constructor = "::scheduler::createConstructThreeStagePipelinePass()";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::ktdf::KTDFDialect",
    "mlir::scf::SCFDialect"
  ];
}

def KTIRLegalityCheckPass : Pass<"ktir-legality-check", "mlir::ModuleOp"> {
  let summary = "Reject KTIR constructs V1 cannot lower yet";
  let description = [{
    Fails fast with a diagnostic when the input contains V1-unsupported
    constructs: scf.for with loop-carried args, scf.while, compute ops outside
    {add, mul, sub}, or the ktdp distributed-memory-view / indirect-access-tile
    ops. Mutates nothing; legal input passes through.
  }];
  let constructor = "::scheduler::createKTIRLegalityCheckPass()";
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::linalg::LinalgDialect"
  ];
}

def FuseLinalgPass : Pass<"fuse-linalg", ""> {
  let summary = "Fuses `linalg.generic` operations";
  let description = [{
    Performs mapping-aware fusion of `linalg.generic` operations. Currently,
    only elementiwse operations are considered for fusion.

    Mapping awareness means that `linalg.generic` operations which are mapped,
    or imply a mapping based on their contents, will only be fused with other
    ops that have compatible mappings. Mappings are compatible if they refer to
    the exact same resources.

    As a result of this pass, implied mappings will be propagated upwards
    directly onto the `linalg.generic` operations that have them.
  }];
  let dependentDialects = [
    "mlir::ktdf_arch::KTDFArchDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

#endif // DATAFLOW_SCHEDULER_CONVERSION_KTIRTOSCHEDULEIR_PASSES_TD
