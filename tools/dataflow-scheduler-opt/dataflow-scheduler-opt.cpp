//===-- dataflow-scheduler-opt.cpp ------------------------------*- c++ -*-===//
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
// This file implements a canonical MLIR opt driver for testing purposes.
//
//===----------------------------------------------------------------------===//

#include <llvm/Support/CommandLine.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllExtensions.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Pass/PassRegistry.h>
#include <mlir/Tools/mlir-opt/MlirOptMain.h>

#include "dataflow-scheduler/Pipeline.h"
#include "dataflow-scheduler/RegisterEverything.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"

using namespace mlir;

void registerPassPipelinesForScheduler() {
  static llvm::cl::opt<std::string> splitDFIROutputDir(
      "split-dfir-output-dir",
      llvm::cl::desc("Output directory for split DFIR files produced by "
                     "-kEmitDFIR (default: same directory as input file)"),
      llvm::cl::init(""));

  static llvm::cl::opt<std::string> schedulerDeviceFilename(
      "device", llvm::cl::desc("Device architecture filename"),
      llvm::cl::value_desc("filename"), llvm::cl::init(""));

  mlir::PassPipelineRegistration<>(
      "kEmitDFIR", "Emit DataflowIR", [&](mlir::OpPassManager& pm) {
        scheduler::EnsureDeviceDeclarationPassOptions options;
        options.deviceFileName = schedulerDeviceFilename.getValue();
        pm.addPass(scheduler::createEnsureDeviceDeclarationPass(options));
        
        scheduler::buildKTDPToDFIRPipeline(
            pm, scheduler::SchedulerExtContext::dummyContext(),
            splitDFIROutputDir);
      });
}

auto main(int argc, char** argv) -> int {
  registerAllPasses();
  scheduler::registerAllPasses();
  registerPassPipelinesForScheduler();

  DialectRegistry registry;
  registerAllDialects(registry);
  scheduler::registerDialects(registry);
  registerAllExtensions(registry);
  scheduler::registerExtensions(registry);

  return asMainReturnCode(MlirOptMain(
      argc, argv, "DataflowScheduler modular optimizer driver\n", registry));
}
