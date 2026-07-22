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

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

namespace scheduler {
#define GEN_PASS_DEF_ENSUREDEVICEDECLARATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace scheduler;

namespace {

struct EnsureDeviceDeclarationPass
    : public impl::EnsureDeviceDeclarationPassBase<
          EnsureDeviceDeclarationPass> {
  using EnsureDeviceDeclarationPassBase::EnsureDeviceDeclarationPassBase;
  void runOnOperation() override {}
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createEnsureDeviceDeclarationPass() {
  return std::make_unique<EnsureDeviceDeclarationPass>();
}

std::unique_ptr<mlir::Pass> scheduler::createEnsureDeviceDeclarationPass(
    EnsureDeviceDeclarationPassOptions options) {
  return std::make_unique<EnsureDeviceDeclarationPass>(options);
}
