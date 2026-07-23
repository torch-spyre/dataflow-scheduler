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

#include <filesystem>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"
#include "dataflow-scheduler/Transforms/Passes.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"

namespace scheduler {
#define GEN_PASS_DEF_ENSUREDEVICEDECLARATIONPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace scheduler;

namespace {

auto getDeviceName(const std::filesystem::path& path,
                   mlir::MLIRContext* context) -> mlir::FailureOr<std::string> {
  std::string error_message;
  auto maybe_file = mlir::openInputFile(path.native(), &error_message);
  if (!maybe_file) {
    return mlir::emitError(mlir::UnknownLoc::get(context),
                           "unable to import device: ")
           << error_message;
  }

  auto source_mgr = std::make_shared<llvm::SourceMgr>();
  source_mgr->AddNewSourceBuffer(std::move(maybe_file), {});
  mlir::SourceMgrDiagnosticHandler import_handler(*source_mgr, context);

  mlir::ParserConfig config(context, true);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(*source_mgr, config);
  if (!module) {
    // Diagnostic was already emitted.
    return mlir::failure();
  }

  auto devices = module->getOps<mlir::ktdf_arch::DeviceOp>();
  if (devices.empty()) {
    return mlir::emitError(module->getLoc(), "no device in import file");
  }
  if (std::next(devices.begin()) != devices.end()) {
    return mlir::emitError(module->getLoc(), "multiple devices in import file");
  }

  return (*devices.begin()).getName().str();
}

mlir::LogicalResult injectDevice(mlir::ModuleOp module,
                                 std::string device_filename,
                                 std::string device_name) {
  auto devices = module.getOps<mlir::ktdf_arch::DeviceOp>();
  if (!devices.empty()) {
    return mlir::success();
  }
  if (device_filename.empty()) {
    return mlir::emitError(module.getLoc())
           << "no 'ktdf_arch.device' is present in the module and no device "
              "file name is provided to ensure-device-declaration pass";
  }

  const std::filesystem::path filename(device_filename.c_str());
  const auto import_path = std::filesystem::absolute(filename);
  if (device_name.empty()) {
    auto maybe_device_name = getDeviceName(import_path, module.getContext());
    if (mlir::failed(maybe_device_name)) {
      return mlir::failure();
    }
    device_name = maybe_device_name.value();
  }
  mlir::OpBuilder builder(module.getBody(), module.getBody()->begin());
  mlir::ktdf_arch::DeviceOp::create(builder, builder.getUnknownLoc(),
                                    device_name, import_path.native());
  return mlir::success();
}

struct EnsureDeviceDeclarationPass
    : public impl::EnsureDeviceDeclarationPassBase<
          EnsureDeviceDeclarationPass> {
  using EnsureDeviceDeclarationPassBase::EnsureDeviceDeclarationPassBase;
  void runOnOperation() override {
    if (mlir::failed(
            injectDevice(getOperation(), deviceFileName, deviceName))) {
      signalPassFailure();
      return;
    }
    auto& device_manager = getAnalysis<mlir::ktdf_arch::DeviceManager>();
    std::ignore = device_manager.getOrImportDevice();
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> scheduler::createEnsureDeviceDeclarationPass() {
  return std::make_unique<EnsureDeviceDeclarationPass>();
}

std::unique_ptr<mlir::Pass> scheduler::createEnsureDeviceDeclarationPass(
    EnsureDeviceDeclarationPassOptions options) {
  return std::make_unique<EnsureDeviceDeclarationPass>(options);
}
