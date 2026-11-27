//===-- ProgramModules.cpp --------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Utils/ProgramModules.h"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>

namespace scheduler {

bool isScheduled(mlir::ModuleOp module) {
  return module->hasAttr(kScheduledAttrName);
}

void markScheduled(mlir::ModuleOp module) {
  module->setAttr(kScheduledAttrName, mlir::UnitAttr::get(module.getContext()));
}

bool isOrchestratorModule(mlir::ModuleOp child) {
  return !child.getSymName().has_value();
}

mlir::ModuleOp getTopLevelChildModule(mlir::Operation* op) {
  auto self = mlir::dyn_cast<mlir::ModuleOp>(op);
  for (auto module = self ? self : op->getParentOfType<mlir::ModuleOp>();
       module; module = module->getParentOfType<mlir::ModuleOp>()) {
    auto parent = module->getParentOfType<mlir::ModuleOp>();
    if (parent && !parent->getParentOp()) return module;
  }
  return nullptr;
}

bool isInScheduledProgram(mlir::Operation* op) {
  mlir::ModuleOp child = getTopLevelChildModule(op);
  return child && isScheduled(child);
}

ModuleKind classifyModule(mlir::ModuleOp module) {
  if (isOrchestratorModule(module)) return ModuleKind::Orchestration;
  return isScheduled(module) ? ModuleKind::Scheduled : ModuleKind::New;
}

}  // namespace scheduler
