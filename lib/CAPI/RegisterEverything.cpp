//===-- RegisterEverything.cpp ----------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler-c/RegisterEverything.h"

#include <mlir/CAPI/IR.h>

#include "dataflow-scheduler/RegisterEverything.h"

void dataflowSchedulerRegisterAllPasses() { scheduler::registerAllPasses(); }

void dataflowSchedulerRegisterAllDialects(MlirDialectRegistry registry) {
  scheduler::registerAllDialects(*unwrap(registry));
}

void dataflowSchedulerRegisterAllExtensions(MlirDialectRegistry registry) {
  scheduler::registerAllExtensions(*unwrap(registry));
}
