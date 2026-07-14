//===-- RegisterEverything.h ------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_C_REGISTER_EVERYTHING_H_
#define DATAFLOW_SCHEDULER_C_REGISTER_EVERYTHING_H_

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Registers all passes defined and used by the scheduler.
MLIR_CAPI_EXPORTED void dataflowSchedulerRegisterAllPasses();
/// Registers all dialects defined and used by the scheduler.
MLIR_CAPI_EXPORTED void dataflowSchedulerRegisterAllDialects(
    MlirDialectRegistry registry);
/// Registers all extensions provided and required by the scheduler.
MLIR_CAPI_EXPORTED void dataflowSchedulerRegisterAllExtensions(
    MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif  // DATAFLOW_SCHEDULER_C_REGISTER_EVERYTHING_H_
