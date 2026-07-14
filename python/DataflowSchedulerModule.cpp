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

#include <mlir/Bindings/Python/NanobindAdaptors.h>

#include "dataflow-scheduler-c/RegisterEverything.h"

namespace nb = nanobind;

NB_MODULE(_dataflow_scheduler, scheduler) {
  scheduler.doc() = "mlir_scheduler main python extension";

  scheduler.def(
      "register_all_passes", []() { dataflowSchedulerRegisterAllPasses(); },
      "Registers all passes defined and used by the scheduler.");
  scheduler.def(
      "register_all_dialects",
      [](MlirDialectRegistry registry) {
        dataflowSchedulerRegisterAllDialects(registry);
      },
      nb::arg("registry"),
      "Registers all dialects defined and used by the scheduler.");
  scheduler.def(
      "register_all_extensions",
      [](MlirDialectRegistry registry) {
        dataflowSchedulerRegisterAllExtensions(registry);
      },
      nb::arg("registry"),
      "Registers all extensions provided and required by the scheduler.");
}
