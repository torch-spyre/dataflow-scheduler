//===------------------------------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_UTILS_SCHEDULEREXTCONTEXT_H_
#define DATAFLOW_SCHEDULER_UTILS_SCHEDULEREXTCONTEXT_H_

#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinOps.h>

namespace scheduler {

// Type alias for resource representation
using ResourceType = mlir::Attribute;

struct SchedulerExtContext {
  SchedulerExtContext();
  virtual ~SchedulerExtContext();

  /// @brief Get a default initialized context for scheduler.
  /// The dummy context is intended for use with passes and analyses that
  /// require default constructors.
  static const SchedulerExtContext& dummyContext();

  virtual bool isDummy() const { return false; }
};

/// @brief Construct a DummySchedulerExtContext. In general prefer
/// SchedulerExtContext::dummyContext.
struct DummySchedulerExtContext : SchedulerExtContext {
  DummySchedulerExtContext() : SchedulerExtContext() {}
  bool isDummy() const override { return true; }
};

}  // namespace scheduler

#endif  // DATAFLOW_SCHEDULER_UTILS_SCHEDULEREXTCONTEXT_H_
