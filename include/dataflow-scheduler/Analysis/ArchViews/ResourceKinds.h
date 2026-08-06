//===-- ResourceKinds.h -----------------------------------------*- c++ -*-===//
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

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_RESOURCEKINDS_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_RESOURCEKINDS_H_

#include <mlir/IR/Attributes.h>
#include <mlir/Pass/AnalysisManager.h>
#include <mlir/Support/TypeID.h>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

namespace scheduler::arch_view {

class ResourceKinds : public mlir::ktdf_arch::DeviceView {
  using map_type = llvm::DenseMap<mlir::Attribute, mlir::ktdf_arch::Resource>;

 public:
  explicit ResourceKinds(const mlir::ktdf_arch::Device& device);

  template <class ResourceType = mlir::ktdf_arch::Resource>
  [[nodiscard]] auto getResource(mlir::Attribute kind) const -> ResourceType {
    if (auto entry = exemplars_.lookup(kind); entry) {
      return mlir::dyn_cast<ResourceType>(entry.getOperation());
    }
    return nullptr;
  }

  template <class PropertyAttr>
  [[nodiscard]] auto getProperty(mlir::Attribute kind) const -> PropertyAttr {
    auto resource = getResource(kind);
    if (!resource) {
      return nullptr;
    }

    return resource.getProperty<PropertyAttr>();
  }

  template <class FeatureAttr>
  [[nodiscard]] auto getFeature(mlir::Attribute kind) const -> FeatureAttr {
    auto resource = getResource(kind);
    if (!resource) {
      return nullptr;
    }

    return resource.getFeature<FeatureAttr>();
  }

  // FIXME: Remove this work-around in favor of a real resource allocation pass.
  [[nodiscard]] auto getComputeKind() const -> mlir::Attribute {
    return compute_;
  }

  //===--------------------------------------------------------------------===//
  // Container Interface
  //===--------------------------------------------------------------------===//

  using value_type = map_type::value_type;
  using size_type = map_type::size_type;
  using iterator = map_type::const_iterator;

  [[nodiscard]] auto empty() const -> bool { return exemplars_.empty(); }
  [[nodiscard]] auto size() const -> size_type { return exemplars_.size(); }

  [[nodiscard]] auto begin() const -> iterator { return exemplars_.begin(); }
  [[nodiscard]] auto end() const -> iterator { return exemplars_.end(); }

 private:
  map_type exemplars_;
  mlir::Attribute compute_;
};

}  // namespace scheduler::arch_view

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_RESOURCEKINDS_H_
