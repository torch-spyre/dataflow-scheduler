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

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"

#include <mlir/Pass/AnalysisManager.h>

#include "dataflow-scheduler/Dialect/KTDFArch/Analysis/DeviceManager.h"
#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

using namespace scheduler::arch_view;

namespace {

auto getComputeKind(const ResourceKinds& resource_kinds) -> mlir::Attribute {
  mlir::ktdf_arch::Resource result;
  for (auto [_, resource] : resource_kinds) {
    if (!resource.getFeature<mlir::ktdf_arch::feature::Compute>()) {
      continue;
    }

    if (result) {
      auto diag = resource->emitError("found second compute resource");
      diag.attachNote(result->getLoc()) << "previous resource is here";
      return nullptr;
    }

    result = resource;
  }

  return result.getKind();
}

}  // namespace

ResourceKinds::ResourceKinds(const mlir::ktdf_arch::Device& device)
    : DeviceView(device) {
  // Visit all Resources in the device.
  device.getBodyRegion().walk(
      [&](mlir::ktdf_arch::Resource resource) -> mlir::WalkResult {
        // Store this exemplar for its kind.
        auto [_, inserted] = exemplars_.insert({resource.getKind(), resource});

        if (resource->hasTrait<mlir::ktdf_arch::IsSubgraph>() && !inserted) {
          // We have already visited this kind of subgraph, there is no need to
          // traverse it again.
          return mlir::WalkResult::skip();
        }

        return mlir::WalkResult::advance();
      });

  // FIXME: Remove this work-around in favor of a real resource allocation pass.
  compute_ = ::getComputeKind(*this);
}
