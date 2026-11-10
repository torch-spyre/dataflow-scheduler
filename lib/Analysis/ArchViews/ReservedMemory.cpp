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

#include "dataflow-scheduler/Analysis/ArchViews/ReservedMemory.h"

#include <algorithm>
#include <cstdlib>

#include "ktir/Dialect/KTDP/KTDPAttrs.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

using namespace scheduler::arch_view;

auto scheduler::arch_view::getFrontendCoreLocalBytes(std::size_t size_in_bytes,
                                                     double frac_avail)
    -> std::size_t {
  assert(frac_avail >= 0.0 && frac_avail <= 1.0 &&
         "DXP_LX_FRAC_AVAIL must be in [0, 1]");
  // The program and debug data at the top are not split.
  if (size_in_bytes <= kCoreLocalProgramDebugBytes) {
    return 0;
  }
  const std::size_t splittable = size_in_bytes - kCoreLocalProgramDebugBytes;
  // Truncated, then rounded up to the allocation granularity, exactly as the
  // front end does, so both land on the same byte.
  const auto frontend = static_cast<std::size_t>(
      static_cast<double>(splittable) * (1.0 - frac_avail));
  const std::size_t granule = kCoreLocalAllocationGranularityBytes;
  return (frontend + granule - 1) / granule * granule;
}

auto scheduler::arch_view::getFrontendFracAvail() -> double {
  const char* value = std::getenv(kFrontendFracAvailEnv);
  if (!value) {
    return kDefaultFrontendFracAvail;
  }
  double frac;
  if (llvm::StringRef(value).trim().getAsDouble(frac) || frac < 0.0 ||
      frac > 1.0) {
    llvm::report_fatal_error(llvm::Twine(kFrontendFracAvailEnv) +
                             " must be a number between 0 and 1, got '" +
                             value + "'");
  }
  return frac;
}

auto scheduler::arch_view::getReservedBytes(mlir::ktdf_arch::MemoryOp mem_op)
    -> std::size_t {
  auto reserved = static_cast<std::size_t>(mem_op.getFirstAvailableAddress());

  // The front end's share is a fraction of the memory's size.
  auto space = mlir::dyn_cast<mlir::ktdp::MemorySpaceAttr>(mem_op.getKind());
  const auto size = mem_op.getSize();
  if (space && space.getKind() == mlir::ktdp::MemorySpaceKind::ct_local &&
      size) {
    reserved =
        std::max(reserved, getFrontendCoreLocalBytes(
                               static_cast<std::size_t>(size->getSExtValue()),
                               getFrontendFracAvail()));
  }
  return reserved;
}
