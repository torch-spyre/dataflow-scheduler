//===-- ReservedMemory.h ----------------------------------------*- c++ -*-===//
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
//
// ReservedMemory
//
// The bytes at the start of a memory that the scheduler must not allocate
// from, as every view of the device reads them.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_RESERVEDMEMORY_H_
#define DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_RESERVEDMEMORY_H_

#include <cstddef>
#include <cstdint>

#include "dataflow-scheduler/Dialect/KTDFArch/KTDFArch.h"

namespace scheduler::arch_view {

/// The environment variable that splits each core-local (`ct_local`) memory
/// between the front end and the scheduler. The front end owns
/// `1 - DXP_LX_FRAC_AVAIL` of it from address zero, and the scheduler
/// allocates above. The name is torch-spyre's, where the core-local memory is
/// Spyre's LX: its scratchpad planner reads this same variable, with the same
/// default, so that the two cannot disagree about a layout they share; a name
/// of its own here would let them.
inline constexpr const char* kFrontendFracAvailEnv = "DXP_LX_FRAC_AVAIL";

/// Default of `DXP_LX_FRAC_AVAIL`, as in torch-spyre.
inline constexpr double kDefaultFrontendFracAvail = 0.2;

/// Bytes at the top of a core-local memory that hold program and debug data
/// and are not part of what the front end and the scheduler split.
inline constexpr std::size_t kCoreLocalProgramDebugBytes = 64 * 1024;

/// Granularity the front end rounds its part of a core-local memory up to:
/// one stick.
inline constexpr std::size_t kCoreLocalAllocationGranularityBytes = 128;

/// The front end's part of a core-local memory of \p size_in_bytes, in bytes
/// from address zero, for a `DXP_LX_FRAC_AVAIL` of \p frac_avail:
/// `(size - 64 KiB) * (1 - frac_avail)`, truncated, rounded up to 128 bytes,
/// as torch-spyre's `_lx_planning_size` computes it. \p frac_avail must be in
/// [0, 1].
[[nodiscard]] auto getFrontendCoreLocalBytes(std::size_t size_in_bytes,
                                             double frac_avail) -> std::size_t;

/// The `DXP_LX_FRAC_AVAIL` in effect: the environment variable if it is set,
/// `kDefaultFrontendFracAvail` otherwise. Aborts with a message if it is set
/// to something that is not a number in [0, 1], as torch-spyre rejects it.
[[nodiscard]] auto getFrontendFracAvail() -> double;

/// Bytes at the start of \p mem_op the scheduler must not use: the memory's
/// own `reserved` (0 if it has none), raised to the front end's part of it for
/// a core-local (`ct_local`) memory. The front end's part is a fraction of the
/// memory's size, so a core-local memory without a size keeps its `reserved`,
/// as every other memory does.
[[nodiscard]] auto getReservedBytes(mlir::ktdf_arch::MemoryOp mem_op)
    -> std::size_t;

}  // namespace scheduler::arch_view

#endif  // DATAFLOW_SCHEDULER_ANALYSIS_ARCHVIEWS_RESERVEDMEMORY_H_
