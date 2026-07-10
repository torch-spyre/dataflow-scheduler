//===-- LoopTiling.cpp ------------------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Analysis/LoopTiling.h"

#include "dataflow-scheduler/Dialect/KTDF/Analysis/Utils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Operation.h"

#define DEBUG_TYPE "loop-tiling-analysis"

using namespace mlir;
using namespace mlir::ktdf;

//===----------------------------------------------------------------------===//
// LoopTilingMetadata
//===----------------------------------------------------------------------===//

void LoopTilingMetadata::dump() const {
  llvm::errs() << "LoopTilingMetadata:\n";
  llvm::errs() << "  divisibility: " << divisibility << "\n";
  llvm::errs() << "  min_value: " << min_value << "\n";
  llvm::errs() << "  strategy: ";
  switch (strategy) {
    case Strategy::kNone:
      llvm::errs() << "None";
      break;
    case Strategy::kTiling:
      llvm::errs() << "Tiling";
      break;
    case Strategy::kStripMining:
      llvm::errs() << "StripMining";
      break;
  }
  llvm::errs() << "\n";
  llvm::errs() << "  is_invalidated: " << (is_invalidated ? "true" : "false")
               << "\n";
}

//===----------------------------------------------------------------------===//
// PerfectNestingInfo
//===----------------------------------------------------------------------===//

void PerfectNestingInfo::dump() const {
  llvm::errs() << "PerfectNestingInfo (depth=" << depth() << "):\n";
  for (size_t i = 0; i < loops.size(); ++i) {
    llvm::errs() << "  Loop " << i << ": ";
    if (loops[i]) {
      loops[i]->print(llvm::errs());
      llvm::errs() << "\n";
    } else {
      llvm::errs() << "<null>\n";
    }
  }
}

//===----------------------------------------------------------------------===//
// LoopTilingInfo
//===----------------------------------------------------------------------===//

LoopTilingInfo::LoopTilingInfo(Operation* root_op) {
  computeLoopNests(root_op);
}

const LoopTilingMetadata* LoopTilingInfo::getMetadataForLoop(
    Operation* loop) const {
  auto it = loop_to_metadata_.find(loop);
  if (it != loop_to_metadata_.end()) {
    return &it->second;
  }
  return nullptr;
}

const PerfectNestingInfo* LoopTilingInfo::getNestContainingLoop(
    Operation* loop) const {
  auto it = loop_to_nest_index_.find(loop);
  if (it != loop_to_nest_index_.end()) {
    return &loop_nests_[it->second];
  }
  return nullptr;
}

void LoopTilingInfo::invalidateLoop(Operation* loop) {
  auto it = loop_to_metadata_.find(loop);
  if (it != loop_to_metadata_.end()) {
    it->second.is_invalidated = true;
  }
}

void LoopTilingInfo::updateMetadata(Operation* loop,
                                    const LoopTilingMetadata& metadata) {
  auto it = loop_to_metadata_.find(loop);
  if (it != loop_to_metadata_.end()) {
    it->second = metadata;
  }
}

void LoopTilingInfo::dump() const {
  llvm::errs() << "LoopTilingInfo: Found " << loop_nests_.size()
               << " perfectly nested loop group(s)\n";
  for (size_t i = 0; i < loop_nests_.size(); ++i) {
    llvm::errs() << "\nNest " << i << ":\n";
    loop_nests_[i].dump();

    // Dump metadata for each loop in this nest
    for (auto loop : loop_nests_[i].loops) {
      if (auto* metadata = getMetadataForLoop(loop)) {
        llvm::errs() << "  Metadata for loop " << loop << ":\n";
        // Indent the metadata dump output
        llvm::errs() << "  ";
        metadata->dump();
      }
    }
  }
}

void LoopTilingInfo::computeLoopNests(Operation* root_op) {
  // Collect all scf.for loops in the module
  llvm::SmallVector<scf::ForOp> all_loops;
  root_op->walk<WalkOrder::PreOrder>(
      [&](scf::ForOp for_op) { all_loops.push_back(for_op); });

  LDBG(1) << "Found " << all_loops.size() << " scf.for loops in total";

  // Track which loops have been processed to avoid duplicates
  llvm::DenseSet<Operation*> visited;

  // Process each loop to find perfectly nested groups
  for (scf::ForOp loop : all_loops) {
    // Skip if already processed as part of another nest
    if (visited.contains(loop.getOperation())) continue;

    // Get perfectly nested loops starting from this loop
    llvm::SmallVector<scf::ForOp> nested_loops;
    getPerfectlyNestedLoops(nested_loops, loop);
    assert(!nested_loops.empty() && "should at least return the loop itself");

    // Limit to maximum depth
    if (nested_loops.size() > kMaxLoopNestDepth) {
      LDBG(1) << "Limiting nest depth from " << nested_loops.size() << " to "
              << kMaxLoopNestDepth;
      nested_loops.resize(kMaxLoopNestDepth);
    }

    // Split the nest at non-parallel loop boundaries to create sub-nests
    // that only contain consecutive parallel loops (for tiling candidacy)
    llvm::SmallVector<llvm::SmallVector<scf::ForOp>> sub_nests;
    llvm::SmallVector<scf::ForOp> current_sub_nest;

    for (scf::ForOp nested_loop : nested_loops) {
      current_sub_nest.push_back(nested_loop);

      // If this loop is not parallel and we have more loops ahead,
      // or if this is the last loop, finalize the current sub-nest
      bool is_parallel = isParallelLoop(nested_loop);
      bool is_last = (nested_loop == nested_loops.back());

      if (!is_parallel || is_last) {
        // Finalize current sub-nest
        sub_nests.push_back(current_sub_nest);
        current_sub_nest.clear();
      }
    }

    // Create nest info for each sub-nest
    for (auto& sub_nest_loops : sub_nests) {
      PerfectNestingInfo nest_info;
      nest_info.loops.assign(sub_nest_loops.begin(), sub_nest_loops.end());

      // Assign metadata to each loop in the nest
      assignMetadata(nest_info);

      // Record this nest
      size_t nest_index = loop_nests_.size();
      loop_nests_.push_back(std::move(nest_info));

      // Mark all loops in this nest as processed and record their nest index
      for (scf::ForOp nested_loop : sub_nest_loops) {
        visited.insert(nested_loop.getOperation());
        loop_to_nest_index_[nested_loop.getOperation()] = nest_index;
      }

      LLVM_DEBUG({
        llvm::dbgs() << "Created nest " << nest_index << " with "
                     << sub_nest_loops.size() << " loop(s):\n";
        for (size_t i = 0; i < sub_nest_loops.size(); ++i) {
          llvm::dbgs() << "  Loop " << i << ": " << sub_nest_loops[i]
                       << " (parallel: "
                       << (isParallelLoop(sub_nest_loops[i]) ? "yes" : "no")
                       << ")\n";
        }
      });
    }
  }

  LDBG(1) << "Total nests created: " << loop_nests_.size();
}

void LoopTilingInfo::assignMetadata(PerfectNestingInfo& nest) {
  size_t nest_depth = nest.depth();

  // For each loop in the nest, assign initial metadata
  for (size_t i = 0; i < nest.loops.size(); ++i) {
    scf::ForOp loop = nest.loops[i];
    LoopTilingMetadata metadata;

    // TODO: Implement smarter heuristics here
    // For now, use conservative defaults:
    // - divisibility = 1 (no specific constraint)
    // - min_value = 1 (at least one iteration)
    metadata.divisibility = 1;
    metadata.min_value = 1;
    metadata.is_invalidated = false;

    // Determine optimization strategy:
    // - Tiling: loop is part of a perfectly nested loop of depth > 1
    //           Note: computeLoopNests() already ensures that nests with depth
    //           > 1 only contain consecutive *parallel* loops so all loops in
    //           such nests are tiling candidates.
    // - StripMining: outermost loop (no parent loop) and nest depth == 1
    // - None: otherwise
    metadata.strategy = LoopTilingMetadata::Strategy::kNone;
    if (nest_depth > 1) {
      // Part of a multi-dimensional nest - candidate for tiling
      metadata.strategy = LoopTilingMetadata::Strategy::kTiling;
    } else if (nest_depth == 1 && i == 0) {
      // Single loop in nest, check if it's outermost (no parent loop)
      bool is_outermost = !loop->getParentOfType<scf::ForOp>();
      if (is_outermost) {
        metadata.strategy = LoopTilingMetadata::Strategy::kStripMining;
      }
    }

    // Store in the map for fast lookup
    loop_to_metadata_[loop.getOperation()] = metadata;

    LLVM_DEBUG({
      llvm::dbgs() << "Assigned metadata to loop " << loop << ":\n";
      metadata.dump();
    });
  }
}

// Made with Bob
