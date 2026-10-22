//===-- ReductionFootprint.h ------------------------------------*- c++ -*-===//
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
// ReductionFootprint: the storage the operand tiles of one chunk of a reduction
// occupy in each memory, and the chunk sizes that fit a budget.
//
//===----------------------------------------------------------------------===//

#ifndef DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_REDUCTIONFOOTPRINT_H_
#define DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_REDUCTIONFOOTPRINT_H_

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Utils/StructuredOpsUtils.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::ktdf {

/// The room the operand tiles of one chunk of a reduction may take, per memory.
///
/// Memories are identified by their memory space, as memrefs carry it.
struct FootprintBudget {
  /// The bytes each memory has for the tiles of one chunk.
  llvm::SmallDenseMap<Attribute, int64_t> capacity;
  /// The memory the tile of each operand takes room in, by operand number. An
  /// operand in no memory, or in one without a capacity, takes none.
  llvm::SmallDenseMap<unsigned, Attribute> placement;

  /// Places the tile of @p operand in @p memory .
  void place(OpOperand& operand, Attribute memory) {
    placement[operand.getOperandNumber()] = memory;
  }

  /// Returns the memory the tile of the operand numbered @p number takes room
  /// in, or std::nullopt if it takes none.
  [[nodiscard]] auto getMemory(unsigned number) const
      -> std::optional<Attribute>;
};

/// Models the bytes the operand tiles of a chunk of a reduction
/// `linalg.generic` occupy, as a function of its tile sizes.
///
/// Each operand takes a tile shaped the way tiling slices it: by its indexing
/// map at the last point of the chunk.
///
/// Only reduction dimensions are chunked, and never the innermost loop
/// dimension or the innermost dimension of an input.
class ReductionFootprint {
 public:
  /// Models @p op , or fails if it cannot be modelled: it must be a
  /// `linalg.generic` with static loop ranges, whose operands are statically
  /// shaped, of a fixed-width element type, and indexed by maps without
  /// symbols.
  [[nodiscard]] static auto get(Operation* op) -> FailureOr<ReductionFootprint>;

  /// Returns the bytes the operand tiles of a chunk of @p tile_sizes , given
  /// per loop dimension, occupy in each memory @p budget has a capacity for.
  [[nodiscard]] auto getFootprint(llvm::ArrayRef<int64_t> tile_sizes,
                                  const FootprintBudget& budget) const
      -> llvm::SmallDenseMap<Attribute, int64_t>;

  /// Returns the tile sizes per loop dimension whose footprint fits every
  /// capacity of @p budget , or std::nullopt if none does.
  ///
  /// Each tile size divides its loop range. Dimensions are shrunk from the
  /// outermost to the innermost in the layout of the budgeted operands, each to
  /// the largest size that fits or, failing that, the one that leaves the
  /// fewest bytes over the capacities; the ones shrunk before the last are
  /// then grown back as far as the budget allows.
  [[nodiscard]] auto getTileSizes(const FootprintBudget& budget) const
      -> std::optional<llvm::SmallVector<int64_t>>;

 private:
  struct Operand {
    unsigned number;
    AffineMap map;
    llvm::SmallVector<int64_t> shape;
    int64_t elem_bytes;
  };

  ReductionFootprint(llvm::SmallVector<int64_t> loop_ranges,
                     llvm::SmallVector<utils::IteratorType> iterator_types,
                     llvm::SmallVector<Operand> operands,
                     llvm::SmallVector<unsigned> pinned)
      : loop_ranges_(std::move(loop_ranges)),
        iterator_types_(std::move(iterator_types)),
        operands_(std::move(operands)),
        pinned_(std::move(pinned)) {}

  /// Returns the bytes the operand tiles of a chunk of @p tile_sizes take over
  /// the capacities of @p budget , in total.
  [[nodiscard]] auto getExcess(llvm::ArrayRef<int64_t> tile_sizes,
                               const FootprintBudget& budget) const -> int64_t;

  /// Returns the dimensions that may be shrunk, outermost first.
  [[nodiscard]] auto getShrinkOrder(const FootprintBudget& budget) const
      -> llvm::SmallVector<unsigned>;

  llvm::SmallVector<int64_t> loop_ranges_;
  llvm::SmallVector<utils::IteratorType> iterator_types_;
  llvm::SmallVector<Operand> operands_;
  /// The dimensions that are never chunked.
  llvm::SmallVector<unsigned> pinned_;
};

}  // namespace mlir::ktdf

#endif  // DATAFLOW_SCHEDULER_DIALECT_KTDF_ANALYSIS_REDUCTIONFOOTPRINT_H_
