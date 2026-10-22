//===-- ReductionFootprint.cpp ----------------------------------*- c++ -*-===//
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

#include "dataflow-scheduler/Dialect/KTDF/Analysis/ReductionFootprint.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/iterator.h>
#include <llvm/ADT/iterator_range.h>
#include <llvm/Support/DebugLog.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/BuiltinTypes.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iterator>
#include <utility>

#define DEBUG_TYPE "reduction-footprint"

namespace {

/// Iterates the divisors of a positive integer, largest first.
///
/// Divisors pair up around the square root: walking up to it yields the large
/// ones as the co-divisors of the small ones, walking back down yields the
/// small ones themselves.
class DivisorIterator
    : public llvm::iterator_facade_base<DivisorIterator,
                                        std::forward_iterator_tag, int64_t,
                                        std::ptrdiff_t, int64_t*, int64_t> {
 public:
  /// Constructs the end iterator.
  DivisorIterator() = default;
  explicit DivisorIterator(int64_t value)
      : value_(value), root_(getRoot(value)), factor_(1), ascending_(true) {
    assert(value > 0 && "only positive integers have divisors");
  }

  auto operator==(const DivisorIterator& other) const -> bool {
    return factor_ == other.factor_ && ascending_ == other.ascending_;
  }
  auto operator*() const -> int64_t {
    return ascending_ ? value_ / factor_ : factor_;
  }
  auto operator++() -> DivisorIterator& {
    if (ascending_) {
      do {
        ++factor_;
      } while (factor_ <= root_ && value_ % factor_ != 0);
      if (factor_ <= root_) {
        return *this;
      }

      // A square root is its own co-divisor, and was yielded as such.
      ascending_ = false;
      factor_ = root_ * root_ == value_ ? root_ : root_ + 1;
    }
    do {
      --factor_;
    } while (factor_ > 0 && value_ % factor_ != 0);
    return *this;
  }

 private:
  [[nodiscard]] static auto getRoot(int64_t value) -> int64_t {
    auto root = static_cast<int64_t>(std::sqrt(static_cast<double>(value)));
    while (root * root > value) {
      --root;
    }
    while ((root + 1) * (root + 1) <= value) {
      ++root;
    }
    return root;
  }

  int64_t value_ = 0;
  int64_t root_ = 0;
  int64_t factor_ = 0;
  bool ascending_ = false;
};

/// Returns the divisors of @p value , largest first.
[[nodiscard]] auto getDivisors(int64_t value)
    -> llvm::iterator_range<DivisorIterator> {
  return llvm::make_range(DivisorIterator(value), DivisorIterator());
}

}  // namespace

auto mlir::ktdf::FootprintBudget::getMemory(unsigned number) const
    -> std::optional<Attribute> {
  const auto it = placement.find(number);
  if (it == placement.end() || !capacity.contains(it->second)) {
    return std::nullopt;
  }
  return it->second;
}

auto mlir::ktdf::ReductionFootprint::get(Operation* op)
    -> FailureOr<ReductionFootprint> {
  auto generic = llvm::dyn_cast<linalg::GenericOp>(op);
  if (!generic) {
    return failure();
  }
  auto loop_ranges = generic.getStaticLoopRanges();
  if (llvm::any_of(loop_ranges, ShapedType::isDynamic)) {
    LDBG() << "dynamic loop range";
    return failure();
  }

  llvm::SmallVector<unsigned> pinned;
  if (const auto num_loops = generic.getNumLoops(); num_loops != 0) {
    pinned.push_back(num_loops - 1);
  }
  llvm::SmallVector<Operand> operands;
  for (auto& operand : generic->getOpOperands()) {
    const auto map = generic.getMatchingIndexingMap(&operand);
    const auto type = llvm::dyn_cast<ShapedType>(operand.get().getType());
    if (!type || !type.hasStaticShape() ||
        !type.getElementType().isIntOrFloat() || map.getNumSymbols() != 0) {
      LDBG() << "operand cannot be modelled: " << operand.get();
      return failure();
    }
    if (generic.isDpsInput(&operand) && map.getNumResults() != 0) {
      if (auto innermost =
              llvm::dyn_cast<AffineDimExpr>(map.getResults().back());
          innermost && !llvm::is_contained(pinned, innermost.getPosition())) {
        pinned.push_back(innermost.getPosition());
      }
    }
    operands.push_back(Operand{
        operand.getOperandNumber(), map, llvm::to_vector(type.getShape()),
        type.getElementType().getIntOrFloatBitWidth() / 8});
  }
  return ReductionFootprint(std::move(loop_ranges),
                            generic.getIteratorTypesArray(),
                            std::move(operands), std::move(pinned));
}

auto mlir::ktdf::ReductionFootprint::getFootprint(
    llvm::ArrayRef<int64_t> tile_sizes, const FootprintBudget& budget) const
    -> llvm::SmallDenseMap<Attribute, int64_t> {
  llvm::SmallDenseMap<Attribute, int64_t> footprint;
  for (auto [memory, capacity] : budget.capacity) {
    footprint[memory] = 0;
  }

  // Tiling sizes a slice by the map at the last point of the tile.
  const auto last_point =
      llvm::map_to_vector(tile_sizes, [](int64_t size) { return size - 1; });
  for (const auto& operand : operands_) {
    const auto memory = budget.getMemory(operand.number);
    if (!memory) {
      continue;
    }
    int64_t elements = 1;
    for (auto [last, size] :
         llvm::zip_equal(operand.map.compose(last_point), operand.shape)) {
      elements *= std::min(size, last + 1);
    }
    footprint[*memory] += elements * operand.elem_bytes;
  }
  return footprint;
}

auto mlir::ktdf::ReductionFootprint::getExcess(
    llvm::ArrayRef<int64_t> tile_sizes, const FootprintBudget& budget) const
    -> int64_t {
  int64_t excess = 0;
  for (auto [memory, bytes] : getFootprint(tile_sizes, budget)) {
    excess += std::max<int64_t>(0, bytes - budget.capacity.lookup(memory));
  }
  return excess;
}

auto mlir::ktdf::ReductionFootprint::getShrinkOrder(
    const FootprintBudget& budget) const -> llvm::SmallVector<unsigned> {
  // How outer a dimension is is the logarithm of its stride in the row-major
  // layout of the budgeted operands, weighed by their size where they
  // disagree.
  const auto num_dims = static_cast<unsigned>(loop_ranges_.size());
  llvm::SmallVector<double> outerness(num_dims, 0.0);
  llvm::SmallVector<double> weights(num_dims, 0.0);
  for (const auto& operand : operands_) {
    if (!budget.getMemory(operand.number)) {
      continue;
    }
    const auto linearize = [&](llvm::ArrayRef<int64_t> point) {
      int64_t offset = 0;
      for (auto [index, size] :
           llvm::zip_equal(operand.map.compose(point), operand.shape)) {
        offset = (offset * size) + index;
      }
      return offset;
    };
    int64_t elements = 1;
    for (auto size : operand.shape) {
      elements *= size;
    }
    const auto weight = static_cast<double>(elements * operand.elem_bytes);

    llvm::SmallVector<int64_t> point(num_dims, 0);
    const auto origin = linearize(point);
    for (auto dim : llvm::seq(num_dims)) {
      point[dim] = 1;
      if (const auto stride = std::abs(linearize(point) - origin);
          stride != 0) {
        outerness[dim] += weight * std::log2(static_cast<double>(stride));
        weights[dim] += weight;
      }
      point[dim] = 0;
    }
  }

  llvm::SmallVector<unsigned> order;
  for (auto dim : llvm::seq(num_dims)) {
    if (iterator_types_[dim] == utils::IteratorType::reduction &&
        weights[dim] > 0.0 && loop_ranges_[dim] > 1 &&
        !llvm::is_contained(pinned_, dim)) {
      outerness[dim] /= weights[dim];
      order.push_back(dim);
    }
  }
  llvm::stable_sort(order, [&](unsigned lhs, unsigned rhs) {
    return outerness[lhs] > outerness[rhs];
  });
  return order;
}

auto mlir::ktdf::ReductionFootprint::getTileSizes(const FootprintBudget& budget)
    const -> std::optional<llvm::SmallVector<int64_t>> {
  auto tile_sizes = loop_ranges_;
  auto current = getExcess(tile_sizes, budget);
  if (current == 0) {
    return tile_sizes;
  }

  const auto excess_with = [&](unsigned dim, int64_t size) {
    auto trial = tile_sizes;
    trial[dim] = size;
    return getExcess(trial, budget);
  };

  llvm::SmallVector<unsigned> shrunk;
  auto fits = false;
  for (auto dim : getShrinkOrder(budget)) {
    std::optional<std::pair<int64_t, int64_t>> smallest;
    for (auto size : getDivisors(loop_ranges_[dim])) {
      const auto excess = excess_with(dim, size);
      if (excess == 0) {
        smallest = {size, excess};
        fits = true;
        break;
      }
      if (!smallest || excess < smallest->second) {
        smallest = {size, excess};
      }
    }
    if (!fits && smallest->second >= current) {
      continue;
    }
    tile_sizes[dim] = smallest->first;
    current = smallest->second;
    shrunk.push_back(dim);
    LDBG() << "chunking d" << dim << " by " << smallest->first << ": "
           << smallest->second << " bytes over";
    if (fits) {
      break;
    }
  }
  if (!fits) {
    return std::nullopt;
  }

  // The last dimension shrunk took what it needed; the ones before may have
  // been shrunk further than that.
  for (auto dim : llvm::reverse(llvm::ArrayRef(shrunk).drop_back())) {
    for (auto size : getDivisors(loop_ranges_[dim])) {
      if (size <= tile_sizes[dim]) {
        break;
      }
      if (excess_with(dim, size) == 0) {
        LDBG() << "growing d" << dim << " back to " << size;
        tile_sizes[dim] = size;
        break;
      }
    }
  }
  return tile_sizes;
}
