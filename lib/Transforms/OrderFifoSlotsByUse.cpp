//===-- OrderFifoSlotsByUse.cpp ---------------------------------*- c++ -*-===//
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

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/DebugLog.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <cassert>
#include <optional>
#include <tuple>
#include <utility>

#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Transforms/Passes.h"  // IWYU pragma: keep

#define PASS_NAME "order-fifo-slots-by-use"
#define DEBUG_TYPE PASS_NAME

static llvm::cl::opt<bool> disable_this_pass(
    "disable-" PASS_NAME,
    llvm::cl::desc("Disable Order FIFO Slots By Use pass"),
    llvm::cl::init(false));

namespace scheduler {
#define GEN_PASS_DEF_ORDERFIFOSLOTSBYUSEPASS
#include "dataflow-scheduler/Transforms/Passes.h.inc"
}  // namespace scheduler

using namespace scheduler;

namespace {

/// A FIFO slot with the one op writing it and the one op reading it.
struct SlotAccess {
  mlir::Value slot;
  mlir::Operation* writer = nullptr;
  mlir::Operation* reader = nullptr;
};

/// Returns the op writing @p slot and the op reading it, when there is exactly
/// one of each.
[[nodiscard]] auto getSlotAccess(mlir::Value slot)
    -> std::optional<SlotAccess> {
  SlotAccess access{slot};
  for (auto& use : slot.getUses()) {
    auto* user = use.getOwner();
    auto is_write = false;
    auto is_read = false;
    if (llvm::isa<mlir::ktdf::WriteToFifoOp>(user)) {
      is_write = true;
    } else if (llvm::isa<mlir::ktdf::ReadFromFifoOp>(user)) {
      is_read = true;
    } else if (auto transfer = llvm::dyn_cast<mlir::ktdf::DataTransferOp>(user);
               transfer) {
      is_write = transfer.getDestination() == slot;
      is_read = transfer.getSource() == slot;
    }

    if (is_write == is_read) {
      return std::nullopt;
    }
    auto*& accessor = is_write ? access.writer : access.reader;
    if (accessor) {
      return std::nullopt;
    }
    accessor = user;
  }

  if (!access.writer || !access.reader) {
    return std::nullopt;
  }
  return access;
}

/// Where in its block the value a slot is read into is first consumed, as an
/// ordered key: the position of the consuming op in the block, then, when that
/// op is a linalg.generic, the position in its body of the first op using the
/// operand. A read transferring the slot into memory consumes it where it is.
[[nodiscard]] auto getConsumptionKey(
    mlir::Operation* reader,
    const llvm::DenseMap<mlir::Operation*, unsigned>& positions)
    -> std::optional<std::pair<unsigned, unsigned>> {
  auto* block = reader->getBlock();
  if (llvm::isa<mlir::ktdf::DataTransferOp>(reader)) {
    return std::make_pair(positions.lookup(reader), 0U);
  }

  std::optional<std::pair<unsigned, unsigned>> key;
  for (auto& use : reader->getResult(0).getUses()) {
    auto* in_block = block->findAncestorOpInBlock(*use.getOwner());
    if (!in_block) {
      return std::nullopt;
    }

    unsigned in_body = 0;
    if (auto generic = llvm::dyn_cast<mlir::linalg::GenericOp>(in_block);
        generic && generic == use.getOwner()) {
      auto argument = generic.getMatchingBlockArgument(&use);
      auto* body = generic.getBody();
      in_body = std::distance(body->begin(), body->end());
      for (auto* arg_user : argument.getUsers()) {
        if (auto* in_generic = body->findAncestorOpInBlock(*arg_user);
            in_generic) {
          in_body = std::min<unsigned>(
              in_body, std::distance(body->begin(), in_generic->getIterator()));
        }
      }
    }

    const auto candidate = std::make_pair(positions.lookup(in_block), in_body);
    if (!key || candidate < *key) {
      key = candidate;
    }
  }
  return key;
}

/// Moves @p ops, all in one block, so that they appear in the given order, all
/// of them in front of the earliest. Returns false, leaving the IR unchanged,
/// when that is not plainly safe: an op would move above a definition of one of
/// its operands, or, when @p keep_memory_order, above an op between them that
/// touches memory.
[[nodiscard]] auto moveIntoOrder(llvm::ArrayRef<mlir::Operation*> ops,
                                 mlir::DominanceInfo& dominance,
                                 bool keep_memory_order) -> bool {
  auto* first = *llvm::min_element(
      ops, [](auto* lhs, auto* rhs) { return lhs->isBeforeInBlock(rhs); });
  auto* last = *llvm::max_element(
      ops, [](auto* lhs, auto* rhs) { return lhs->isBeforeInBlock(rhs); });

  const llvm::SmallPtrSet<mlir::Operation*, 8> moved(ops.begin(), ops.end());
  for (auto it = first->getIterator(); keep_memory_order && &*it != last;
       ++it) {
    if (!moved.contains(&*it) && !mlir::isMemoryEffectFree(&*it)) {
      return false;
    }
  }
  for (auto* op : ops) {
    for (auto operand : op->getOperands()) {
      if (!dominance.properlyDominates(operand, first)) {
        return false;
      }
    }
  }

  // The first takes the place of the earliest, and each of the others follows
  // the one before it.
  mlir::Operation* previous = nullptr;
  for (auto* op : ops) {
    if (!previous) {
      if (op != first) {
        op->moveBefore(first);
      }
    } else {
      op->moveAfter(previous);
    }
    previous = op;
  }
  return true;
}

struct OrderFifoSlotsByUsePass
    : impl::OrderFifoSlotsByUsePassBase<OrderFifoSlotsByUsePass> {
  using OrderFifoSlotsByUsePassBase::OrderFifoSlotsByUsePassBase;

  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    getOperation()->walk([&](mlir::ktdf::PipelineOp pipeline) {
      auto private_op = pipeline.getPrivateOp();
      if (!private_op) {
        return;
      }
      mlir::DominanceInfo dominance(pipeline);

      // The slots of one FIFO, i.e. one pair of ends, written in one block and
      // read in another, in the order they are written.
      using Key = std::tuple<mlir::Attribute, mlir::Attribute, mlir::Block*,
                             mlir::Block*>;
      llvm::DenseMap<Key, unsigned> fifo_index;
      llvm::SmallVector<llvm::SmallVector<SlotAccess>> fifos;
      for (auto slot : private_op->getResults()) {
        auto type = llvm::dyn_cast<mlir::ktdf::FifoSlotType>(slot.getType());
        if (!type) {
          continue;
        }
        auto access = getSlotAccess(slot);
        if (!access) {
          continue;
        }
        const Key key{type.getSrc(), type.getDest(), access->writer->getBlock(),
                      access->reader->getBlock()};
        auto [it, inserted] = fifo_index.try_emplace(key, fifos.size());
        if (inserted) {
          fifos.emplace_back();
        }
        fifos[it->second].push_back(*access);
      }

      for (auto& accesses : fifos) {
        if (accesses.size() < 2) {
          continue;
        }
        orderFifo(accesses, dominance);
      }
    });
  }

 private:
  void orderFifo(llvm::SmallVector<SlotAccess>& accesses,
                 mlir::DominanceInfo& dominance) {
    auto* reader_block = accesses.front().reader->getBlock();
    llvm::DenseMap<mlir::Operation*, unsigned> positions;
    for (auto [index, op] : llvm::enumerate(*reader_block)) {
      positions[&op] = index;
    }

    // The order the consumer takes the slots in.
    llvm::SmallVector<std::pair<std::pair<unsigned, unsigned>, SlotAccess>>
        keyed;
    for (const auto& access : accesses) {
      auto key = getConsumptionKey(access.reader, positions);
      if (!key) {
        LDBG() << "slot consumed outside its block, not ordered: "
               << access.slot;
        return;
      }
      keyed.emplace_back(*key, access);
    }
    llvm::stable_sort(keyed, [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
    });

    // Nothing to do when the writes and the reads already follow it.
    const auto in_order = [&](auto get_op) {
      for (size_t i = 1; i < keyed.size(); ++i) {
        if (get_op(keyed[i].second)
                ->isBeforeInBlock(get_op(keyed[i - 1].second))) {
          return false;
        }
      }
      return true;
    };
    const auto writer = [](const SlotAccess& access) { return access.writer; };
    const auto reader = [](const SlotAccess& access) { return access.reader; };
    if (in_order(writer) && in_order(reader)) {
      return;
    }

    const auto writers = llvm::map_to_vector(
        keyed, [&](const auto& entry) { return writer(entry.second); });
    const auto readers = llvm::map_to_vector(
        keyed, [&](const auto& entry) { return reader(entry.second); });
    // The writes go first, as they are the side that can fail, which leaves
    // both sides as they were. A read only takes its slot, which is defined
    // above the stages, and moving it up does not change where its value is
    // consumed, so the reads always move.
    if (!in_order(writer) &&
        !moveIntoOrder(writers, dominance, /*keep_memory_order=*/true)) {
      LDBG() << "unable to reorder the writes of " << *writers.front();
      return;
    }
    if (!in_order(reader)) {
      [[maybe_unused]] const auto moved =
          moveIntoOrder(readers, dominance, /*keep_memory_order=*/false);
      assert(moved && "reads only take their slot");
    }
    ++num_reordered;
  }
};

}  // namespace
