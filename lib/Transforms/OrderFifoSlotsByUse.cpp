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
#include <llvm/Support/ErrorHandling.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <cassert>
#include <optional>
#include <utility>

#include "dataflow-scheduler/Dialect/KTDF/Analysis/StageGraph.h"
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

/// The pair of ends that names one physical FIFO.
using Ends = std::pair<mlir::Attribute, mlir::Attribute>;

/// The ops that take a slot of each FIFO.
using ChannelAccesses =
    llvm::DenseMap<Ends, llvm::SmallPtrSet<mlir::Operation*, 8>>;

/// Collects, for each FIFO, every op in @p func that takes one of its slots.
///
/// A FIFO is named by its ends alone: the slots of all pipelines in @p func
/// between the same ends go through it, whichever allocation they come from.
[[nodiscard]] auto collectChannelAccesses(mlir::func::FuncOp func)
    -> ChannelAccesses {
  ChannelAccesses channels;
  func.walk([&](mlir::Operation* op) {
    // Yielding a slot out of the private only defines it.
    if (llvm::isa<mlir::ktdf::PrivateYieldOp>(op)) {
      return;
    }
    for (auto operand : op->getOperands()) {
      if (auto type =
              llvm::dyn_cast<mlir::ktdf::FifoSlotType>(operand.getType())) {
        channels[{type.getSrc(), type.getDest()}].insert(op);
      }
    }
  });
  return channels;
}

struct OrderFifoSlotsByUsePass
    : impl::OrderFifoSlotsByUsePassBase<OrderFifoSlotsByUsePass> {
  using OrderFifoSlotsByUsePassBase::OrderFifoSlotsByUsePassBase;

  void runOnOperation() override {
    if (disable_this_pass) {
      return;
    }

    getOperation()->walk([&](mlir::func::FuncOp func) {
      const auto channels = collectChannelAccesses(func);
      func.walk([&](mlir::ktdf::PipelineOp pipeline) {
        orderPipeline(pipeline, channels);
      });
    });
  }

 private:
  void orderPipeline(mlir::ktdf::PipelineOp pipeline,
                     const ChannelAccesses& channels) {
    auto private_op = pipeline.getPrivateOp();
    if (!private_op) {
      return;
    }
    mlir::DominanceInfo dominance(pipeline);

    // The slots of each FIFO, i.e. each pair of ends, in the order they are
    // written.
    llvm::DenseMap<Ends, unsigned> fifo_index;
    llvm::SmallVector<std::pair<Ends, llvm::SmallVector<SlotAccess>>> fifos;
    for (auto slot : private_op->getResults()) {
      auto type = llvm::dyn_cast<mlir::ktdf::FifoSlotType>(slot.getType());
      if (!type) {
        continue;
      }
      auto access = getSlotAccess(slot);
      if (!access) {
        continue;
      }
      const Ends ends{type.getSrc(), type.getDest()};
      auto [it, inserted] = fifo_index.try_emplace(ends, fifos.size());
      if (inserted) {
        fifos.emplace_back(ends, llvm::SmallVector<SlotAccess>{});
      }
      fifos[it->second].second.push_back(*access);
    }

    for (auto& [ends, accesses] : fifos) {
      if (accesses.size() < 2) {
        continue;
      }
      if (!hasKnownProtocol(accesses, channels.lookup(ends))) {
        ++num_skipped;
        continue;
      }
      orderFifo(accesses, dominance);
    }
  }

  /// Determines whether the FIFO the slots of @p accesses go through delivers
  /// them in an order this pass can tell, given @p channel , every op of the
  /// function accessing the FIFO.
  ///
  /// That holds when one block writes all of them, another reads all of them,
  /// and no op other than those touches the FIFO. Each write and read is then
  /// run once per run of its block, so the FIFO carries the order of the writes
  /// over and over, and the consumer takes them in the order of the reads.
  /// Applying one permutation to both keeps them in step.
  [[nodiscard]] static auto hasKnownProtocol(
      llvm::ArrayRef<SlotAccess> accesses,
      const llvm::SmallPtrSet<mlir::Operation*, 8>& channel) -> bool {
    auto* writer_block = accesses.front().writer->getBlock();
    auto* reader_block = accesses.front().reader->getBlock();
    llvm::SmallPtrSet<mlir::Operation*, 8> own;
    for (const auto& access : accesses) {
      if (access.writer->getBlock() != writer_block ||
          access.reader->getBlock() != reader_block) {
        LDBG() << "slots written or read in more than one block, not ordered: "
               << access.slot;
        return false;
      }
      own.insert(access.writer);
      own.insert(access.reader);
    }

    // Each op of `own` takes a slot of the FIFO, so it is one of `channel`.
    if (own.size() != channel.size()) {
      LDBG() << "FIFO accessed by other ops too, not ordered: "
             << accesses.front().slot;
      return false;
    }

    auto writer_stage =
        accesses.front().writer->getParentOfType<mlir::ktdf::StageOp>();
    auto reader_stage =
        accesses.front().reader->getParentOfType<mlir::ktdf::StageOp>();
    assert(writer_stage && reader_stage && "a slot is accessed in a stage");
    // A pipeline is a DAG of stages, so the order in which its stages run puts
    // no access of the FIFO on a cycle.
    if (isOnCycle(writer_stage) || isOnCycle(reader_stage)) {
      llvm_unreachable("a pipeline is a DAG of stages");
    }
    if (writer_stage == reader_stage) {
      LDBG() << "slots written and read in the same stage, not ordered: "
             << accesses.front().slot;
      return false;
    }
    return true;
  }

  /// Determines whether @p stage (transitively) depends on itself.
  [[nodiscard]] static auto isOnCycle(mlir::ktdf::StageOp stage) -> bool {
    return llvm::any_of(
        mlir::ktdf::StageGraph::Node(stage).getProducers(),
        [&](mlir::ktdf::StageOp producer) {
          return mlir::ktdf::StageGraph::Node(producer).dependsOn(stage);
        });
  }

  void orderFifo(llvm::ArrayRef<SlotAccess> accesses,
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
    // consumed. No other op takes from the FIFO (see hasKnownProtocol), so
    // the reads always move.
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
