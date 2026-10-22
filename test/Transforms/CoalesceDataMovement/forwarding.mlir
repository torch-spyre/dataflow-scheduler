// RUN: dataflow-scheduler-opt --ktdf-coalesce-data-movement --allow-unregistered-dialect %s | FileCheck %s

// Forwarding of FIFO producers to their consumers within a stage.

// A read of a slot written in the same stage uses the written data.
// CHECK-LABEL: func.func @read_of_write(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      %[[DATA:.+]] = "test.def"()
// CHECK-NEXT:      "test.use"(%[[DATA]])
// CHECK-NEXT:    }
func.func @read_of_write(%slot: !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      %data = "test.def"() : () -> tensor<64xf16>
      ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A read of a reshaped write collapses the written data.
// CHECK-LABEL: func.func @read_of_write_collapsed(
// CHECK-SAME:      %[[DATA:.+]]: tensor<1x64xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      %[[COLLAPSED:.+]] = tensor.collapse_shape %[[DATA]] {{\[}}[0, 1]] : tensor<1x64xf16> into tensor<64xf16>
// CHECK-NEXT:      "test.use"(%[[COLLAPSED]])
// CHECK-NEXT:    }
func.func @read_of_write_collapsed(%data: tensor<1x64xf16>, %slot: !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %data, %slot : tensor<1x64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A read of a reshaped write expands the written data.
// CHECK-LABEL: func.func @read_of_write_expanded(
// CHECK-SAME:      %[[DATA:.+]]: tensor<64xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      %[[EXPANDED:.+]] = tensor.expand_shape %[[DATA]] {{\[}}[0, 1]] output_shape [2, 32] : tensor<64xf16> into tensor<2x32xf16>
// CHECK-NEXT:      "test.use"(%[[EXPANDED]])
// CHECK-NEXT:    }
func.func @read_of_write_expanded(%data: tensor<64xf16>, %slot: !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<2x32xf16>
      "test.use"(%read) : (tensor<2x32xf16>) -> ()
    }
  }
  return
}

// A read of a write with neither shape a collapse of the other goes through
// the flat shape.
// CHECK-LABEL: func.func @read_of_write_regrouped(
// CHECK-SAME:      %[[DATA:.+]]: tensor<2x32xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      %[[FLAT:.+]] = tensor.collapse_shape %[[DATA]] {{\[}}[0, 1]] : tensor<2x32xf16> into tensor<64xf16>
// CHECK-NEXT:      %[[EXPANDED:.+]] = tensor.expand_shape %[[FLAT]] {{\[}}[0, 1]] output_shape [4, 16] : tensor<64xf16> into tensor<4x16xf16>
// CHECK-NEXT:      "test.use"(%[[EXPANDED]])
// CHECK-NEXT:    }
func.func @read_of_write_regrouped(%data: tensor<2x32xf16>, %slot: !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %data, %slot : tensor<2x32xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<4x16xf16>
      "test.use"(%read) : (tensor<4x16xf16>) -> ()
    }
  }
  return
}

// A tensor read of a memref write is kept.
// CHECK-LABEL: func.func @read_of_write_memref(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.write_to_fifo
// CHECK-NEXT:      ktdf.read_from_fifo
func.func @read_of_write_memref(%data: memref<64xf16>, %slot: !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %data, %slot : memref<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A splat read is kept.
// CHECK-LABEL: func.func @read_of_write_splat(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.write_to_fifo
// CHECK-NEXT:      ktdf.read_from_fifo {{.+}} {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
func.func @read_of_write_splat(%data: tensor<64xf16>, %slot: !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      %read = ktdf.read_from_fifo %slot {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>} : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A read of a slot written in another stage is kept.
// CHECK-LABEL: func.func @read_of_write_other_stage(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.write_to_fifo
// CHECK-NEXT:    }
// CHECK-NEXT:    ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.read_from_fifo
func.func @read_of_write_other_stage(%slot: !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>, %data: tensor<64xf16>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A read of a slot filled from another FIFO in the same stage reads that FIFO.
// CHECK-LABEL: func.func @read_of_fifo_transfer(
// CHECK:         %[[SLOTS:.+]]:2 = ktdf.private
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      %[[READ:.+]] = ktdf.read_from_fifo %[[SLOTS]]#0
// CHECK-NEXT:      "test.use"(%[[READ]])
// CHECK-NEXT:    }
func.func @read_of_fifo_transfer() {
  ktdf.pipeline {
    %slots:2 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
      %src = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      %dst = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      ktdf.private_yield %src, %dst : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %slots#0 size [64] to %slots#1 size [64] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      %read = ktdf.read_from_fifo %slots#1 : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A read of a slot filled from a FIFO of unknown allocation is kept.
// CHECK-LABEL: func.func @read_of_unallocated_fifo_transfer(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer
// CHECK-NEXT:      ktdf.read_from_fifo
func.func @read_of_unallocated_fifo_transfer(%src: !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, %dst: !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src size [64] to %dst size [64] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      %read = ktdf.read_from_fifo %dst : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A read of a slot filled from memory in the same stage is kept.
// CHECK-LABEL: func.func @read_of_memory_transfer(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer
// CHECK-NEXT:      ktdf.read_from_fifo
func.func @read_of_memory_transfer(%src: memref<64xf16>, %dst: !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [64] to %dst size [64] : memref<64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      %read = ktdf.read_from_fifo %dst : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A transfer out of a slot filled by a transfer in the same stage transfers
// from the original source.
// CHECK-LABEL: func.func @transfer_of_transfer(
// CHECK-SAME:      %[[SRC:.+]]: memref<64xf16>, %[[DST:.+]]: memref<64xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer from %[[SRC]]{{.+}} to %[[DST]]
// CHECK-NEXT:    }
func.func @transfer_of_transfer(%src: memref<64xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %hop = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [64] to %hop size [64] : memref<64xf16>, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      ktdf.data_transfer from %hop size [64] to %dst[%c0] size [64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<64xf16>
    }
  }
  return
}

// A transfer out of a slot of unknown allocation is kept.
// CHECK-LABEL: func.func @transfer_of_unallocated_transfer(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer
// CHECK-NEXT:      ktdf.data_transfer
func.func @transfer_of_unallocated_transfer(%src: memref<64xf16>, %hop: !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [64] to %hop size [64] : memref<64xf16>, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      ktdf.data_transfer from %hop size [64] to %dst[%c0] size [64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<64xf16>
    }
  }
  return
}

// A transfer out of a slot filled in another stage is kept.
// CHECK-LABEL: func.func @transfer_of_transfer_other_stage(
// CHECK-SAME:      %[[SRC:.+]]: memref<64xf16>, %[[HOP:.+]]: !ktdf.fifo.slot<{{.+}}>, %[[DST:.+]]: memref<64xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer from %[[SRC]]{{.+}} to %[[HOP]]
// CHECK-NEXT:    }
// CHECK-NEXT:    ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer from %[[HOP]]
func.func @transfer_of_transfer_other_stage(%src: memref<64xf16>, %hop: !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [64] to %hop size [64] : memref<64xf16>, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %hop size [64] to %dst[%c0] size [64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<64xf16>
    }
  }
  return
}

// A read that precedes the write is kept.
// CHECK-LABEL: func.func @read_before_write(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      %[[READ:.+]] = ktdf.read_from_fifo
// CHECK-NEXT:      %[[DATA:.+]] = "test.def"()
// CHECK-NEXT:      ktdf.write_to_fifo %[[DATA]]
// CHECK-NEXT:      "test.use"(%[[READ]])
// CHECK-NEXT:    }
func.func @read_before_write(%slot: !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      %data = "test.def"() : () -> tensor<64xf16>
      ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A read whose data feeds the write to the same slot is kept.
// CHECK-LABEL: func.func @read_feeds_write(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      %[[READ:.+]] = ktdf.read_from_fifo
// CHECK-NEXT:      %[[DATA:.+]] = "test.def"(%[[READ]])
// CHECK-NEXT:      ktdf.write_to_fifo %[[DATA]]
// CHECK-NEXT:    }
func.func @read_feeds_write(%slot: !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      %data = "test.def"(%read) : (tensor<64xf16>) -> tensor<64xf16>
      ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
  }
  return
}

// A transfer that precedes the hop filling its slot is kept.
// CHECK-LABEL: func.func @transfer_before_hop(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer from %{{.+}} size [64] to
// CHECK-NEXT:      ktdf.data_transfer from %{{.+}} to
// CHECK-NEXT:    }
func.func @transfer_before_hop(%src: memref<64xf16>, %hop: !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %hop size [64] to %dst[%c0] size [64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<64xf16>
      ktdf.data_transfer from %src[%c0] size [64] to %hop size [64] : memref<64xf16>, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
    }
  }
  return
}

// A read of a dynamically sized private slot written in the same stage uses
// the written data.
// CHECK-LABEL: func.func @read_of_write_private(
// CHECK-SAME:      %{{.+}}: index, %[[DATA:.+]]: tensor<?xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      "test.use"(%[[DATA]])
// CHECK-NEXT:    }
func.func @read_of_write_private(%size: index, %data: tensor<?xf16>) {
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", ?xf16>) {
      %alloc = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %data, %slot : tensor<?xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16>
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16> -> tensor<?xf16>
      "test.use"(%read) : (tensor<?xf16>) -> ()
    }
  }
  return
}

// A read of a slot written in the same loop iteration uses the written data.
// CHECK-LABEL: func.func @read_of_write_in_loop(
// CHECK-SAME:      %{{.+}}: !ktdf.fifo.slot<{{.+}}>, %[[DATA:.+]]: tensor<64xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      scf.for {{.+}} {
// CHECK-NEXT:        "test.use"(%[[DATA]])
// CHECK-NEXT:      }
// CHECK-NEXT:    }
func.func @read_of_write_in_loop(%slot: !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>, %data: tensor<64xf16>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      scf.for %i = %c0 to %n step %c1 {
        ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
        %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
        "test.use"(%read) : (tensor<64xf16>) -> ()
      }
    }
  }
  return
}

// A read of a slot filled with part of a larger FIFO is kept.
// CHECK-LABEL: func.func @read_of_partial_fifo_transfer(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer
// CHECK-NEXT:      ktdf.read_from_fifo
func.func @read_of_partial_fifo_transfer(%src: !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, %dst: !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src size [64] to %dst size [64] : !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      %read = ktdf.read_from_fifo %dst : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A read of a slot filled with a whole dynamically sized FIFO reads that FIFO.
// CHECK-LABEL: func.func @read_of_dynamic_fifo_transfer(
// CHECK:         ktdf.pipeline {
// CHECK:           %[[SLOTS:.+]]:2 = ktdf.private
// CHECK:           ktdf.stage {{.+}} {
// CHECK-NEXT:        %[[READ:.+]] = ktdf.read_from_fifo %[[SLOTS]]#0
// CHECK-NEXT:        "test.use"(%[[READ]])
// CHECK-NEXT:      }
func.func @read_of_dynamic_fifo_transfer(%size: index) {
  ktdf.pipeline {
    %slots:2 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16>) {
      %src = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>
      %dst = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16>
      ktdf.private_yield %src, %dst : !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %slots#0 size [%size] to %slots#1 size [%size] : !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16>
      %read = ktdf.read_from_fifo %slots#1 : !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16> -> tensor<?xf16>
      "test.use"(%read) : (tensor<?xf16>) -> ()
    }
  }
  return
}

// A read of a slot filled from a FIFO with a size not known to be its whole is
// kept.
// CHECK-LABEL: func.func @read_of_unknown_fifo_transfer(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer
// CHECK-NEXT:      ktdf.read_from_fifo
func.func @read_of_unknown_fifo_transfer(%size: index, %other: index) {
  ktdf.pipeline {
    %slots:2 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16>) {
      %src = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>
      %dst = ktdf.fifo.allocate(%other) -> !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16>
      ktdf.private_yield %src, %dst : !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %slots#0 size [%other] to %slots#1 size [%other] : !ktdf.fifo.slot<"L1LU" -> "SFU", ?xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16>
      %read = ktdf.read_from_fifo %slots#1 : !ktdf.fifo.slot<"SFU" -> "SFU", ?xf16> -> tensor<?xf16>
      "test.use"(%read) : (tensor<?xf16>) -> ()
    }
  }
  return
}
