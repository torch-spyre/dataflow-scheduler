// RUN: dataflow-scheduler-opt --ktdf-coalesce-data-movement --allow-unregistered-dialect %s | FileCheck %s

// Folding must not reorder accesses to slots of the same allocation, nor move
// a memory read past an effect on memory.

// A read of a slot filled from a FIFO, past an access to another allocation,
// reads that FIFO.
// CHECK-LABEL: func.func @read_of_fifo_transfer_past_other_alloc(
// CHECK:         %[[SLOTS:.+]]:3 = ktdf.private
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      %[[OTHER:.+]] = ktdf.read_from_fifo %[[SLOTS]]#2
// CHECK-NEXT:      %[[READ:.+]] = ktdf.read_from_fifo %[[SLOTS]]#0
// CHECK-NEXT:      "test.use"(%[[OTHER]], %[[READ]])
// CHECK-NEXT:    }
func.func @read_of_fifo_transfer_past_other_alloc() {
  ktdf.pipeline {
    %slots:3 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) {
      %src = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      %dst = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      %other = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      ktdf.private_yield %src, %dst, %other : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %slots#0 size [64] to %slots#1 size [64] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      %other = ktdf.read_from_fifo %slots#2 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %read = ktdf.read_from_fifo %slots#1 : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%other, %read) : (tensor<64xf16>, tensor<64xf16>) -> ()
    }
  }
  return
}

// A read of a slot filled from a FIFO, past an access to a sibling slot of
// that FIFO, is kept.
// CHECK-LABEL: func.func @read_of_fifo_transfer_past_sibling(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer
// CHECK-NEXT:      ktdf.read_from_fifo
// CHECK-NEXT:      ktdf.read_from_fifo
func.func @read_of_fifo_transfer_past_sibling() {
  ktdf.pipeline {
    %slots:3 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
      %src:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      %dst = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      ktdf.private_yield %src#0, %src#1, %dst : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %slots#0 size [64] to %slots#2 size [64] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      %sibling = ktdf.read_from_fifo %slots#1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %read = ktdf.read_from_fifo %slots#2 : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%sibling, %read) : (tensor<64xf16>, tensor<64xf16>) -> ()
    }
  }
  return
}

// A read of a slot filled from a FIFO, past a nested access to a sibling slot
// of that FIFO, is kept.
// CHECK-LABEL: func.func @read_of_fifo_transfer_past_nested_sibling(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer
// CHECK-NEXT:      scf.execute_region
// CHECK:           ktdf.read_from_fifo
func.func @read_of_fifo_transfer_past_nested_sibling() {
  ktdf.pipeline {
    %slots:3 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
      %src:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      %dst = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      ktdf.private_yield %src#0, %src#1, %dst : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %slots#0 size [64] to %slots#2 size [64] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      %sibling = scf.execute_region -> tensor<64xf16> {
        %r = ktdf.read_from_fifo %slots#1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
        scf.yield %r : tensor<64xf16>
      }
      %read = ktdf.read_from_fifo %slots#2 : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%sibling, %read) : (tensor<64xf16>, tensor<64xf16>) -> ()
    }
  }
  return
}

// A transfer out of a slot filled from a FIFO, past an access to a sibling
// slot of that FIFO, is kept.
// CHECK-LABEL: func.func @transfer_of_fifo_transfer_past_sibling(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer
// CHECK-NEXT:      ktdf.read_from_fifo
// CHECK-NEXT:      ktdf.data_transfer
func.func @transfer_of_fifo_transfer_past_sibling(%dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %slots:3 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>) {
      %src:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      %hop = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      ktdf.private_yield %src#0, %src#1, %hop : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %slots#0 size [64] to %slots#2 size [64] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      %sibling = ktdf.read_from_fifo %slots#1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      ktdf.data_transfer from %slots#2 size [64] to %dst[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64xf16>
      "test.use"(%sibling) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A transfer out of a slot filled from memory, past a store to that memory, is
// kept.
// CHECK-LABEL: func.func @transfer_of_memory_transfer_past_store(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer
// CHECK-NEXT:      memref.store
// CHECK-NEXT:      ktdf.data_transfer
func.func @transfer_of_memory_transfer_past_store(%src: memref<64xf16>, %hop: !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, %dst: memref<64xf16>, %value: f16) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [64] to %hop size [64] : memref<64xf16>, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      memref.store %value, %src[%c0] : memref<64xf16>
      ktdf.data_transfer from %hop size [64] to %dst[%c0] size [64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<64xf16>
    }
  }
  return
}
