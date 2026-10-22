// RUN: dataflow-scheduler-opt --ktdf-coalesce-data-movement --allow-unregistered-dialect %s | FileCheck %s

// Folding of copies relayed through a FIFO. Other transfer modes are kept.

// A transfer out of a slot filled by a splat keeps the splat.
// CHECK-LABEL: func.func @transfer_of_splat_transfer(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer {{.+}} size [1] to %{{.+}} size [64] {transfer_mode = #ktdf.transfer_mode<splat>}
// CHECK-NEXT:      ktdf.data_transfer
func.func @transfer_of_splat_transfer(%src: memref<64xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %hop = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [1] to %hop size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<64xf16>, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      ktdf.data_transfer from %hop size [64] to %dst[%c0] size [64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<64xf16>
    }
  }
  return
}

// A transfer out of a slot filled by a pad keeps the pad.
// CHECK-LABEL: func.func @transfer_of_pad_transfer(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer {{.+}} size [48] to %{{.+}} size [64] {transfer_mode = #ktdf.transfer_mode<zero_pad_back>}
// CHECK-NEXT:      ktdf.data_transfer
func.func @transfer_of_pad_transfer(%src: memref<48xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %hop = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [48] to %hop size [64] {transfer_mode = #ktdf.transfer_mode<zero_pad_back>} : memref<48xf16>, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      ktdf.data_transfer from %hop size [64] to %dst[%c0] size [64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<64xf16>
    }
  }
  return
}

// A splat out of a slot filled by a splat is kept.
// CHECK-LABEL: func.func @splat_of_splat_transfer(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer {{.+}} size [1] to %{{.+}} size [64] {transfer_mode = #ktdf.transfer_mode<splat>}
// CHECK-NEXT:      ktdf.data_transfer {{.+}} {transfer_mode = #ktdf.transfer_mode<splat>}
func.func @splat_of_splat_transfer(%src: memref<64xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %hop = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [1] to %hop size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<64xf16>, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      ktdf.data_transfer from %hop size [64] to %dst[%c0] size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<64xf16>
    }
  }
  return
}

// A pad out of a slot filled by a pad is kept.
// CHECK-LABEL: func.func @pad_of_pad_transfer(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer {{.+}} size [48] to %{{.+}} size [64] {transfer_mode = #ktdf.transfer_mode<zero_pad_back>}
// CHECK-NEXT:      ktdf.data_transfer {{.+}} {transfer_mode = #ktdf.transfer_mode<zero_pad_back>}
func.func @pad_of_pad_transfer(%src: memref<48xf16>, %dst: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %hop = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [48] to %hop size [64] {transfer_mode = #ktdf.transfer_mode<zero_pad_back>} : memref<48xf16>, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      ktdf.data_transfer from %hop size [64] to %dst[%c0] size [128] {transfer_mode = #ktdf.transfer_mode<zero_pad_back>} : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<128xf16>
    }
  }
  return
}

// A transfer out of a slot filled with as many elements from a memory of
// another rank transfers from that memory.
// CHECK-LABEL: func.func @transfer_of_reshaping_transfer(
// CHECK-SAME:      %[[SRC:.+]]: memref<2x32xf16>, %[[DST:.+]]: memref<64xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer from %[[SRC]]{{.+}} size [2, 32] to %[[DST]]
// CHECK-NEXT:    }
func.func @transfer_of_reshaping_transfer(%src: memref<2x32xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %hop = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0, %c0] size [2, 32] to %hop size [64] : memref<2x32xf16>, !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>
      ktdf.data_transfer from %hop size [64] to %dst[%c0] size [64] : !ktdf.fifo.slot<"L1LU" -> "L1SU", 64xf16>, memref<64xf16>
    }
  }
  return
}

// A transfer out of a slot filled with a dynamic number of elements transfers
// from the original source.
// CHECK-LABEL: func.func @transfer_of_dynamic_transfer(
// CHECK-SAME:      %[[SRC:.+]]: memref<?xf16>, %[[DST:.+]]: memref<?xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer from %[[SRC]]{{.+}} to %[[DST]]
// CHECK-NEXT:    }
func.func @transfer_of_dynamic_transfer(%src: memref<?xf16>, %dst: memref<?xf16>, %size: index) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %hop = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", ?xf16>) {
      %alloc = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"L1LU" -> "L1SU", ?xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"L1LU" -> "L1SU", ?xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [%size] to %hop size [%size] : memref<?xf16>, !ktdf.fifo.slot<"L1LU" -> "L1SU", ?xf16>
      ktdf.data_transfer from %hop size [%size] to %dst[%c0] size [%size] : !ktdf.fifo.slot<"L1LU" -> "L1SU", ?xf16>, memref<?xf16>
    }
  }
  return
}

// A copy out of a slot filled by a copy of dynamic sizes is trusted to move as
// many elements, and transfers from the original source.
// CHECK-LABEL: func.func @transfer_of_unproven_dynamic_transfer(
// CHECK-SAME:      %[[SRC:.+]]: memref<?xf16>, %[[DST:.+]]: memref<?xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer from %[[SRC]]{{.+}} to %[[DST]]
// CHECK-NEXT:    }
func.func @transfer_of_unproven_dynamic_transfer(%src: memref<?xf16>, %dst: memref<?xf16>, %size: index, %other: index) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %hop = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "L1SU", ?xf16>) {
      %alloc = ktdf.fifo.allocate(%size) -> !ktdf.fifo.slot<"L1LU" -> "L1SU", ?xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"L1LU" -> "L1SU", ?xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [%other] to %hop size [%size] : memref<?xf16>, !ktdf.fifo.slot<"L1LU" -> "L1SU", ?xf16>
      ktdf.data_transfer from %hop size [%size] to %dst[%c0] size [%size] : !ktdf.fifo.slot<"L1LU" -> "L1SU", ?xf16>, memref<?xf16>
    }
  }
  return
}
