// RUN: dataflow-scheduler-opt --ktdf-coalesce-data-movement --allow-unregistered-dialect %s | FileCheck %s

// Folding of `memref.subview` into the memory side of copies.

// A subview of the source folds into its offsets.
// CHECK-LABEL: func.func @subview_of_source(
// CHECK-SAME:      %[[SRC:.+]]: memref<128xf16>, %[[DST:.+]]: memref<64xf16>
// CHECK-NOT:     memref.subview
// CHECK:         ktdf.data_transfer from %[[SRC]][32] size [64] to %[[DST]]{{.+}} size [64]
func.func @subview_of_source(%src: memref<128xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  %view = memref.subview %src[32] [64] [1] : memref<128xf16> to memref<64xf16, strided<[1], offset: 32>>
  ktdf.data_transfer from %view[%c0] size [64] to %dst[%c0] size [64] : memref<64xf16, strided<[1], offset: 32>>, memref<64xf16>
  return
}

// A subview of the destination folds into its offsets.
// CHECK-LABEL: func.func @subview_of_destination(
// CHECK-SAME:      %[[SRC:.+]]: memref<64xf16>, %[[DST:.+]]: memref<128xf16>
// CHECK-NOT:     memref.subview
// CHECK:         ktdf.data_transfer from %[[SRC]]{{.+}} size [64] to %[[DST]][64] size [64]
func.func @subview_of_destination(%src: memref<64xf16>, %dst: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  %view = memref.subview %dst[64] [64] [1] : memref<128xf16> to memref<64xf16, strided<[1], offset: 64>>
  ktdf.data_transfer from %src[%c0] size [64] to %view[%c0] size [64] : memref<64xf16>, memref<64xf16, strided<[1], offset: 64>>
  return
}

// A subview at a dynamic offset composes with the offset of the transfer.
// CHECK-LABEL: func.func @subview_dynamic_offset(
// CHECK-SAME:      %[[SRC:.+]]: memref<?xf16>, %[[DST:.+]]: memref<32xf16>, %[[OFFSET:.+]]: index, %[[INDEX:.+]]: index
// CHECK-NOT:     memref.subview
// CHECK:         ktdf.data_transfer from %[[SRC]][%[[OFFSET]] + %[[INDEX]]] size [32]
func.func @subview_dynamic_offset(%src: memref<?xf16>, %dst: memref<32xf16>, %offset: index, %index: index) {
  %c0 = arith.constant 0 : index
  %view = memref.subview %src[%offset] [64] [1] : memref<?xf16> to memref<64xf16, strided<[1], offset: ?>>
  ktdf.data_transfer from %view[%index] size [32] to %dst[%c0] size [32] : memref<64xf16, strided<[1], offset: ?>>, memref<32xf16>
  return
}

// A strided subview folds where the transfer reads one element per stride.
// CHECK-LABEL: func.func @subview_strided_unit(
// CHECK-SAME:      %[[SRC:.+]]: memref<8x64xf16>, %[[DST:.+]]: memref<1x64xf16>
// CHECK-NOT:     memref.subview
// CHECK:         ktdf.data_transfer from %[[SRC]][6, 0] size [1, 64]
func.func @subview_strided_unit(%src: memref<8x64xf16>, %dst: memref<1x64xf16>) {
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %view = memref.subview %src[0, 0] [4, 64] [2, 1] : memref<8x64xf16> to memref<4x64xf16, strided<[128, 1]>>
  ktdf.data_transfer from %view[%c3, %c0] size [1, 64] to %dst[%c0, %c0] size [1, 64] : memref<4x64xf16, strided<[128, 1]>>, memref<1x64xf16>
  return
}

// A strided subview is kept where the transfer reads across the stride.
// CHECK-LABEL: func.func @subview_strided(
// CHECK:         %[[VIEW:.+]] = memref.subview
// CHECK:         ktdf.data_transfer from %[[VIEW]]
func.func @subview_strided(%src: memref<128xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  %view = memref.subview %src[0] [64] [2] : memref<128xf16> to memref<64xf16, strided<[2]>>
  ktdf.data_transfer from %view[%c0] size [64] to %dst[%c0] size [64] : memref<64xf16, strided<[2]>>, memref<64xf16>
  return
}

// A rank-reducing subview is kept.
// CHECK-LABEL: func.func @subview_rank_reducing(
// CHECK:         %[[VIEW:.+]] = memref.subview
// CHECK:         ktdf.data_transfer from %[[VIEW]]
func.func @subview_rank_reducing(%src: memref<2x64xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  %view = memref.subview %src[1, 0] [1, 64] [1, 1] : memref<2x64xf16> to memref<64xf16, strided<[1], offset: 64>>
  ktdf.data_transfer from %view[%c0] size [64] to %dst[%c0] size [64] : memref<64xf16, strided<[1], offset: 64>>, memref<64xf16>
  return
}

// A subview of the source of a splat is kept.
// CHECK-LABEL: func.func @subview_of_splat(
// CHECK:         %[[VIEW:.+]] = memref.subview
// CHECK:         ktdf.data_transfer from %[[VIEW]]
func.func @subview_of_splat(%src: memref<128xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  %view = memref.subview %src[32] [64] [1] : memref<128xf16> to memref<64xf16, strided<[1], offset: 32>>
  ktdf.data_transfer from %view[%c0] size [1] to %dst[%c0] size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<64xf16, strided<[1], offset: 32>>, memref<64xf16>
  return
}

// A subview of the memory side of a FIFO transfer folds into its offsets.
// CHECK-LABEL: func.func @subview_into_fifo(
// CHECK-SAME:      %[[SRC:.+]]: memref<128xf16>
// CHECK-NOT:     memref.subview
// CHECK:         ktdf.data_transfer from %[[SRC]][32] size [64] to
func.func @subview_into_fifo(%src: memref<128xf16>, %slot: !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) {
  %c0 = arith.constant 0 : index
  %view = memref.subview %src[32] [64] [1] : memref<128xf16> to memref<64xf16, strided<[1], offset: 32>>
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %view[%c0] size [64] to %slot size [64] : memref<64xf16, strided<[1], offset: 32>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
  }
  return
}

// A subview with other users folds, and is kept for them.
// CHECK-LABEL: func.func @subview_shared(
// CHECK-SAME:      %[[SRC:.+]]: memref<128xf16>
// CHECK:         %[[VIEW:.+]] = memref.subview
// CHECK:         ktdf.data_transfer from %[[SRC]][32] size [64]
// CHECK:         "test.use"(%[[VIEW]])
func.func @subview_shared(%src: memref<128xf16>, %dst: memref<64xf16>) {
  %c0 = arith.constant 0 : index
  %view = memref.subview %src[32] [64] [1] : memref<128xf16> to memref<64xf16, strided<[1], offset: 32>>
  ktdf.data_transfer from %view[%c0] size [64] to %dst[%c0] size [64] : memref<64xf16, strided<[1], offset: 32>>, memref<64xf16>
  "test.use"(%view) : (memref<64xf16, strided<[1], offset: 32>>) -> ()
  return
}
