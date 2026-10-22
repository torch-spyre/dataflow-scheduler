// RUN: dataflow-scheduler-opt --ktdf-coalesce-data-movement --allow-unregistered-dialect %s | FileCheck %s

// Folding of slices of FIFO data into the copy that fills or drains the slot,
// resizing the slot to the slice.

// A slice of a read folds into the copy filling the slot.
// CHECK-LABEL: func.func @fill(
// CHECK-SAME:      %[[SRC:.+]]: memref<128xf16>
// CHECK:         %[[SLOT:.+]] = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 16xf16>) {
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer from %[[SRC]][16] size [16] to %[[SLOT]] size [16]
// CHECK-NEXT:    }
// CHECK-NEXT:    ktdf.stage {{.+}} {
// CHECK-NEXT:      %[[READ:.+]] = ktdf.read_from_fifo %[[SLOT]] : {{.+}} -> tensor<16xf16>
// CHECK-NEXT:      "test.use"(%[[READ]])
func.func @fill(%src: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [64] to %slot size [64] : memref<128xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %slice = tensor.extract_slice %read[16] [16] [1] : tensor<64xf16> to tensor<16xf16>
      "test.use"(%slice) : (tensor<16xf16>) -> ()
    }
  }
  return
}

// A slice of a read of a reshaped region maps across its unit dimensions.
// CHECK-LABEL: func.func @fill_unit_dims(
// CHECK-SAME:      %[[SRC:.+]]: memref<4x64xf16>
// CHECK:         ktdf.data_transfer from %[[SRC]][2, 16] size [1, 16] to %{{.+}} size [16]
func.func @fill_unit_dims(%src: memref<4x64xf16>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c2, %c0] size [1, 64] to %slot size [64] : memref<4x64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
      %slice = tensor.extract_slice %read[0, 16] [1, 16] [1, 1] : tensor<1x64xf16> to tensor<1x16xf16>
      "test.use"(%slice) : (tensor<1x16xf16>) -> ()
    }
  }
  return
}

// A slice of a read of a splat is kept.
// CHECK-LABEL: func.func @fill_splat(
// CHECK:         ktdf.data_transfer {{.+}} size [64] {transfer_mode = #ktdf.transfer_mode<splat>}
// CHECK:         tensor.extract_slice
func.func @fill_splat(%src: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [1] to %slot size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<128xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %slice = tensor.extract_slice %read[16] [16] [1] : tensor<64xf16> to tensor<16xf16>
      "test.use"(%slice) : (tensor<16xf16>) -> ()
    }
  }
  return
}

// A strided slice of a read is kept, since the copy must stay contiguous.
// CHECK-LABEL: func.func @fill_strided(
// CHECK:         ktdf.data_transfer {{.+}} size [64] to
// CHECK:         tensor.extract_slice
func.func @fill_strided(%src: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [64] to %slot size [64] : memref<128xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %slice = tensor.extract_slice %read[0] [16] [2] : tensor<64xf16> to tensor<16xf16>
      "test.use"(%slice) : (tensor<16xf16>) -> ()
    }
  }
  return
}

// A slice of a read at an offset only known where it is read is kept.
// CHECK-LABEL: func.func @fill_local_offset(
// CHECK:         ktdf.data_transfer {{.+}} size [64] to
// CHECK:         tensor.extract_slice
func.func @fill_local_offset(%src: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %src[%c0] size [64] to %slot size [64] : memref<128xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %offset = "test.def"() : () -> index
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %slice = tensor.extract_slice %read[%offset] [16] [1] : tensor<64xf16> to tensor<16xf16>
      "test.use"(%slice) : (tensor<16xf16>) -> ()
    }
  }
  return
}

// A slice inserted into an empty folds into the copy draining the slot.
// CHECK-LABEL: func.func @drain(
// CHECK-SAME:      %[[DATA:.+]]: tensor<16xf16>, %[[DST:.+]]: memref<128xf16>
// CHECK:         %[[SLOT:.+]] = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1SU", 16xf16>) {
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.write_to_fifo %[[DATA]], %[[SLOT]] : tensor<16xf16>
// CHECK-NEXT:    }
// CHECK-NEXT:    ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.data_transfer from %[[SLOT]] size [16] to %[[DST]][16] size [16]
func.func @drain(%data: tensor<16xf16>, %dst: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %empty = tensor.empty() : tensor<64xf16>
      %inserted = tensor.insert_slice %data into %empty[16] [16] [1] : tensor<16xf16> into tensor<64xf16>
      ktdf.write_to_fifo %inserted, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %slot size [64] to %dst[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<128xf16>
    }
  }
  return
}

// A slice inserted into a pad drained to memory is kept.
// CHECK-LABEL: func.func @drain_pad(
// CHECK:         tensor.insert_slice
// CHECK:         ktdf.data_transfer {{.+}} {transfer_mode = #ktdf.transfer_mode<zero_pad_back>}
func.func @drain_pad(%data: tensor<16xf16>, %dst: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %empty = tensor.empty() : tensor<64xf16>
      %inserted = tensor.insert_slice %data into %empty[16] [16] [1] : tensor<16xf16> into tensor<64xf16>
      ktdf.write_to_fifo %inserted, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %slot size [64] to %dst[%c0] size [128] {transfer_mode = #ktdf.transfer_mode<zero_pad_back>} : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<128xf16>
    }
  }
  return
}

// A slice inserted into other data is kept.
// CHECK-LABEL: func.func @drain_into_data(
// CHECK:         tensor.insert_slice
// CHECK:         ktdf.write_to_fifo %{{.+}} : tensor<64xf16>
func.func @drain_into_data(%data: tensor<16xf16>, %base: tensor<64xf16>, %dst: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %inserted = tensor.insert_slice %data into %base[16] [16] [1] : tensor<16xf16> into tensor<64xf16>
      ktdf.write_to_fifo %inserted, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %slot size [64] to %dst[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<128xf16>
    }
  }
  return
}

// A slice inserted at an offset only known where it is written is kept.
// CHECK-LABEL: func.func @drain_local_offset(
// CHECK:         tensor.insert_slice
// CHECK:         ktdf.write_to_fifo %{{.+}} : tensor<64xf16>
func.func @drain_local_offset(%data: tensor<16xf16>, %dst: memref<128xf16>) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %offset = "test.def"() : () -> index
      %empty = tensor.empty() : tensor<64xf16>
      %inserted = tensor.insert_slice %data into %empty[%offset] [16] [1] : tensor<16xf16> into tensor<64xf16>
      ktdf.write_to_fifo %inserted, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %slot size [64] to %dst[%c0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<128xf16>
    }
  }
  return
}
