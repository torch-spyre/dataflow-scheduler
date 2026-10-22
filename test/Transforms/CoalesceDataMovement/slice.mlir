// RUN: dataflow-scheduler-opt --ktdf-coalesce-data-movement --allow-unregistered-dialect %s | FileCheck %s

// Folding of slices of FIFO data into the other end of the FIFO, resizing the
// slot to the slice.

// A slice of a read moves to the write, and the slot shrinks to the slice.
// CHECK-LABEL: func.func @extract_of_read(
// CHECK-SAME:      %[[DATA:.+]]: tensor<64xf16>
// CHECK:         %[[SLOT:.+]] = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", 16xf16>) {
// CHECK-NEXT:      ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 16xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      %[[SLICE:.+]] = tensor.extract_slice %[[DATA]][16] [16] [1]
// CHECK-NEXT:      ktdf.write_to_fifo %[[SLICE]], %[[SLOT]] : tensor<16xf16>
// CHECK-NEXT:    }
// CHECK-NEXT:    ktdf.stage {{.+}} {
// CHECK-NEXT:      %[[READ:.+]] = ktdf.read_from_fifo %[[SLOT]] : {{.+}} -> tensor<16xf16>
// CHECK-NEXT:      "test.use"(%[[READ]])
// CHECK-NEXT:    }
func.func @extract_of_read(%data: tensor<64xf16>) {
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      %slice = tensor.extract_slice %read[16] [16] [1] : tensor<64xf16> to tensor<16xf16>
      "test.use"(%slice) : (tensor<16xf16>) -> ()
    }
  }
  return
}

// A slice of a splat read is kept.
// CHECK-LABEL: func.func @extract_of_splat_read(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.write_to_fifo %{{.+}} : tensor<64xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.read_from_fifo
// CHECK-NEXT:      tensor.extract_slice
func.func @extract_of_splat_read(%data: tensor<64xf16>) {
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>} : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      %slice = tensor.extract_slice %read[16] [16] [1] : tensor<64xf16> to tensor<16xf16>
      "test.use"(%slice) : (tensor<16xf16>) -> ()
    }
  }
  return
}

// A slice of a read with other users is kept.
// CHECK-LABEL: func.func @extract_of_shared_read(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.write_to_fifo %{{.+}} : tensor<64xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.read_from_fifo
// CHECK-NEXT:      tensor.extract_slice
func.func @extract_of_shared_read(%data: tensor<64xf16>) {
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      %slice = tensor.extract_slice %read[16] [16] [1] : tensor<64xf16> to tensor<16xf16>
      "test.use"(%read, %slice) : (tensor<64xf16>, tensor<16xf16>) -> ()
    }
  }
  return
}

// A dynamic slice of a read is kept.
// CHECK-LABEL: func.func @extract_of_read_dynamic(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.write_to_fifo %{{.+}} : tensor<64xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.read_from_fifo
// CHECK-NEXT:      tensor.extract_slice
func.func @extract_of_read_dynamic(%data: tensor<64xf16>, %size: index) {
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      %slice = tensor.extract_slice %read[0] [%size] [1] : tensor<64xf16> to tensor<?xf16>
      "test.use"(%slice) : (tensor<?xf16>) -> ()
    }
  }
  return
}

// A slice of a read at an offset only known where it is read is kept.
// CHECK-LABEL: func.func @extract_of_read_local_offset(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.write_to_fifo %{{.+}} : tensor<64xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      "test.def"
// CHECK-NEXT:      ktdf.read_from_fifo
// CHECK-NEXT:      tensor.extract_slice
func.func @extract_of_read_local_offset(%data: tensor<64xf16>) {
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %data, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %offset = "test.def"() : () -> index
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      %slice = tensor.extract_slice %read[%offset] [16] [1] : tensor<64xf16> to tensor<16xf16>
      "test.use"(%slice) : (tensor<16xf16>) -> ()
    }
  }
  return
}

// A slice inserted into an empty moves to the read, and the slot shrinks to the
// slice.
// CHECK-LABEL: func.func @insert_into_write(
// CHECK-SAME:      %[[DATA:.+]]: tensor<16xf16>
// CHECK:         %[[SLOT:.+]] = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", 16xf16>) {
// CHECK-NEXT:      ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 16xf16>
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      ktdf.write_to_fifo %[[DATA]], %[[SLOT]] : tensor<16xf16>
// CHECK-NEXT:    }
// CHECK-NEXT:    ktdf.stage {{.+}} {
// CHECK-NEXT:      %[[READ:.+]] = ktdf.read_from_fifo %[[SLOT]] : {{.+}} -> tensor<16xf16>
// CHECK-NEXT:      %[[EMPTY:.+]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:      %[[INSERTED:.+]] = tensor.insert_slice %[[READ]] into %[[EMPTY]][16] [16] [1]
// CHECK-NEXT:      "test.use"(%[[INSERTED]])
// CHECK-NEXT:    }
func.func @insert_into_write(%data: tensor<16xf16>) {
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %empty = tensor.empty() : tensor<64xf16>
      %inserted = tensor.insert_slice %data into %empty[16] [16] [1] : tensor<16xf16> into tensor<64xf16>
      ktdf.write_to_fifo %inserted, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A slice inserted into other data is kept.
// CHECK-LABEL: func.func @insert_into_data_write(
// CHECK:         ktdf.stage {{.+}} {
// CHECK-NEXT:      tensor.insert_slice
// CHECK-NEXT:      ktdf.write_to_fifo %{{.+}} : tensor<64xf16>
func.func @insert_into_data_write(%data: tensor<16xf16>, %base: tensor<64xf16>) {
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %inserted = tensor.insert_slice %data into %base[16] [16] [1] : tensor<16xf16> into tensor<64xf16>
      ktdf.write_to_fifo %inserted, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A slice inserted into an empty written to a splat read is kept.
// CHECK-LABEL: func.func @insert_into_write_splat_read(
// CHECK:         ktdf.stage {{.+}} {
// CHECK:           tensor.insert_slice
// CHECK-NEXT:      ktdf.write_to_fifo %{{.+}} : tensor<64xf16>
func.func @insert_into_write_splat_read(%data: tensor<16xf16>) {
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %empty = tensor.empty() : tensor<64xf16>
      %inserted = tensor.insert_slice %data into %empty[16] [16] [1] : tensor<16xf16> into tensor<64xf16>
      ktdf.write_to_fifo %inserted, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>} : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}

// A slice inserted at an offset only known where it is written is kept.
// CHECK-LABEL: func.func @insert_into_write_local_offset(
// CHECK:         ktdf.stage {{.+}} {
// CHECK:           tensor.insert_slice
// CHECK-NEXT:      ktdf.write_to_fifo %{{.+}} : tensor<64xf16>
func.func @insert_into_write_local_offset(%data: tensor<16xf16>) {
  ktdf.pipeline {
    %slot = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>) {
      %alloc = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
      ktdf.private_yield %alloc : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %offset = "test.def"() : () -> index
      %empty = tensor.empty() : tensor<64xf16>
      %inserted = tensor.insert_slice %data into %empty[%offset] [16] [1] : tensor<16xf16> into tensor<64xf16>
      ktdf.write_to_fifo %inserted, %slot : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"SFU" -> "SFU", 64xf16> -> tensor<64xf16>
      "test.use"(%read) : (tensor<64xf16>) -> ()
    }
  }
  return
}
