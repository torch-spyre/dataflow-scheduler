// RUN: dataflow-scheduler-opt --ktdf-coalesce-data-movement %s | FileCheck %s

// Folding of reshapes into FIFO reads and writes.

// A read reshaped by its only user yields the reshaped type.
// CHECK-LABEL: func.func @read_reshaped(
// CHECK-SAME:      %[[SLOT:.+]]: !ktdf.fifo.slot
// CHECK:         %[[READ:.+]] = ktdf.read_from_fifo %[[SLOT]] : {{.+}} -> tensor<64xf16>
// CHECK-NOT:     tensor.collapse_shape
// CHECK:         return %[[READ]]
func.func @read_reshaped(%slot: !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) -> tensor<64xf16> {
  %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
  %reshaped = tensor.collapse_shape %read [[0, 1]] : tensor<1x64xf16> into tensor<64xf16>
  return %reshaped : tensor<64xf16>
}

// A read with users besides the reshape is kept as is.
// CHECK-LABEL: func.func @read_reshaped_and_used(
// CHECK:         %[[READ:.+]] = ktdf.read_from_fifo {{.+}} -> tensor<1x64xf16>
// CHECK:         %[[RESHAPED:.+]] = tensor.collapse_shape %[[READ]]
// CHECK:         return %[[READ]], %[[RESHAPED]]
func.func @read_reshaped_and_used(%slot: !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>) -> (tensor<1x64xf16>, tensor<64xf16>) {
  %read = ktdf.read_from_fifo %slot : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
  %reshaped = tensor.collapse_shape %read [[0, 1]] : tensor<1x64xf16> into tensor<64xf16>
  return %read, %reshaped : tensor<1x64xf16>, tensor<64xf16>
}

// A write of reshaped data writes the data before the reshape.
// CHECK-LABEL: func.func @write_reshaped(
// CHECK-SAME:      %[[DATA:.+]]: tensor<64xf16>, %[[SLOT:.+]]: !ktdf.fifo.slot
// CHECK-NOT:     tensor.expand_shape
// CHECK:         ktdf.write_to_fifo %[[DATA]], %[[SLOT]] : tensor<64xf16>
func.func @write_reshaped(%data: tensor<64xf16>, %slot: !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>) {
  %reshaped = tensor.expand_shape %data [[0, 1]] output_shape [1, 64] : tensor<64xf16> into tensor<1x64xf16>
  ktdf.write_to_fifo %reshaped, %slot : tensor<1x64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
  return
}
