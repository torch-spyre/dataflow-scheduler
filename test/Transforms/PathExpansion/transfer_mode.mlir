// RUN: dataflow-scheduler-opt --path-expansion --split-input-file %s | FileCheck %s

// Tests expanding a transfer that is not a copy. The adapted transfer keeps the
// mode and the change of shape, and the hops after it are plain copies.

#map = affine_map<(d0) -> (d0)>

ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

// A splat fills the buffer that replaces its FIFO, and the hop from it copies.
// CHECK-LABEL: func.func @splat_on_adapted_hop
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:8 = ktdf.private
// CHECK:             memref.alloc() : memref<64xf16, "L1">
// CHECK:           ktdf.stage {{.+}} {
// CHECK-NEXT:        ktdf.data_transfer from %{{.+}}[0] size [1] to %[[P]]#0[0] size [64] {transfer_mode = #ktdf.transfer_mode<splat>} :
// CHECK:           } {applicable_units = ["MNILU"]}
// CHECK:           ktdf.stage {{.+}} {
// CHECK-NEXT:        ktdf.data_transfer from %[[P]]#0[0] size [64] to %[[P]]#1 size [64] :
// CHECK:           } {applicable_units = ["L1LU"]}
func.func @splat_on_adapted_hop(%src: memref<64xf16, "DDR">, %dst: memref<64xf16, "DDR">) {
  ktdf.pipeline {
    %p:4 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token) {
      %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      ktdf.private_yield %f0, %f1, %t0, %t1 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%p#2) {
      ktdf.data_transfer from %src[0] size [1] to %p#0 size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(%p#2) depends_out(%p#3) {
      %0 = ktdf.read_from_fifo %p#0 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
      %1 = tensor.empty() : tensor<64xf16>
      %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<64xf16>) outs(%1 : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %3 = math.sqrt %in : f16
        linalg.yield %3 : f16
      } -> tensor<64xf16>
      ktdf.write_to_fifo %2, %p#1 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
    } {applicable_units = ["SFU"]}
    ktdf.stage depends_in(%p#3) depends_out(none) {
      ktdf.data_transfer from %p#1 size [64] to %dst[0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
    }
  }
  return
}

// -----

#map = affine_map<(d0) -> (d0)>

ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

// A pad fills the buffer that replaces its FIFO, and the hop from it copies.
// CHECK-LABEL: func.func @pad_on_adapted_hop
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:8 = ktdf.private
// CHECK:             memref.alloc() : memref<64xf16, "L1">
// CHECK:           ktdf.stage {{.+}} {
// CHECK-NEXT:        ktdf.data_transfer from %{{.+}}[0] size [16] to %[[P]]#0[0] size [64] {transfer_mode = #ktdf.transfer_mode<zero_pad_back>} :
// CHECK:           } {applicable_units = ["MNILU"]}
// CHECK:           ktdf.stage {{.+}} {
// CHECK-NEXT:        ktdf.data_transfer from %[[P]]#0[0] size [64] to %[[P]]#1 size [64] :
// CHECK:           } {applicable_units = ["L1LU"]}
func.func @pad_on_adapted_hop(%src: memref<64xf16, "DDR">, %dst: memref<64xf16, "DDR">) {
  ktdf.pipeline {
    %p:4 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token) {
      %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
      %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      ktdf.private_yield %f0, %f1, %t0, %t1 : !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%p#2) {
      ktdf.data_transfer from %src[0] size [16] to %p#0 size [64] {transfer_mode = #ktdf.transfer_mode<zero_pad_back>} : memref<64xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
    }
    ktdf.stage depends_in(%p#2) depends_out(%p#3) {
      %0 = ktdf.read_from_fifo %p#0 : <"DDR" -> "SFU", 64xf16> -> tensor<64xf16>
      %1 = tensor.empty() : tensor<64xf16>
      %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%0 : tensor<64xf16>) outs(%1 : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %3 = math.sqrt %in : f16
        linalg.yield %3 : f16
      } -> tensor<64xf16>
      ktdf.write_to_fifo %2, %p#1 : tensor<64xf16>, <"SFU" -> "DDR", 64xf16>
    } {applicable_units = ["SFU"]}
    ktdf.stage depends_in(%p#3) depends_out(none) {
      ktdf.data_transfer from %p#1 size [64] to %dst[0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
    }
  }
  return
}
