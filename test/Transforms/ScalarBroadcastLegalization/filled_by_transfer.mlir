// RUN: dataflow-scheduler-opt --scalar-broadcast-legalization %s | FileCheck %s

// Tests a scalar broadcast operand that a transfer on its path already fills.
// That transfer and the hops after it are left as they are, and only the
// broadcast map goes.

// CHECK: #[[$ID:.+]] = affine_map<(d0) -> (d0)>

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (0)>

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<"DDR" = "DDR", "L1" = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

// A splat into a memory, followed by a copy into the FIFO, as path expansion
// leaves it.
// CHECK-LABEL: func.func @splat_before_copy
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:6 = ktdf.private
// CHECK:           ktdf.stage {{.+}} {
// CHECK-NEXT:        ktdf.data_transfer from %{{.+}}[0] size [1] to %[[P]]#0[0] size [64] {transfer_mode = #ktdf.transfer_mode<splat>} :
// CHECK:           ktdf.stage {{.+}} {
// CHECK-NEXT:        ktdf.data_transfer from %[[P]]#0[0] size [64] to %[[P]]#1 size [64] : memref
// CHECK:           ktdf.stage {{.+}} {
// CHECK:             linalg.generic {indexing_maps = [#[[$ID]], #[[$ID]]]
func.func @splat_before_copy(%src: memref<64xf16, "DDR">, %dst: memref<64xf16, "L1">) {
  ktdf.pipeline {
    %p:6 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {
      %l1 = memref.alloc() : memref<64xf16, "L1">
      %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      %t2 = ktdf.create_token : !ktdf.token
      ktdf.private_yield %l1, %f0, %f1, %t0, %t1, %t2 : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%p#3) {
      ktdf.data_transfer from %src[0] size [1] to %p#0[0] size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<64xf16, "DDR">, memref<64xf16, "L1">
    } {applicable_units = ["MNILU"]}
    ktdf.stage depends_in(%p#3) depends_out(%p#4) {
      ktdf.data_transfer from %p#0[0] size [64] to %p#1 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(%p#4) depends_out(%p#5) {
      %0 = ktdf.read_from_fifo %p#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %1 = tensor.empty() : tensor<64xf16>
      %2 = linalg.generic {indexing_maps = [#map1, #map], iterator_types = ["parallel"]} ins(%0 : tensor<64xf16>) outs(%1 : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %3 = math.sqrt %in : f16
        linalg.yield %3 : f16
      } -> tensor<64xf16>
      ktdf.write_to_fifo %2, %p#2 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
    } {applicable_units = ["SFU"]}
    ktdf.stage depends_in(%p#5) depends_out(none) {
      ktdf.data_transfer from %p#2 size [64] to %dst[0] size [64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<64xf16, "L1">
    } {applicable_units = ["L1SU"]}
  }
  return
}
