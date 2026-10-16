// When every slot of a FIFO is read by a transfer into memory, nothing in the
// consumer says which comes first, so they follow the producer: the first result
// is the first slot written and the first read, whatever order they were built in.

// RUN: dataflow-scheduler-opt -order-fifo-slots-by-use %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

// A compute with two results, each stored to its own output. The writes and the
// transfers were built in reverse of the results, and agree with each other.
// CHECK-LABEL:   func.func @two_results(
// CHECK:           ktdf.stage
// CHECK:             %[[R:.*]]:2 = linalg.generic
// CHECK:             ktdf.write_to_fifo %[[R]]#0, %[[P:.*]]#0 :
// CHECK-NEXT:        ktdf.write_to_fifo %[[R]]#1, %[[P]]#1 :
// CHECK:           ktdf.stage
// CHECK-NEXT:        ktdf.data_transfer from %[[P]]#0 size [64] to %arg1
// CHECK-NEXT:        ktdf.data_transfer from %[[P]]#1 size [64] to %arg2
func.func @two_results(%in: tensor<2x64xf16>, %out_0: memref<64xf16, "DDR">, %out_1: memref<64xf16, "DDR">) {
  ktdf.pipeline {
    %p:3 = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token) {
      %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
      %t = ktdf.create_token : !ktdf.token
      ktdf.private_yield %slots#0, %slots#1, %t : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(%p#2) {
      %init_0 = tensor.empty() : tensor<64xf16>
      %init_1 = tensor.empty() : tensor<64xf16>
      %r:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["reduction", "parallel"]} ins(%in : tensor<2x64xf16>) outs(%init_0, %init_1 : tensor<64xf16>, tensor<64xf16>) {
      ^bb0(%x: f16, %o_0: f16, %o_1: f16):
        %sum = arith.addf %x, %o_0 : f16
        %sq = arith.mulf %x, %x : f16
        linalg.yield %sum, %sq : f16, f16
      } -> (tensor<64xf16>, tensor<64xf16>)
      ktdf.write_to_fifo %r#1, %p#1 : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
      ktdf.write_to_fifo %r#0, %p#0 : tensor<64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>
    } {applicable_units = ["SFU"]}
    ktdf.stage depends_in(%p#2) depends_out(none) {
      ktdf.data_transfer from %p#1 size [64] to %out_1[0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
      ktdf.data_transfer from %p#0 size [64] to %out_0[0] size [64] : !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<64xf16, "DDR">
    } {applicable_units = ["DDR"]}
  }
  return
}
