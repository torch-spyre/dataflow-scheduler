// RUN: dataflow-scheduler-opt --reduction-dim-chunking="chunk-size-threshold=64" %s -verify-diagnostics

// The threshold path must not split a SIMD lane that spans two input dims. On the
// sample device a SIMD lane is 64 f16 elements, so for tensor<1x4x32x2xf16> it is
// the 32x2 of the two innermost dims, and only the 4-wide d1 can be chunked.
//
// total_input_bytes = 4 * 32 * 2 * 2 = 512; threshold=64 -> N=8. d1 takes 4
// of the 8 chunks; the remaining 2 would have to cut the 32-wide d2, splitting
// every SIMD lane in two, so the count cannot be distributed.

#in  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#out = affine_map<(d0, d1, d2, d3) -> (d0)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @two_dim_simd_lane(%src: memref<2x1x4x32x2xf16, "L1">, %dst: memref<2x1xf16, "L1">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %b = %c0 to %c2 step %c1 {
      // expected-error @+1 {{reduction-dim-chunking: chunk count could not be fully distributed across the chunkable reduction dimensions}}
      ktdf.pipeline {
        %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, !ktdf.token, !ktdf.token) {
          %i = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
          %o = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %i, %o, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#2) {
          ktdf.data_transfer from %src[%b, 0, 0, 0, 0] size [1, 1, 4, 32, 2] to %p#0 size [256] : memref<2x1x4x32x2xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%p#2) depends_out(%p#3) {
          %x = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 256xf16> -> tensor<1x4x32x2xf16>
          %e = tensor.empty() : tensor<1xf16>
          %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction", "reduction", "reduction"]} ins(%x : tensor<1x4x32x2xf16>) outs(%e : tensor<1xf16>) {
          ^bb0(%a: f16, %acc: f16):
            %s = arith.addf %a, %acc : f16
            linalg.yield %s : f16
          } -> tensor<1xf16>
          ktdf.write_to_fifo %r, %p#1 : tensor<1xf16>, <"SFU" -> "L1SU", 1xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%p#3) depends_out(none) {
          ktdf.data_transfer from %p#1 size [1] to %dst[%b, 0] size [1, 1] : !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, memref<2x1xf16, "L1">
        } {applicable_units = ["L1SU"]}
      }
    }
    return
  }
}
