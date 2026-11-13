// RUN: dataflow-scheduler-opt --reduction-dim-chunking="num-chunks=2" %s -verify-diagnostics

// Chunking a reduction dim slices the input along the input dim it indexes, so
// each input dim has to be indexed by a single loop dim: the input map must be
// a projected permutation.  Here the first input dim is indexed by d0 + d1, as
// in a convolution, where an output position and a reduction (window) position
// combine in one subscript.  Chunking d1 would need a sliding-window slice
// whose chunks overlap, which the pass does not do, so it is rejected and the
// pipeline is left untransformed.  (d1 also indexes the second input dim on
// its own, which gives it a static extent.)

#in  = affine_map<(d0, d1, d2) -> (d0 + d1, d1, d2)>
#out = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @windowed_sum(%src: memref<7x4x64xf16, "L1">, %dst: memref<4x64xf16, "L1">) {
    // expected-error @below {{reduction-dim-chunking: the first input of the reduction is not indexed by a projected permutation, so its chunk cannot be sliced}}
    ktdf.pipeline {
      %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 1792xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, !ktdf.token, !ktdf.token) {
        %i = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 1792xf16>
        %o = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %i, %o, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 1792xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#2) {
        ktdf.data_transfer from %src[0, 0, 0] size [7, 4, 64] to %p#0 size [1792] : memref<7x4x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 1792xf16>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%p#2) depends_out(%p#3) {
        %x = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 1792xf16> -> tensor<7x4x64xf16>
        %e = tensor.empty() : tensor<4x64xf16>
        %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction", "parallel"]} ins(%x : tensor<7x4x64xf16>) outs(%e : tensor<4x64xf16>) {
        ^bb0(%a: f16, %acc: f16):
          %s = arith.addf %a, %acc : f16
          linalg.yield %s : f16
        } -> tensor<4x64xf16>
        ktdf.write_to_fifo %r, %p#1 : tensor<4x64xf16>, <"SFU" -> "L1SU", 256xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#3) depends_out(none) {
        ktdf.data_transfer from %p#1 size [256] to %dst[0, 0] size [4, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, memref<4x64xf16, "L1">
      } {applicable_units = ["L1SU"]}
    }
    return
  }
}
