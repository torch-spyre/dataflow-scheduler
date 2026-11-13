// RUN: dataflow-scheduler-opt --reduction-dim-chunking="num-chunks=4" %s -verify-diagnostics

// Chunking reduces each chunk on its own and merges a chunk's result into the
// partial result of the chunks before it by reapplying the reduction body.
// That is only the reduction when the body is a single associative and
// commutative op over the input element and the accumulator.  A sum of squares,
// acc + x * x, would square the partial result too when merging, so it is
// rejected and left untransformed.

#in  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#out = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @sum_of_squares(%src: memref<2x1x256x64xf16, "L1">, %dst: memref<2x1x64xf16, "L1">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %b = %c0 to %c2 step %c1 {
      ktdf.pipeline {
        %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
          %i = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
          %o = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %i, %o, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#2) {
          ktdf.data_transfer from %src[%b, 0, 0, 0] size [1, 1, 256, 64] to %p#0 size [16384] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%p#2) depends_out(%p#3) {
          %x = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 16384xf16> -> tensor<1x256x64xf16>
          %e = tensor.empty() : tensor<1x64xf16>
          // expected-error @below {{reduction-dim-chunking: the reduction body is not a single associative and commutative op over the input element and the accumulator, so its chunks cannot be merged}}
          %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction", "parallel"]} ins(%x : tensor<1x256x64xf16>) outs(%e : tensor<1x64xf16>) {
          ^bb0(%a: f16, %acc: f16):
            %sq = arith.mulf %a, %a : f16
            %s = arith.addf %sq, %acc : f16
            linalg.yield %s : f16
          } -> tensor<1x64xf16>
          ktdf.write_to_fifo %r, %p#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%p#3) depends_out(none) {
          ktdf.data_transfer from %p#1 size [64] to %dst[%b, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
        } {applicable_units = ["L1SU"]}
      }
    }
    return
  }
}
