// RUN: dataflow-scheduler-opt --reduction-dim-chunking="num-chunks=4" %s -verify-diagnostics

// The pass can chunk only a single input, so a reduction with two inputs is
// rejected and left untransformed.

#in  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#out = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @two_inputs(%src: memref<2x1x256x64xf16, "L1">, %src2: memref<2x1x256x64xf16, "L1">, %dst: memref<2x1x64xf16, "L1">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %b = %c0 to %c2 step %c1 {
      ktdf.pipeline {
        %p:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
          %i = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
          %j = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
          %o = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %i, %j, %o, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#3) {
          ktdf.data_transfer from %src[%b, 0, 0, 0] size [1, 1, 256, 64] to %p#0 size [16384] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
          ktdf.data_transfer from %src2[%b, 0, 0, 0] size [1, 1, 256, 64] to %p#1 size [16384] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%p#3) depends_out(%p#4) {
          %x = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 16384xf16> -> tensor<1x256x64xf16>
          %y = ktdf.read_from_fifo %p#1 : <"L1LU" -> "SFU", 16384xf16> -> tensor<1x256x64xf16>
          %e = tensor.empty() : tensor<1x64xf16>
          // expected-error @below {{reduction-dim-chunking: a reduction with 2 inputs is not supported; the pass can chunk only a single input}}
          %r = linalg.generic {indexing_maps = [#in, #in, #out], iterator_types = ["parallel", "reduction", "parallel"]} ins(%x, %y : tensor<1x256x64xf16>, tensor<1x256x64xf16>) outs(%e : tensor<1x64xf16>) {
          ^bb0(%a: f16, %a2: f16, %acc: f16):
            %d = arith.subf %a, %a2 : f16
            %s = arith.addf %d, %acc : f16
            linalg.yield %s : f16
          } -> tensor<1x64xf16>
          ktdf.write_to_fifo %r, %p#2 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%p#4) depends_out(none) {
          ktdf.data_transfer from %p#2 size [64] to %dst[%b, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
        } {applicable_units = ["L1SU"]}
      }
    }
    return
  }
}
