// RUN: dataflow-scheduler-opt --ktir-legality-check %s | FileCheck %s

// A math.fma in a linalg.generic body is accepted: the lowering turns it into
// a multiply-accumulate, and a contraction may be written with one.

// CHECK-LABEL: func.func @fma_contraction
// CHECK:         math.fma
#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>
module {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @fma_contraction(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>) -> tensor<64x64xf16> attributes {grid = [1]} {
    %init = arith.constant dense<0.0> : tensor<64x64xf16>
    %c = linalg.generic {indexing_maps = [#mk, #kn, #mn], iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%init : tensor<64x64xf16>) {
    ^bb0(%x: f16, %y: f16, %acc: f16):
      %r = math.fma %x, %y, %acc : f16
      linalg.yield %r : f16
    } -> tensor<64x64xf16>
    return %c : tensor<64x64xf16>
  }
}
