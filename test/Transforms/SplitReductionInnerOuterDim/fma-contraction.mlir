// RUN: dataflow-scheduler-opt --reduction-loop-exposure %s | FileCheck %s
// RUN: dataflow-scheduler-opt --split-reduction-inner-outer-dim %s | FileCheck %s
// RUN: dataflow-scheduler-opt --map-reduction-partials %s | FileCheck %s

// A contraction whose body is a single math.fma is left alone by the
// reduction passes, as one written as a multiply and an add is. Its
// accumulator is the third operand of the fma, which none of them handle.

// CHECK-LABEL: func.func @fma_contraction
// CHECK:         linalg.generic
// CHECK-SAME:      iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NEXT:    ^bb0(%[[X:.*]]: f16, %[[Y:.*]]: f16, %[[ACC:.*]]: f16):
// CHECK-NEXT:      %[[R:.*]] = math.fma %[[X]], %[[Y]], %[[ACC]] : f16
// CHECK-NEXT:      linalg.yield %[[R]] : f16
// CHECK-NOT:     scf.for
#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
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
