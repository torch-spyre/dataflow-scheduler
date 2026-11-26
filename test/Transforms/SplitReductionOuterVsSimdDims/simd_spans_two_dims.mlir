// RUN: dataflow-scheduler-opt --split-reduction-outer-vs-simd-dims %s | FileCheck %s

// The SIMD dims can be more than one input dim. On the sample device a SIMD vector is
// 64 f16 elements, so for tensor<2x256x32x2xf16> it is the 32x2 of the two
// innermost dims. Both of their reduction dims (d2, d3) are inner: Generic 1
// reduces only d1, keeping the SIMD dims, and Generic 2 reduces them.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @two_dim_simd(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x256x32x2xf16>) -> tensor<2xf16> {
// CHECK-NEXT:     %[[EMPTY_0:.*]] = tensor.empty() : tensor<2x32x2xf16>
// CHECK-NEXT:     %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "reduction", "parallel", "parallel"]} ins(%[[ARG0]] : tensor<2x256x32x2xf16>) outs(%[[EMPTY_0]] : tensor<2x32x2xf16>) {
// CHECK-NEXT:     ^bb0(%[[VAL_0:.*]]: f16, %[[VAL_1:.*]]: f16):
// CHECK-NEXT:       %[[ADDF_0:.*]] = arith.addf %[[VAL_0]], %[[VAL_1]] : f16
// CHECK-NEXT:       linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:     } -> tensor<2x32x2xf16>
// CHECK-NEXT:     %[[EMPTY_1:.*]] = tensor.empty() : tensor<2xf16>
// CHECK-NEXT:     %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_3]]], iterator_types = ["parallel", "reduction", "reduction"]} ins(%[[GENERIC_0]] : tensor<2x32x2xf16>) outs(%[[EMPTY_1]] : tensor<2xf16>) {
// CHECK-NEXT:     ^bb0(%[[VAL_2:.*]]: f16, %[[VAL_3:.*]]: f16):
// CHECK-NEXT:       %[[ADDF_1:.*]] = arith.addf %[[VAL_2]], %[[VAL_3]] : f16
// CHECK-NEXT:       linalg.yield %[[ADDF_1]] : f16
// CHECK-NEXT:     } -> tensor<2xf16>
// CHECK-NEXT:     return %[[GENERIC_1]] : tensor<2xf16>
// CHECK-NEXT:   }



// The SIMD dims can be more than one input dim. On the sample device a SIMD vector is
// 64 f16 elements, so for tensor<2x256x32x2xf16> it is the 32x2 of the two
// innermost dims. Both of their reduction dims (d2, d3) are inner: Generic 1
// reduces only d1, keeping the SIMD dims, and Generic 2 reduces them.

#in  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#out = affine_map<(d0, d1, d2, d3) -> (d0)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @two_dim_simd(%in: tensor<2x256x32x2xf16>) -> tensor<2xf16> {
    %init = tensor.empty() : tensor<2xf16>
    %res = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction", "reduction", "reduction"]} ins(%in : tensor<2x256x32x2xf16>) outs(%init : tensor<2xf16>) {
    ^bb0(%a: f16, %acc: f16):
      %s = arith.addf %a, %acc : f16
      linalg.yield %s : f16
    } -> tensor<2xf16>
    return %res : tensor<2xf16>
  }
}
