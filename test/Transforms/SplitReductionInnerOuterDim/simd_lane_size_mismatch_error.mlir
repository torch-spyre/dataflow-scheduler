// RUN: dataflow-scheduler-opt --split-reduction-inner-outer-dim %s -verify-diagnostics

// The SIMD-lane dims must hold exactly one SIMD lane. On the sample device a SIMD lane
// is 64 f16 elements; for tensor<2x8x3x32xf16> the trailing dims reach it only
// at 3x32 = 96 elements, which the split cannot reduce as one SIMD lane.

#in  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#out = affine_map<(d0, d1, d2, d3) -> (d0)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @misaligned_simd_lane(%in: tensor<2x8x3x32xf16>) -> tensor<2xf16> {
    %init = tensor.empty() : tensor<2xf16>
    // expected-error @+1 {{split-reduction-inner-outer-dim: the SIMD-lane reduction dims hold 96 elements, not one SIMD lane of 64}}
    %res = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction", "reduction", "reduction"]} ins(%in : tensor<2x8x3x32xf16>) outs(%init : tensor<2xf16>) {
    ^bb0(%a: f16, %acc: f16):
      %s = arith.addf %a, %acc : f16
      linalg.yield %s : f16
    } -> tensor<2xf16>
    return %res : tensor<2xf16>
  }
}
