// RUN: dataflow-scheduler-opt --fuse-linalg %s | FileCheck %s

// Tests that a reduction whose accumulator is initialised by a splat
// arith.constant keeps that constant as its init: it carries the value the
// accumulator starts from, so it is not replaced with a tensor.empty.

#in = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL:   func.func @max_const_init
// CHECK:           %[[CST:.*]] = arith.constant dense<1.000000e+01> : tensor<32xf32>
// CHECK:           linalg.generic
// CHECK-SAME:        outs(%[[CST]] : tensor<32xf32>)
// CHECK:             arith.maximumf
func.func @max_const_init(%arg0: tensor<256x32xf32>) -> tensor<32xf32> {
  %0 = arith.constant dense<10.0> : tensor<32xf32>
  %1 = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<256x32xf32>) outs(%0 : tensor<32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %2 = arith.maximumf %in, %out : f32
    linalg.yield %2 : f32
  } -> tensor<32xf32>
  return %1 : tensor<32xf32>
}
