// RUN: dataflow-scheduler-opt --fuse-linalg %s | FileCheck %s

// Tests that a reduction whose accumulator is initialised by a linalg.fill
// keeps that fill as its init: the fill carries the value the accumulator
// starts from, so it is not replaced with a tensor.empty.

#in = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL:   func.func @max_fill_init
// CHECK:           %[[CST:.*]] = arith.constant 1.000000e+01 : f32
// CHECK:           %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32)
// CHECK:           linalg.generic
// CHECK-SAME:        outs(%[[FILL]] : tensor<32xf32>)
// CHECK:             arith.maximumf
func.func @max_fill_init(%arg0: tensor<256x32xf32>) -> tensor<32xf32> {
  %cst = arith.constant 10.0 : f32
  %0 = tensor.empty() : tensor<32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  %2 = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["reduction", "parallel"]} ins(%arg0 : tensor<256x32xf32>) outs(%1 : tensor<32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %3 = arith.maximumf %in, %out : f32
    linalg.yield %3 : f32
  } -> tensor<32xf32>
  return %2 : tensor<32xf32>
}
