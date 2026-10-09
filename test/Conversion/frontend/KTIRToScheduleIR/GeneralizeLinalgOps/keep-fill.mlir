// RUN: dataflow-scheduler-opt --generalize-linalg-ops %s | FileCheck %s

// A named compute op becomes a linalg.generic, but the linalg.fill that
// initialises a reduction's accumulator stays named, so the user-supplied
// initial value is still recognisable as an initializer.

// CHECK-LABEL: func.func @keep_fill
// CHECK:         %[[CST:.*]] = arith.constant 1.000000e+01 : f32
// CHECK:         %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%{{.*}} : tensor<32xf32>) -> tensor<32xf32>
// CHECK:         linalg.generic
// CHECK-SAME:      outs(%[[FILL]] : tensor<32xf32>)
// CHECK:           arith.maximumf
// CHECK-NOT:     linalg.reduce
func.func @keep_fill(%arg0: tensor<256x32xf32>) -> tensor<32xf32> {
  %cst = arith.constant 10.0 : f32
  %0 = tensor.empty() : tensor<32xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  %2 = linalg.reduce ins(%arg0 : tensor<256x32xf32>) outs(%1 : tensor<32xf32>) dimensions = [0]
    (%in: f32, %out: f32) {
      %3 = arith.maximumf %in, %out : f32
      linalg.yield %3 : f32
    }
  return %2 : tensor<32xf32>
}

// CHECK-LABEL: func.func @generalize_named
// CHECK-NOT:     linalg.add
// CHECK:         linalg.generic
// CHECK:           arith.addf
func.func @generalize_named(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) -> tensor<32xf32> {
  %0 = tensor.empty() : tensor<32xf32>
  %1 = linalg.add ins(%arg0, %arg1 : tensor<32xf32>, tensor<32xf32>) outs(%0 : tensor<32xf32>) -> tensor<32xf32>
  return %1 : tensor<32xf32>
}
