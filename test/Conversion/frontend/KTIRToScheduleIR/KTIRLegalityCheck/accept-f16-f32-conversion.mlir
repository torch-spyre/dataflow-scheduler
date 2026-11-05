// RUN: dataflow-scheduler-opt --ktir-legality-check %s | FileCheck %s
#map = affine_map<(d0) -> (d0)>
// The device converts between f16 and f32 in both directions, with dl16tofp32
// and fp32todl16, so both conversions are legal in a generic's body.
// CHECK-LABEL: func.func @dl16tofp32
func.func @dl16tofp32(%a: tensor<64xf16>, %o: tensor<64xf32>) -> tensor<64xf32> {
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%a : tensor<64xf16>) outs(%o : tensor<64xf32>) {
  ^bb0(%in: f16, %out: f32):
    %e = arith.extf %in : f16 to f32
    linalg.yield %e : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// CHECK-LABEL: func.func @fp32todl16
func.func @fp32todl16(%a: tensor<64xf32>, %o: tensor<64xf16>) -> tensor<64xf16> {
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%a : tensor<64xf32>) outs(%o : tensor<64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// Both ends of the f32 connection can be standalone conversions. Check the
// connection without canonicalization, which could remove the round trip.
// CHECK-LABEL: func.func @dl16tofp32_to_dl16
// CHECK:         %[[WIDE:.*]] = linalg.generic
// CHECK:           arith.extf
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[WIDE]] : tensor<64xf32>)
// CHECK:           arith.truncf
func.func @dl16tofp32_to_dl16(%a: tensor<64xf16>, %f: tensor<64xf32>, %h: tensor<64xf16>) -> tensor<64xf16> {
  %wide = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%a : tensor<64xf16>) outs(%f : tensor<64xf32>) {
  ^bb0(%in: f16, %out: f32):
    %e = arith.extf %in : f16 to f32
    linalg.yield %e : f32
  } -> tensor<64xf32>
  %narrow = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%wide : tensor<64xf32>) outs(%h : tensor<64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %narrow : tensor<64xf16>
}
