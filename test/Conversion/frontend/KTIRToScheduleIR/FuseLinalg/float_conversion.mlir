// RUN: dataflow-scheduler-opt --fuse-linalg %s | FileCheck %s

// A conversion between f16 and f32 stays a linalg.generic of its own: the
// device converts a whole vector of f16 into two of f32, or two back into one,
// which is lowered with the FIFO on its wide side and not as one lane-wise op
// of a fused body.

#id = affine_map<(d0) -> (d0)>

// CHECK-LABEL:   func.func @extf_producer
// CHECK:           linalg.generic
// CHECK:             arith.extf
// CHECK-NOT:         arith.addf
// CHECK:           linalg.generic
// CHECK:             arith.addf
func.func @extf_producer(%a: tensor<64xf16>) -> tensor<64xf32> {
  %e0 = tensor.empty() : tensor<64xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
       ins(%a : tensor<64xf16>) outs(%e0 : tensor<64xf32>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f16, %o: f32):
    %c = arith.extf %x : f16 to f32
    linalg.yield %c : f32
  } -> tensor<64xf32>
  %e1 = tensor.empty() : tensor<64xf32>
  %1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
       ins(%0 : tensor<64xf32>) outs(%e1 : tensor<64xf32>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %o: f32):
    %s = arith.addf %x, %x : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  return %1 : tensor<64xf32>
}

// CHECK-LABEL:   func.func @truncf_consumer
// CHECK:           linalg.generic
// CHECK:             arith.addf
// CHECK-NOT:         arith.truncf
// CHECK:           linalg.generic
// CHECK:             arith.truncf
func.func @truncf_consumer(%a: tensor<64xf32>) -> tensor<64xf16> {
  %e0 = tensor.empty() : tensor<64xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
       ins(%a : tensor<64xf32>) outs(%e0 : tensor<64xf32>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %o: f32):
    %s = arith.addf %x, %x : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  %e1 = tensor.empty() : tensor<64xf16>
  %1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
       ins(%0 : tensor<64xf32>) outs(%e1 : tensor<64xf16>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %o: f16):
    %c = arith.truncf %x : f32 to f16
    linalg.yield %c : f16
  } -> tensor<64xf16>
  return %1 : tensor<64xf16>
}
