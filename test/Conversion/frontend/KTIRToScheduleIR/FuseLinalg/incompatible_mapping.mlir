// RUN: dataflow-scheduler-opt --fuse-linalg %s | FileCheck %s

// Tests that operations which would otherwise be fused are kept apart when
// their mappings refer to different resources.

#id = affine_map<(d0) -> (d0)>

// CHECK-LABEL:   func.func @different_kinds
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "SFU"
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "PE"
// CHECK-NOT:       linalg.generic
func.func @different_kinds(%a: tensor<8xf32>, %b: tensor<8xf32>)
    -> tensor<8xf32> {
  %e0 = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<8xf32>, tensor<8xf32>) outs(%e0 : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  %e1 = tensor.empty() : tensor<8xf32>
  %1 = linalg.generic {indexing_maps = [#id, #id],
                       iterator_types = ["parallel"]}
       ins(%0 : tensor<8xf32>) outs(%e1 : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = "PE"} {
  ^bb0(%x: f32, %o: f32):
    %s = math.sqrt %x : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// A specific resource and a resource kind are not the same resource.
// CHECK-LABEL:   func.func @resource_versus_kind
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = @unit0
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "SFU"
// CHECK-NOT:       linalg.generic
func.func @resource_versus_kind(%a: tensor<8xf32>, %b: tensor<8xf32>)
    -> tensor<8xf32> {
  %e0 = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<8xf32>, tensor<8xf32>) outs(%e0 : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = @unit0} {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  %e1 = tensor.empty() : tensor<8xf32>
  %1 = linalg.generic {indexing_maps = [#id, #id],
                       iterator_types = ["parallel"]}
       ins(%0 : tensor<8xf32>) outs(%e1 : tensor<8xf32>)
       attrs = {ktdf_arch.maps_to = "SFU"} {
  ^bb0(%x: f32, %o: f32):
    %s = math.sqrt %x : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %1 : tensor<8xf32>
}

// The conflict is detected on mappings that are only implied by the payloads,
// which means propagation has to happen before fusion is attempted.
// CHECK-LABEL:   func.func @implied_conflict
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "SFU"
// CHECK:           linalg.generic
// CHECK-SAME:        ktdf_arch.maps_to = "PE"
// CHECK-NOT:       linalg.generic
func.func @implied_conflict(%a: tensor<8xf32>, %b: tensor<8xf32>)
    -> tensor<8xf32> {
  %e0 = tensor.empty() : tensor<8xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id, #id],
                       iterator_types = ["parallel"]}
       ins(%a, %b : tensor<8xf32>, tensor<8xf32>) outs(%e0 : tensor<8xf32>) {
  ^bb0(%x: f32, %y: f32, %o: f32):
    %m = arith.mulf %x, %y {ktdf_arch.maps_to = "SFU"} : f32
    linalg.yield %m : f32
  } -> tensor<8xf32>
  %e1 = tensor.empty() : tensor<8xf32>
  %1 = linalg.generic {indexing_maps = [#id, #id],
                       iterator_types = ["parallel"]}
       ins(%0 : tensor<8xf32>) outs(%e1 : tensor<8xf32>) {
  ^bb0(%x: f32, %o: f32):
    %s = math.sqrt %x {ktdf_arch.maps_to = "PE"} : f32
    linalg.yield %s : f32
  } -> tensor<8xf32>
  return %1 : tensor<8xf32>
}
