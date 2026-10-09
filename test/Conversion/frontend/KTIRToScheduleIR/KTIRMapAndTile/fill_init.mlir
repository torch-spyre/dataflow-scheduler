// RUN: dataflow-scheduler-opt --ktir-map-and-tile %s | FileCheck %s

// A reduction whose accumulator is initialised by a linalg.fill keeps that
// fill outside the loops: tiling leaves the accumulator a slice of the loops'
// carried value, and breaking that dependency slices the fill itself instead.

// CHECK-LABEL: func.func @max_fill_init
// CHECK:         %[[CST:.*]] = arith.constant 1.000000e+01 : f32
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<256x32xf32>
// CHECK-NEXT:    %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[EMPTY]] : tensor<256x32xf32>) -> tensor<256x32xf32>
// CHECK:         scf.for %[[IV0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:      scf.for %[[IV1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        %[[LOAD:.*]] = ktdp_lowering.load {{.*}} -> tensor<2x1x32xf32>
// CHECK-NEXT:        %[[SLICE:.*]] = tensor.extract_slice %[[FILL]][%[[IV0]], %[[IV1]]] [1, 32] [1, 1] : tensor<256x32xf32> to tensor<1x32xf32>
// CHECK-NEXT:        %[[RES:.*]] = linalg.generic
// CHECK-SAME:          ins(%[[LOAD]] : tensor<2x1x32xf32>) outs(%[[SLICE]] : tensor<1x32xf32>)
// CHECK:               arith.maximumf
// CHECK:             ktdp_lowering.store %[[RES]]
// CHECK-NOT:     linalg.fill

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 31 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 31 >= 0)>
module {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @max_fill_init() attributes {grid = [1]} {
    %c8589934592 = arith.constant 8589934592 : index
    %cst = arith.constant 1.000000e+01 : f32
    %c0 = arith.constant 0 : index
    %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 32], strides: [8192, 32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x32xf32>
    %1 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<2x256x32xf32> -> !ktdp.access_tile<2x256x32xindex>
    %2 = tensor.empty() : tensor<256x32xf32>
    %3 = ktdp.load %1 : <2x256x32xindex> -> tensor<2x256x32xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%2 : tensor<256x32xf32>) -> tensor<256x32xf32>
    %5 = ktdp.construct_memory_view %c8589934592, sizes: [256, 32], strides: [32, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<256x32xf32>
    %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%3 : tensor<2x256x32xf32>) outs(%4 : tensor<256x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.maximumf %in, %out : f32
      linalg.yield %8 : f32
    } -> tensor<256x32xf32>
    %7 = ktdp.construct_access_tile %5[%c0, %c0] {access_tile_order = #map3, access_tile_set = #set1} : memref<256x32xf32> -> !ktdp.access_tile<256x32xindex>
    ktdp.store %6, %7 : tensor<256x32xf32>, <256x32xindex>
    return
  }
}
