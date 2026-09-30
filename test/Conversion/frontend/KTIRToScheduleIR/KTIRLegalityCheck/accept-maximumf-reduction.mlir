// RUN: dataflow-scheduler-opt --ktir-legality-check %s | FileCheck %s

// A reduction linalg.generic with arith.maximumf combiner and a user-supplied
// linalg.fill initializer must be accepted without diagnostics.

// CHECK-LABEL: func.func @max_onstick_1core
// CHECK:         linalg.fill
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 31 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 31 >= 0)>
module {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @max_onstick_1core() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c8589934592 = arith.constant 8589934592 : index
    %0 = ktdp.construct_memory_view %c0_0, sizes: [2, 256, 32], strides: [8192, 32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x32xf32>
    %1 = ktdp.construct_memory_view %c8589934592, sizes: [256, 32], strides: [32, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<256x32xf32>
    %2 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<2x256x32xf32> -> !ktdp.access_tile<2x256x32xindex>
    %3 = ktdp.load %2 : <2x256x32xindex> -> tensor<2x256x32xf32>
    %cst = arith.constant 10.0 : f32
    %4 = tensor.empty() : tensor<256x32xf32>
    %init = linalg.fill ins(%cst : f32) outs(%4 : tensor<256x32xf32>) -> tensor<256x32xf32>
    %5 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%3 : tensor<2x256x32xf32>) outs(%init : tensor<256x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.maximumf %in, %out : f32
      linalg.yield %7 : f32
    } -> tensor<256x32xf32>
    %6 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map3, access_tile_set = #set1} : memref<256x32xf32> -> !ktdp.access_tile<256x32xindex>
    ktdp.store %5, %6 : tensor<256x32xf32>, <256x32xindex>
    return
  }
}
