// RUN: dataflow-scheduler-opt --ktir-legality-check %s | FileCheck %s

// A reduction linalg.generic with arith.maximumf combiner whose accumulator is
// initialised by loading from local (ct_local) memory must be accepted without
// diagnostics.

// CHECK-LABEL: func.func @max_onstick_local_init
// CHECK:         linalg.fill
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 31 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 31 >= 0)>
module {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @max_onstick_local_init() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c8589934592 = arith.constant 8589934592 : index
    %c1024 = arith.constant 1024 : index

    // Input tensor: 2x256x32 from global memory.
    %0 = ktdp.construct_memory_view %c0_0, sizes: [2, 256, 32], strides: [8192, 32, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x32xf32>
    // Output tensor: 256x32 in global memory.
    %1 = ktdp.construct_memory_view %c8589934592, sizes: [256, 32], strides: [32, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<256x32xf32>
    // Initial-value tensor: 256x32 scalar read from local (ct_local) memory.
    %2 = ktdp.construct_memory_view %c1024, sizes: [256, 32], strides: [32, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<ct_local>} : memref<256x32xf32, #ktdp.memory_space<ct_local>>

    %3 = ktdp.construct_access_tile %0[%c0, %c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<2x256x32xf32> -> !ktdp.access_tile<2x256x32xindex>
    %inp = ktdp.load %3 : <2x256x32xindex> -> tensor<2x256x32xf32>

    // Load a single element (1x1 tile) from local memory and extract the scalar.
    %4 = ktdp.construct_access_tile %2[%c0, %c0] {access_tile_order = affine_map<(d0, d1) -> (d0, d1)>, access_tile_set = affine_set<(d0, d1) : (d0 >= 0, d0 <= 0, d1 >= 0, d1 <= 0)>} : memref<256x32xf32, #ktdp.memory_space<ct_local>> -> !ktdp.access_tile<1x1xindex>
    %init_tile = ktdp.load %4 : <1x1xindex> -> tensor<1x1xf32>
    %idx = arith.constant 0 : index
    %scalar = tensor.extract %init_tile[%idx, %idx] : tensor<1x1xf32>
    %empty = tensor.empty() : tensor<256x32xf32>
    %init = linalg.fill ins(%scalar : f32) outs(%empty : tensor<256x32xf32>) -> tensor<256x32xf32>

    %5 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["reduction", "parallel", "reduction", "parallel"]} ins(%inp : tensor<2x256x32xf32>) outs(%init : tensor<256x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.maximumf %in, %out : f32
      linalg.yield %7 : f32
    } -> tensor<256x32xf32>

    %6 = ktdp.construct_access_tile %1[%c0, %c0] {access_tile_order = #map3, access_tile_set = #set1} : memref<256x32xf32> -> !ktdp.access_tile<256x32xindex>
    ktdp.store %5, %6 : tensor<256x32xf32>, <256x32xindex>
    return
  }
}
