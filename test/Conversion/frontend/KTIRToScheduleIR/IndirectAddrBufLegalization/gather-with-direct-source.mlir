// RUN: dataflow-scheduler-opt --mlir-print-local-scope --indirect-addr-buf-legalization %s | FileCheck %s

// Gather case with a direct source input:
// linalg.generic computes `addf %indirect_data, %direct_data`.
// When IndirectAddrBufLegalization materializes window and entry loops,
// both the destination store access tile AND the direct source load access tile
// must be pinned with the loop IVs, ensuring their ranks match their base memrefs.

// CHECK-LABEL: func.func @local_schedule_1
// CHECK:       scf.for [[I1:%arg[0-9]+]] = {{.*}} to %c2 step
// CHECK:         scf.for [[I2:%arg[0-9]+]] = %c0{{.*}} to %c32 step %c1{{.*}} {
// CHECK:           [[INDIRECT_LOAD:%.+]] = ktdp.load {{.*}} : <2x64xindex> -> tensor<2x64xf16>
// CHECK:           [[DIRECT_TILE:%.+]] = ktdp.construct_access_tile {{.*}}{{\[}}[[I1]], [[I2]], {{.*}}{{\]}}
// CHECK-SAME:          access_tile_order = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-SAME:          access_tile_set = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK-SAME:          -> !ktdp.access_tile<1x1x2x64xindex>
// CHECK:           [[DIRECT_LOAD:%.+]] = ktdp.load [[DIRECT_TILE]] : <1x1x2x64xindex> -> tensor<1x1x2x64xf16>
// CHECK:           [[DIRECT_COLLAPSE:%.+]] = tensor.collapse_shape [[DIRECT_LOAD]] {{\[\[}}0, 1, 2{{\]}}, [3{{\]\]}}
// CHECK-SAME:          tensor<1x1x2x64xf16> into tensor<2x64xf16>
// CHECK:           linalg.generic
// CHECK-SAME:          ins([[INDIRECT_LOAD]], [[DIRECT_COLLAPSE]] : tensor<2x64xf16>, tensor<2x64xf16>)
// CHECK:           tensor.expand_shape {{.*}} tensor<2x64xf16> into tensor<1x1x2x64xf16>
// CHECK:           ktdp.construct_access_tile {{.*}}{{\[}}[[I1]], [[I2]], {{.*}}{{\]}}
// CHECK-SAME:          -> !ktdp.access_tile<1x1x2x64xindex>
// CHECK:           ktdp.store

#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set4 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

module @local_schedule_1 {
  ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @local_schedule_1() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c1000 = arith.constant 1000 : index
    %c2000 = arith.constant 2000 : index
    %c3000 = arith.constant 3000 : index
    %addr_buf = ktdp.construct_memory_view %c2000,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>

    %iab_mv = ktdp_lowering.construct_memory_view %c0,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set2, memory_space = "IAB"}
        : memref<2x32xindex, "IAB">

    %addr_tile = ktdp.construct_access_tile %addr_buf[%c0, %c0]
        {access_tile_order = #map1, access_tile_set = #set2}
        : memref<2x32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<2x32xindex>
    %addr_vals = ktdp.load %addr_tile : !ktdp.access_tile<2x32xindex> -> tensor<2x32xindex>
    %iab_tile = ktdp.construct_access_tile %iab_mv[%c0, %c0]
        {access_tile_order = #map1, access_tile_set = #set2}
        : memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32xindex>
    ktdp.store %addr_vals, %iab_tile : tensor<2x32xindex>, !ktdp.access_tile<2x32xindex>

    %desc_1 = ktdp.construct_memory_view %c0,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>
    %tmp1 = ktdp_lowering.construct_indirect_access_tile
        intermediate_variables(%arg5, %arg6, %arg7, %arg8)
        base_ptr = %iab_mv[%arg5, %arg6]
        %desc_1[(%c0), (%c0 + %arg7), (%arg8)]
        {variables_space_order = #map, variables_space_set = #set4}
        : memref<64x2x64xf16>, memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32x2x64xindex>
    %tmp1_0 = ktdp.load %tmp1 : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

    %desc_direct = ktdp.construct_memory_view %c3000,
        sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
        {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>}
        : memref<2x32x2x64xf16>
    %direct_tile = ktdp.construct_access_tile %desc_direct[%c0, %c0, %c0, %c0]
        {access_tile_order = #map, access_tile_set = #set4}
        : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
    %direct_val = ktdp.load %direct_tile : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

    %empty = tensor.empty() : tensor<2x32x2x64xf16>
    %computed = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%tmp1_0, %direct_val : tensor<2x32x2x64xf16>, tensor<2x32x2x64xf16>)
        outs(%empty : tensor<2x32x2x64xf16>) {
    ^bb0(%in_a: f16, %in_b: f16, %out: f16):
      %sum = arith.addf %in_a, %in_b : f16
      linalg.yield %sum : f16
    } -> tensor<2x32x2x64xf16>

    %desc_2 = ktdp.construct_memory_view %c1000,
        sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
        {coordinate_set = #set4, memory_space = #ktdp.memory_space<global>}
        : memref<2x32x2x64xf16>
    %tmp2 = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0]
        {access_tile_order = #map, access_tile_set = #set4}
        : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
    ktdp.store %computed, %tmp2 : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
    return
  }
}
