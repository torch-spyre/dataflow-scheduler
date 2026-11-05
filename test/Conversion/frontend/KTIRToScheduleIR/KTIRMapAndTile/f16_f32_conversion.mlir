// RUN: dataflow-scheduler-opt --ktir-map-and-tile --split-input-file %s | FileCheck %s

// A conversion between f16 and f32 is tiled by the lanes f16 fills, 64, in
// either direction: one conversion reads, or writes, a whole f16 vector, which
// is two of f32. Tiled by the f32 lanes, 32, a tile would be half of that.

// CHECK-LABEL: func.func @dl16tofp32()
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %[[C64]] {
// CHECK:           ktdp_lowering.load {{.+}} [1, 64] [1, 1] {dataflow_scheduler.throttle = 64 : i64} : !ktdp.access_tile<4x128xindex> -> tensor<1x64xf16>
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%{{.+}} : tensor<1x64xf16>) outs(%{{.+}} : tensor<1x64xf32>)
// CHECK-SAME:        dataflow_scheduler.throttle = 64 : i64
// CHECK:             arith.extf
// CHECK:           ktdp_lowering.store {{.+}} [1, 64] [1, 1] {dataflow_scheduler.throttle = 64 : i64} : tensor<1x64xf32> into !ktdp.access_tile<4x128xindex>

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @dl16tofp32() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c18432 = arith.constant 18432 : index
    %0 = ktdp.construct_memory_view %c1024, sizes: [4, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x128xf16>
    %1 = ktdp.construct_access_tile %0[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<4x128xf16> -> !ktdp.access_tile<4x128xindex>
    %2 = ktdp.construct_memory_view %c18432, sizes: [4, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x128xf32>
    %3 = ktdp.construct_access_tile %2[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<4x128xf32> -> !ktdp.access_tile<4x128xindex>
    %4 = ktdp.load %1 : <4x128xindex> -> tensor<4x128xf16>
    %5 = tensor.empty() : tensor<4x128xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<4x128xf16>) outs(%5 : tensor<4x128xf32>) {
    ^bb0(%in: f16, %out: f32):
      %7 = arith.extf %in : f16 to f32
      linalg.yield %7 : f32
    } -> tensor<4x128xf32>
    ktdp.store %6, %3 : tensor<4x128xf32>, <4x128xindex>
    return
  }
}

// -----

// CHECK-LABEL: func.func @fp32todl16()
// CHECK-DAG:     %[[C64:.+]] = arith.constant 64 : index
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %[[C64]] {
// CHECK:           ktdp_lowering.load {{.+}} [1, 64] [1, 1] {dataflow_scheduler.throttle = 64 : i64} : !ktdp.access_tile<4x128xindex> -> tensor<1x64xf32>
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%{{.+}} : tensor<1x64xf32>) outs(%{{.+}} : tensor<1x64xf16>)
// CHECK-SAME:        dataflow_scheduler.throttle = 64 : i64
// CHECK:             arith.truncf
// CHECK:           ktdp_lowering.store {{.+}} [1, 64] [1, 1] {dataflow_scheduler.throttle = 64 : i64} : tensor<1x64xf16> into !ktdp.access_tile<4x128xindex>

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @fp32todl16() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c18432 = arith.constant 18432 : index
    %0 = ktdp.construct_memory_view %c1024, sizes: [4, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x128xf32>
    %1 = ktdp.construct_access_tile %0[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<4x128xf32> -> !ktdp.access_tile<4x128xindex>
    %2 = ktdp.construct_memory_view %c18432, sizes: [4, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x128xf16>
    %3 = ktdp.construct_access_tile %2[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<4x128xf16> -> !ktdp.access_tile<4x128xindex>
    %4 = ktdp.load %1 : <4x128xindex> -> tensor<4x128xf32>
    %5 = tensor.empty() : tensor<4x128xf16>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<4x128xf32>) outs(%5 : tensor<4x128xf16>) {
    ^bb0(%in: f32, %out: f16):
      %7 = arith.truncf %in : f32 to f16
      linalg.yield %7 : f16
    } -> tensor<4x128xf16>
    ktdp.store %6, %3 : tensor<4x128xf16>, <4x128xindex>
    return
  }
}

// -----

// Without a conversion, an f32 compute keeps the lanes f32 fills.

// CHECK-LABEL: func.func @fp32_without_conversion()
// CHECK-DAG:     %[[C32:.+]] = arith.constant 32 : index
// CHECK:         scf.for %{{.+}} = %{{.+}} to %{{.+}} step %[[C32]] {
// CHECK:           linalg.generic
// CHECK-SAME:        ins(%{{.+}} : tensor<1x32xf32>) outs(%{{.+}} : tensor<1x32xf32>)
// CHECK-SAME:        dataflow_scheduler.throttle = 32 : i64

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @fp32_without_conversion() attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c18432 = arith.constant 18432 : index
    %0 = ktdp.construct_memory_view %c1024, sizes: [4, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x128xf32>
    %1 = ktdp.construct_access_tile %0[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<4x128xf32> -> !ktdp.access_tile<4x128xindex>
    %2 = ktdp.construct_memory_view %c18432, sizes: [4, 128], strides: [128, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x128xf32>
    %3 = ktdp.construct_access_tile %2[%c0, %c0] {access_tile_order = #map, access_tile_set = #set} : memref<4x128xf32> -> !ktdp.access_tile<4x128xindex>
    %4 = ktdp.load %1 : <4x128xindex> -> tensor<4x128xf32>
    %5 = tensor.empty() : tensor<4x128xf32>
    %6 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<4x128xf32>) outs(%5 : tensor<4x128xf32>) {
    ^bb0(%in: f32, %out: f32):
      %7 = arith.mulf %in, %in : f32
      linalg.yield %7 : f32
    } -> tensor<4x128xf32>
    ktdp.store %6, %3 : tensor<4x128xf32>, <4x128xindex>
    return
  }
}
