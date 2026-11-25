// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-bufferize | FileCheck %s
// Preserve the physical stick for all 32-bit integer signedness variants.
// CHECK-LABEL: func.func @i32_splat_view(
// CHECK: memref.reinterpret_cast {{.*}}sizes: [4, 32], strides: [32, 1]
// CHECK: ktdp_lowering.load {{.*}} [1, 32] [1, 0]
// CHECK-SAME: memref<4x32xi32, strided<[32, 1], offset: ?>, "L1">
// CHECK-LABEL: func.func @ui32_splat_view(
// CHECK: memref.reinterpret_cast {{.*}}sizes: [4, 32], strides: [32, 1]
// CHECK: ktdp_lowering.load {{.*}} [1, 32] [1, 0]
// CHECK-SAME: memref<4x32xui32, strided<[32, 1], offset: ?>, "L1">
// CHECK-LABEL: func.func @si32_splat_view(
// CHECK: memref.reinterpret_cast {{.*}}sizes: [4, 32], strides: [32, 1]
// CHECK: ktdp_lowering.load {{.*}} [1, 32] [1, 0]
// CHECK-SAME: memref<4x32xsi32, strided<[32, 1], offset: ?>, "L1">

#memory = affine_set<(r, l) : (r >= 0, 7 - r >= 0, l >= 0, 31 - l >= 0)>
#scalar = affine_set<(r, l) : (r >= 0, 3 - r >= 0, l == 0)>
#identity = affine_map<(r, l) -> (r, l)>

module {
  ktdf_arch.device @sample_device attributes {
    mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<ct_local> = "L1">
  } import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @i32_splat_view() -> tensor<1x32xi32> {
    %zero = arith.constant 0 : index
    %row = arith.constant 4 : index
    %src = ktdp.construct_memory_view %zero,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<ct_local>
    } : memref<8x32xi32>
    %tile = ktdp.construct_access_tile %src[%row, %zero] {
      access_tile_set = #scalar, access_tile_order = #identity
    } : memref<8x32xi32> -> !ktdp.access_tile<4x1xindex>
    %value = ktdp_lowering.load %tile[0, 0] [1, 32] [1, 0]
        : !ktdp.access_tile<4x1xindex> -> tensor<1x32xi32>
    return %value : tensor<1x32xi32>
  }

  func.func @ui32_splat_view() -> tensor<1x32xui32> {
    %zero = arith.constant 0 : index
    %row = arith.constant 4 : index
    %src = ktdp.construct_memory_view %zero,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<ct_local>
    } : memref<8x32xui32>
    %tile = ktdp.construct_access_tile %src[%row, %zero] {
      access_tile_set = #scalar, access_tile_order = #identity
    } : memref<8x32xui32> -> !ktdp.access_tile<4x1xindex>
    %value = ktdp_lowering.load %tile[0, 0] [1, 32] [1, 0]
        : !ktdp.access_tile<4x1xindex> -> tensor<1x32xui32>
    return %value : tensor<1x32xui32>
  }

  func.func @si32_splat_view() -> tensor<1x32xsi32> {
    %zero = arith.constant 0 : index
    %row = arith.constant 4 : index
    %src = ktdp.construct_memory_view %zero,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<ct_local>
    } : memref<8x32xsi32>
    %tile = ktdp.construct_access_tile %src[%row, %zero] {
      access_tile_set = #scalar, access_tile_order = #identity
    } : memref<8x32xsi32> -> !ktdp.access_tile<4x1xindex>
    %value = ktdp_lowering.load %tile[0, 0] [1, 32] [1, 0]
        : !ktdp.access_tile<4x1xindex> -> tensor<1x32xsi32>
    return %value : tensor<1x32xsi32>
  }
}
