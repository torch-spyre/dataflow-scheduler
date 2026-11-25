// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-bufferize | FileCheck %s
// These cases test Bufferize only: splat legalization still assumes that the
// innermost dimension is the physically contiguous load direction.

// CHECK-LABEL: func.func @fp32_contiguous_first_dimension(
// CHECK: memref.reinterpret_cast {{.*}}offset: [{{.*}}], sizes: [32, 1], strides: [1, 32]
// CHECK: ktdp_lowering.load {{.*}} [32, 1] [0, 1]
// CHECK-SAME: memref<32x1xf32, strided<[1, 32], offset: ?>, "L1">
// CHECK-LABEL: func.func @fp32_contiguous_middle_dimension(
// CHECK: memref.reinterpret_cast {{.*}}offset: [{{.*}}], sizes: [1, 32, 1], strides: [256, 1, 32]
// CHECK: ktdp_lowering.load {{.*}} [1, 32, 1] [1, 0, 1]
// CHECK-SAME: memref<1x32x1xf32, strided<[256, 1, 32], offset: ?>, "L1">
// CHECK-LABEL: func.func @fp32_nonzero_contiguous_start(
// CHECK: memref.reinterpret_cast {{.*}}sizes: [1, 1], strides: [1, 32]
// CHECK: ktdp_lowering.load {{.*}} [32, 1] [0, 1]
// CHECK-SAME: memref<1x1xf32, strided<[1, 32], offset: ?>, "L1">
// CHECK-LABEL: func.func @fp32_noncontiguous_row_dimension(
// CHECK: memref.reinterpret_cast {{.*}}sizes: [1, 32], strides: [32, 1]
// CHECK: ktdp_lowering.load {{.*}} [4, 32] [0, 1]
// CHECK-SAME: memref<1x32xf32, strided<[32, 1], offset: ?>, "L1">

#memory = affine_set<(l, r) : (l >= 0, 31 - l >= 0, r >= 0, 7 - r >= 0)>
#middle_memory = affine_set<(g, l, r) : (g >= 0, 1 - g >= 0, l >= 0, 31 - l >= 0, r >= 0, 7 - r >= 0)>
#row_memory = affine_set<(r, l) : (r >= 0, 7 - r >= 0, l >= 0, 31 - l >= 0)>
#point = affine_set<(l, r) : (l == 0, r == 0)>
#middle_point = affine_set<(g, l, r) : (g == 0, l == 0, r == 0)>
#row = affine_set<(r, l) : (r == 0, l >= 0, 31 - l >= 0)>
#identity = affine_map<(l, r) -> (l, r)>
#middle_identity = affine_map<(g, l, r) -> (g, l, r)>

module {
  ktdf_arch.device @sample_device attributes {
    mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<ct_local> = "L1">
  } import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // Lane zero of physical row 4 remains readable as a full 32-lane stick.
  func.func @fp32_contiguous_first_dimension() -> tensor<32x1xf32> {
    %zero = arith.constant 0 : index
    %row = arith.constant 4 : index
    %src = ktdp.construct_memory_view %zero,
        sizes: [32, 8], strides: [1, 32] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<ct_local>
    } : memref<32x8xf32>
    %tile = ktdp.construct_access_tile %src[%zero, %row] {
      access_tile_set = #point, access_tile_order = #identity
    } : memref<32x8xf32> -> !ktdp.access_tile<1x1xindex>
    %value = ktdp_lowering.load %tile[0, 0] [32, 1] [0, 1]
        : !ktdp.access_tile<1x1xindex> -> tensor<32x1xf32>
    return %value : tensor<32x1xf32>
  }

  // The contiguous lane dimension may also be between two other dimensions.
  func.func @fp32_contiguous_middle_dimension() -> tensor<1x32x1xf32> {
    %zero = arith.constant 0 : index
    %group = arith.constant 1 : index
    %row = arith.constant 4 : index
    %src = ktdp.construct_memory_view %zero,
        sizes: [2, 32, 8], strides: [256, 1, 32] {
      coordinate_set = #middle_memory, memory_space = #ktdp.memory_space<ct_local>
    } : memref<2x32x8xf32>
    %tile = ktdp.construct_access_tile %src[%group, %zero, %row] {
      access_tile_set = #middle_point, access_tile_order = #middle_identity
    } : memref<2x32x8xf32> -> !ktdp.access_tile<1x1x1xindex>
    %value = ktdp_lowering.load %tile[0, 0, 0] [1, 32, 1] [1, 0, 1]
        : !ktdp.access_tile<1x1x1xindex> -> tensor<1x32x1xf32>
    return %value : tensor<1x32x1xf32>
  }

  // A nonzero lane start cannot expose the original dimension's full size.
  func.func @fp32_nonzero_contiguous_start() -> tensor<32x1xf32> {
    %zero = arith.constant 0 : index
    %lane = arith.constant 1 : index
    %row = arith.constant 4 : index
    %src = ktdp.construct_memory_view %zero,
        sizes: [32, 8], strides: [1, 32] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<ct_local>
    } : memref<32x8xf32>
    %tile = ktdp.construct_access_tile %src[%lane, %row] {
      access_tile_set = #point, access_tile_order = #identity
    } : memref<32x8xf32> -> !ktdp.access_tile<1x1xindex>
    %value = ktdp_lowering.load %tile[0, 0] [32, 1] [0, 1]
        : !ktdp.access_tile<1x1xindex> -> tensor<32x1xf32>
    return %value : tensor<32x1xf32>
  }

  // Row broadcast already reads a complete stick; its stride-32 axis stays 1.
  func.func @fp32_noncontiguous_row_dimension() -> tensor<4x32xf32> {
    %zero = arith.constant 0 : index
    %row = arith.constant 4 : index
    %src = ktdp.construct_memory_view %zero,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #row_memory, memory_space = #ktdp.memory_space<ct_local>
    } : memref<8x32xf32>
    %tile = ktdp.construct_access_tile %src[%row, %zero] {
      access_tile_set = #row, access_tile_order = #identity
    } : memref<8x32xf32> -> !ktdp.access_tile<1x32xindex>
    %value = ktdp_lowering.load %tile[0, 0] [4, 32] [0, 1]
        : !ktdp.access_tile<1x32xindex> -> tensor<4x32xf32>
    return %value : tensor<4x32xf32>
  }
}
