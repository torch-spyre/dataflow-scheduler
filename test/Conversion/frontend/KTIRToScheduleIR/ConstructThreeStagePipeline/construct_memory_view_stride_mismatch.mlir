// RUN: dataflow-scheduler-opt --construct-three-stage-pipeline %s | FileCheck %s
//
// When a ktdp.construct_memory_view declares non-row-major strides (e.g., column-major
// [1, 96] for a 96x64 shape) but the result memref type carries no strided
// layout (i.e., it implies row-major [64, 1]), the pass previously read only
// the type and emitted wrong offsets/strides.
//
// The pass tracks back to the defining construct_memory_view op (both ktdp and
// ktdp_lowering) to get the authoritative strides, mirroring what getMemorySpace
// already does for memory space. This is a temporary workaround; the real fix is to
// add a consistency check in the ktdp op verifier.

// CHECK-LABEL:   func.func @column_major(
// CHECK-SAME:      %[[ROW:.*]]: index) {
// CHECK:           %[[VIEW:.*]] = ktdp.construct_memory_view {{.*}}, sizes: [96, 64], strides: [1, 96] {{.*}} : memref<96x64xf16>
// CHECK:           %[[MSC:.*]] = memref.memory_space_cast %[[VIEW]] : memref<96x64xf16> to memref<96x64xf16, "DDR">
// CHECK:           %[[RC:.*]] = memref.reinterpret_cast %[[MSC]] to offset: [%[[ROW]]], sizes: [1, 64], strides: [1, 96]
// CHECK-SAME:      : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[1, 96], offset: ?>, "DDR">

// CHECK-LABEL:   func.func @strided_descriptor(
// CHECK-SAME:      %[[ROW:.*]]: index) {
// CHECK:           %[[VIEW:.*]] = ktdp.construct_memory_view {{.*}}, sizes: [64, 2, 64], strides: [64, 4096, 1] {{.*}} : memref<64x2x64xf16>
// CHECK:           %[[MSC:.*]] = memref.memory_space_cast %[[VIEW]] : memref<64x2x64xf16> to memref<64x2x64xf16, "DDR">
// CHECK:           %[[RC:.*]] = memref.reinterpret_cast %[[MSC]] to offset: [{{.*}}], sizes: [1, 1, 64], strides: [64, 4096, 1]
// CHECK-SAME:      : memref<64x2x64xf16, "DDR"> to memref<1x1x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">

// Also test ktdp_lowering.construct_memory_view for IAB-style views
// ktdp_lowering verifier requires the type to carry the strided layout
// when strides differ from row-major, so the type includes it.
// CHECK-LABEL:   func.func @iab_column_major(
// CHECK-SAME:      %[[ROW:.*]]: index) {
// CHECK:           %[[VIEW:.*]] = ktdp_lowering.construct_memory_view {{.*}}, sizes: [32, 64], strides: [1, 32] {{.*}} : memref<32x64xindex, strided<[1, 32]>, "IAB">
// CHECK:           %[[RC:.*]] = memref.reinterpret_cast %[[VIEW]] to offset: [%[[ROW]]], sizes: [1, 64], strides: [1, 32]
// CHECK-SAME:      : memref<32x64xindex, strided<[1, 32]>, "IAB"> to memref<1x64xindex, strided<[1, 32], offset: ?>, "IAB">

// Without the fix, the column-major test would emit:
//   strides: [64, 1]  (row-major from type shape)
//   offset: row * 64  (wrong! should be row * 1)
//
// And the strided descriptor would emit:
//   strides: [128, 64, 1]  (row-major from shape)
//   offset: row * 128 + ...  (wrong! should use [64, 4096, 1])

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // Column-major view: stride 1 along dim 0, 96 along dim 1.
  // The memref type is bare memref<96x64xf16> (no strided layout),
  // which MLIR treats as row-major [64, 1].
  // The op declares strides: [1, 96] which is column-major.
  // With the fix, the reinterpret_cast should use strides [1, 96]
  // and offset = row * 1.
  func.func @column_major(%row: index) {
    %c0 = arith.constant 0 : index
    %addr = arith.constant 1024 : index
    %view = ktdp.construct_memory_view %addr, sizes: [96, 64], strides: [1, 96] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>
    %tile = ktdp.construct_access_tile %view[%row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>
    %data = ktdp.load %tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
    %init = tensor.empty() : tensor<1x64xf16>
    %res = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                           iterator_types = ["parallel", "parallel"]}
        ins(%data : tensor<1x64xf16>) outs(%init : tensor<1x64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %c = arith.constant 1.0 : f16
        %s = arith.addf %in, %c : f16
        linalg.yield %s : f16
    } -> tensor<1x64xf16>
    ktdp.store %res, %tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    return
  }

  // Strided (non-transposed) descriptor: sizes [64, 2, 64], strides [64, 4096, 1]
  // Row-major for this shape would be [128, 64, 1].
  // The memref type is bare memref<64x2x64xf16> (no strided layout).
  // With the fix, the reinterpret_cast should use strides [64, 4096, 1].
  func.func @strided_descriptor(%row: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %addr = arith.constant 2048 : index
    %view = ktdp.construct_memory_view %addr, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<64x2x64xf16>
    %tile = ktdp.construct_access_tile %view[%row, %c0, %c1] {
        access_tile_set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 0 >= 0, d2 >= 0, -d2 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    } : memref<64x2x64xf16> -> !ktdp.access_tile<1x1x64xindex>
    %data = ktdp.load %tile : !ktdp.access_tile<1x1x64xindex> -> tensor<1x1x64xf16>
    %init = tensor.empty() : tensor<1x1x64xf16>
    %res = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
                           iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%data : tensor<1x1x64xf16>) outs(%init : tensor<1x1x64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %c = arith.constant 1.0 : f16
        %s = arith.addf %in, %c : f16
        linalg.yield %s : f16
    } -> tensor<1x1x64xf16>
    ktdp.store %res, %tile : tensor<1x1x64xf16>, !ktdp.access_tile<1x1x64xindex>
    return
  }

  // Test ktdp_lowering.construct_memory_view (for IAB-style views)
  // Column-major: sizes [32, 64], strides [1, 32]
  // ktdp_lowering verifier requires the type to carry the strided layout
  // when strides differ from row-major, so we include it in the type.
  func.func @iab_column_major(%row: index) {
    %c0 = arith.constant 0 : index
    %addr = arith.constant 4096 : index
    %view = ktdp_lowering.construct_memory_view %addr, sizes: [32, 64], strides: [1, 32] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 31 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = "IAB"
    } : memref<32x64xindex, strided<[1, 32]>, "IAB">
    %tile = ktdp.construct_access_tile %view[%row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<32x64xindex, strided<[1, 32]>, "IAB"> -> !ktdp.access_tile<1x64xindex>
    %data = ktdp.load %tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xindex>
    %init = tensor.empty() : tensor<1x64xindex>
    %res = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                           iterator_types = ["parallel", "parallel"]}
        ins(%data : tensor<1x64xindex>) outs(%init : tensor<1x64xindex>) {
      ^bb0(%in: index, %out: index):
        %c = arith.constant 1 : index
        %s = arith.addi %in, %c : index
        linalg.yield %s : index
    } -> tensor<1x64xindex>
    ktdp.store %res, %tile : tensor<1x64xindex>, !ktdp.access_tile<1x64xindex>
    return
  }
}