// RUN: dataflow-scheduler-opt --construct-three-stage-pipeline %s | FileCheck %s

// Gather, degenerate tiling: the variables space is 1x64xf16 and SFU has 64 f16
// lanes, so the tile covers the whole space and the indirect transfer moves it
// in one go. See indirect_gather_tiled.mlir for the case the tile splits.

// The loop-invariant casts leave the loop nest entirely: the base is addressed
// at offset 0 in its full shape because the per-iteration origin lives in the
// transfer's dir_src map, and the buffer window is fixed.  One cast serves both
// the fill's destination and the transfer's ind_src.
// CHECK-LABEL:   func.func @gather_pipeline() {
// CHECK:           %[[ADDR_MV:.*]] = ktdp.construct_memory_view {{.*}} : memref<2x32xindex, #ktdp.memory_space<global>>
// CHECK:           %[[IAB_MV:.*]] = ktdp_lowering.construct_memory_view {{.*}} : memref<32xindex, "IAB">
// CHECK:           %[[BASE_MV:.*]] = ktdp.construct_memory_view {{.*}} : memref<64x2x64xf16, strided<[64, 4096, 1]>>
// CHECK:           %[[DST_MV:.*]] = ktdp.construct_memory_view {{.*}} : memref<64x64xf16>
// The buffer view is already a flat string space, so it takes no memory_space_cast.
// CHECK:           %[[IAB_RC:.*]] = memref.reinterpret_cast %[[IAB_MV]] to offset: [0], sizes: [32], strides: [1] : memref<32xindex, "IAB"> to memref<32xindex, strided<[1], offset: ?>, "IAB">
// CHECK:           %[[ADDR_MSC:.*]] = memref.memory_space_cast %[[ADDR_MV]] : memref<2x32xindex, #ktdp.memory_space<global>> to memref<2x32xindex, "DDR">
// CHECK:           %[[DST_MSC:.*]] = memref.memory_space_cast %[[DST_MV]] : memref<64x64xf16> to memref<64x64xf16, "DDR">
// CHECK:           %[[BASE_MSC:.*]] = memref.memory_space_cast %[[BASE_MV]] : memref<64x2x64xf16, strided<[64, 4096, 1]>> to memref<64x2x64xf16, strided<[64, 4096, 1]>, "DDR">
// CHECK:           %[[BASE_RC:.*]] = memref.reinterpret_cast %[[BASE_MSC]] to offset: [0], sizes: [64, 2, 64], strides: [64, 4096, 1] : memref<64x2x64xf16, strided<[64, 4096, 1]>, "DDR"> to memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">

// The address-row slice is invariant in %i2, so it sits at %i1's level.
// CHECK:           scf.for %[[I1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             %[[ADDR_OFF:.*]] = arith.muli %[[I1]], %{{.*}} : index
// CHECK:             %[[ADDR_RC:.*]] = memref.reinterpret_cast %[[ADDR_MSC]] to offset: {{\[}}%[[ADDR_OFF]]], sizes: [32], strides: [1] : memref<2x32xindex, "DDR"> to memref<32xindex, strided<[1], offset: ?>, "DDR">
// CHECK:             scf.for %[[I2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:               %[[ROW_I1:.*]] = arith.muli %[[I1]], %{{.*}} : index
// CHECK:               %[[ROW:.*]] = arith.addi %[[ROW_I1]], %[[I2]] : index
// CHECK:               %[[DST_OFF:.*]] = arith.muli %[[ROW]], %{{.*}} : index
// CHECK:               %[[DST_RC:.*]] = memref.reinterpret_cast %[[DST_MSC]] to offset: {{\[}}%[[DST_OFF]]], sizes: [1, 64], strides: [64, 1] : memref<64x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">

// The tile equals the variables space, so both tiled loops are single-trip.
// CHECK:               scf.for %[[T0:.*]] =
// CHECK:                 scf.for %[[T1:.*]] =
// CHECK:                   ktdf.pipeline {
// CHECK:                     %[[PRV:.*]]:5 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token) {

// Stage 1: the guarded fill and the indirect load, in that order and in one
// stage -- the buffer is unit-local state that does not cross a stage boundary.
// CHECK:                     ktdf.stage depends_in(none) depends_out(%[[PRV]]#2) {
// CHECK-NEXT:                  scf.if %[[GUARD:.*]] {
// CHECK-NEXT:                    ktdf.data_transfer from %[[ADDR_RC]][0] size [32] to %[[IAB_RC]][0] size [32] {dataflow_scheduler.throttle = 1 : i64} : memref<32xindex, strided<[1], offset: ?>, "DDR">, memref<32xindex, strided<[1], offset: ?>, "IAB">
// CHECK-NEXT:                  }
// CHECK-NEXT:                  ktdf.ind_data_transfer
// CHECK-NEXT:                    ind_src = %[[IAB_RC]]{{\[}}%[[I2]]]
// CHECK-NEXT:                    dir_src = %[[BASE_RC]]{{\[}}%{{.*}}, %[[T0]] + %{{.*}}, %[[T1]]] size [1, 1, 64]
// CHECK-NEXT:                    ind_dst = none
// CHECK-NEXT:                    dir_dst = %[[PRV]]#0 size [1, 64]
// CHECK-NEXT:                    {dataflow_scheduler.throttle = 64 : i64} : memref<32xindex, strided<[1], offset: ?>, "IAB">, memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">, none, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
// CHECK-NEXT:                }

// Stages 2 and 3 are untouched by indirection.
// CHECK-NEXT:                ktdf.stage depends_in(%[[PRV]]#2) depends_out(%[[PRV]]#3) {
// CHECK-NEXT:                  %[[READ:.*]] = ktdf.read_from_fifo %[[PRV]]#0 : <"DDR" -> "SFU", 64xf16> -> tensor<1x64xf16>
// CHECK:                       linalg.generic
// CHECK:                       ktdf.write_to_fifo %{{.*}}, %[[PRV]]#1 : tensor<1x64xf16>, <"SFU" -> "DDR", 64xf16>
// CHECK-NEXT:                } {applicable_units = ["SFU"]}
// CHECK-NEXT:                ktdf.stage depends_in(%[[PRV]]#3) depends_out(%[[PRV]]#4) {
// CHECK-NEXT:                  ktdf.data_transfer from %[[PRV]]#1 size [1, 64] to %[[DST_RC:.*]]{{\[}}%[[T0]], %[[T1]]] size [1, 64]

// No ktdp indirection survives the pass.
// CHECK-NOT: ktdp_lowering.construct_indirect_access_tile
// CHECK-NOT: ktdp.load
// CHECK-NOT: ktdp.store

// The addr_buf row (%i1), the desc_2 slice (%i1, %i2) and the base tensor.
#set_addr_buf  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_addr_row  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_iab       = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_base      = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// The variables space: 1x64, matching the SFU tile exactly.
#set_vars      = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_dst       = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_dst_tile  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>

// Additional sets for gather_pipeline_strided_inner
#set4 = affine_set<(d3) : (d3 >= 0, -d3 + 63 >= 0)>   // 1D: only dim with stride 1 and full coverage
#set5 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// 4D access tile set for the desc_2 store — d0, d1, d2 are all pinned to 0
// (tile-local; %i1, %i2, %i3 are already the base offsets via the access-tile indices),
// d3 ranges over the one retained direct dim (64 elements, unit stride).
#set6 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0)>

#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @gather_pipeline() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c2  = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 999 : index

    // Table of row pointers in global, one row of 32 per %i1.
    %addr_buf = ktdp.construct_memory_view %addr_buf_base, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_addr_buf, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>

    // The hardware indirect address buffer: 32 entries, rank 1.
    %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    scf.for %i1 = %c0 to %c2 step %c1 {
      scf.for %i2 = %c0 to %c32 step %c1 {
        // Destination row for this iteration. Index arithmetic feeding an access
        // tile has to precede the pipeline the tile is hoisted above.
        %row_i1 = arith.muli %i1, %c32 : index
        %row = arith.addi %row_i1, %i2 : index

        // Refill the buffer once per row of the table.
        %eq0 = arith.cmpi eq, %i2, %c0 : index
        scf.if %eq0 {
          %addr_row = ktdp.construct_access_tile %addr_buf[%i1, %c0]
              {access_tile_order = #map2, access_tile_set = #set_addr_row}
              : memref<2x32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<1x32xindex>
          %addr_vals_row = ktdp.load %addr_row
              : !ktdp.access_tile<1x32xindex> -> tensor<1x32xindex>
          %addr_vals = tensor.collapse_shape %addr_vals_row [[0, 1]]
              : tensor<1x32xindex> into tensor<32xindex>
          %iab_tile = ktdp.construct_access_tile %iab_mv[%c0]
              {access_tile_order = #map1, access_tile_set = #set_iab}
              : memref<32xindex, "IAB"> -> !ktdp.access_tile<32xindex>
          ktdp.store %addr_vals, %iab_tile : tensor<32xindex>, !ktdp.access_tile<32xindex>
        }

        // Indirect source: buffer entry %i2 supplies the row address, the
        // remaining two dimensions are direct.
        %desc_src = ktdp.construct_memory_view %c0, sizes: [64, 2, 64], strides: [64, 4096, 1]
            {coordinate_set = #set_base, memory_space = #ktdp.memory_space<global>}
            : memref<64x2x64xf16, strided<[64, 4096, 1]>>
        %src_tile = ktdp_lowering.construct_indirect_access_tile
            intermediate_variables(%arg7, %arg8)
            base_ptr = %iab_mv[%i2]
            %desc_src[%c0, %c0 + %arg7, %arg8]
            {variables_space_order = #map2, variables_space_set = #set_vars}
            : memref<64x2x64xf16, strided<[64, 4096, 1]>>, memref<32xindex, "IAB">
            -> !ktdp.access_tile<1x64xindex>
        %src = ktdp.load %src_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        %empty = tensor.empty() : tensor<1x64xf16>
        %result = linalg.generic
            {indexing_maps = [#map2, #map2],
             iterator_types = ["parallel", "parallel"]}
            ins(%src : tensor<1x64xf16>) outs(%empty : tensor<1x64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %cf10 = arith.constant 1.000000e+01 : f16
            %sum = arith.addf %in, %cf10 : f16
            linalg.yield %sum : f16
        } -> tensor<1x64xf16>

        // Direct destination: one row per (%i1, %i2).
        %desc_dst = ktdp.construct_memory_view %c10000, sizes: [64, 64], strides: [64, 1]
            {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
            : memref<64x64xf16>
        %dst_tile = ktdp.construct_access_tile %desc_dst[%row, %c0]
            {access_tile_order = #map2, access_tile_set = #set_dst_tile}
            : memref<64x64xf16> -> !ktdp.access_tile<1x64xindex>
        ktdp.store %result, %dst_tile
            : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
      }
    }
    return
  }

  // Test with three loop levels and single intermediate variable.
  // This exercises the case where the store has a different shape than the load:
  // the load produces tensor<64xf16> (from a 1D variables space), but the store
  // expands it to tensor<1x1x1x64xf16> to match a 4D access tile (1x1x1x64).
  // - %i1 (outer): 2 iterations
  // - %i2 (middle): 32 iterations, triggers addr_buf refill when %i2 == 0
  // - %i3 (inner): 2 iterations, materialized from strided dimension
  // - Single intermediate variable %arg8 for the densely packed dimension
  func.func @gather_pipeline_strided_inner() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 999 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_addr_buf, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>

    %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    scf.for %i1 = %c0 to %c2 step %c1 {
      scf.for %i2 = %c0 to %c32 step %c1 {
        scf.for %i3 = %c0 to %c2 step %c1 {
          %eq0 = arith.cmpi eq, %i2, %c0 : index
          scf.if %eq0 {
            %addr_row = ktdp.construct_access_tile %addr_buf[%i1, %c0]
                {access_tile_order = #map2, access_tile_set = #set_addr_row}
                : memref<2x32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<1x32xindex>
            %addr_vals_row = ktdp.load %addr_row
                : !ktdp.access_tile<1x32xindex> -> tensor<1x32xindex>
            %addr_vals = tensor.collapse_shape %addr_vals_row [[0, 1]]
                : tensor<1x32xindex> into tensor<32xindex>
            %iab_tile = ktdp.construct_access_tile %iab_mv[%c0]
                {access_tile_order = #map1, access_tile_set = #set_iab}
                : memref<32xindex, "IAB"> -> !ktdp.access_tile<32xindex>
            ktdp.store %addr_vals, %iab_tile : tensor<32xindex>, !ktdp.access_tile<32xindex>
          }

          %desc_1 = ktdp.construct_memory_view %c0, sizes: [64, 2, 64], strides: [64, 4096, 1] {coordinate_set = #set_base, memory_space = #ktdp.memory_space<global>} : memref<64x2x64xf16>
          // Single intermediate variable %arg8 for the innermost densely packed dimension
          // variables_space_set is #set4 (1D: d3), matching the 1D access_tile<64xindex>
          // variables_space_order #map1 maps 1 intermediate var to 1 dimension
          %tmp1 = ktdp_lowering.construct_indirect_access_tile intermediate_variables(%arg8) base_ptr=%iab_mv[%i2] %desc_1[(%c0), (%c0 + %i3), (%arg8)] {variables_space_order = #map1, variables_space_set = #set4} : memref<64x2x64xf16>, memref<32xindex, "IAB"> -> !ktdp.access_tile<64xindex>
          %tmp1_0 = ktdp.load %tmp1 : !ktdp.access_tile<64xindex> -> tensor<64xf16>

          %12 = tensor.empty() : tensor<64xf16>
          %13 = linalg.generic { indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"] } ins(%tmp1_0: tensor<64xf16>) outs(%12 : tensor<64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %cf10 = arith.constant 10.000000e+00 : f16
            %14 = arith.addf %in, %cf10 : f16
            linalg.yield %14 : f16
          } -> tensor<64xf16>

          // Strided dest: sizes [2, 32, 2, 64], strides [4096, 128, 64, 1]
          // The load produces tensor<64xf16>, store expands to 1x1x1x64xf16
          %desc_2 = ktdp.construct_memory_view %c10000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {coordinate_set = #set5, memory_space = #ktdp.memory_space<global>} : memref<2x32x2x64xf16>
          %expanded = tensor.expand_shape %13 [[0, 1, 2, 3]] output_shape [1, 1, 1, 64] : tensor<64xf16> into tensor<1x1x1x64xf16>
          %0 = ktdp.construct_access_tile %desc_2[%i1, %i2, %i3, %c0] {access_tile_order = #map3, access_tile_set = #set6} : memref<2x32x2x64xf16> -> !ktdp.access_tile<1x1x1x64xindex>
          ktdp.store %expanded, %0 : tensor<1x1x1x64xf16>, !ktdp.access_tile<1x1x1x64xindex>
        }
      }
    }
    return
  }
}
