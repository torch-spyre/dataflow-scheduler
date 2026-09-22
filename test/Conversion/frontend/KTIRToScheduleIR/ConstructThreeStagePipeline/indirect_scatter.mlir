// RUN: dataflow-scheduler-opt --construct-three-stage-pipeline %s | FileCheck %s

// Scatter: the indirect tile is on the store side, so stage 1 is an ordinary
// direct load and the buffer fill plus the ktdf.ind_data_transfer both land in
// stage 3, with ind_dst set and ind_src none.

// CHECK-LABEL:   func.func @scatter_pipeline() {
// CHECK:           %[[ADDR_MV:.*]] = ktdp.construct_memory_view {{.*}} : memref<2x32xindex, #ktdp.memory_space<global>>
// CHECK:           %[[IAB_MV:.*]] = ktdp_lowering.construct_memory_view {{.*}} : memref<32xindex, "IAB">
// CHECK:           %[[SRC_MV:.*]] = ktdp.construct_memory_view {{.*}} : memref<64x64xf16>
// CHECK:           %[[DST_MV:.*]] = ktdp.construct_memory_view {{.*}} : memref<64x2x64xf16, strided<[4096, 64, 1]>>
// CHECK:           %[[IAB_RC:.*]] = memref.reinterpret_cast %[[IAB_MV]] to offset: [0], sizes: [32], strides: [1] : memref<32xindex, "IAB"> to memref<32xindex, strided<[1], offset: ?>, "IAB">
// CHECK:           %[[SRC_MSC:.*]] = memref.memory_space_cast %[[SRC_MV]] : memref<64x64xf16> to memref<64x64xf16, "DDR">
// CHECK:           %[[ADDR_MSC:.*]] = memref.memory_space_cast %[[ADDR_MV]] : memref<2x32xindex, #ktdp.memory_space<global>> to memref<2x32xindex, "DDR">
// CHECK:           %[[DST_MSC:.*]] = memref.memory_space_cast %[[DST_MV]] : memref<64x2x64xf16, strided<[4096, 64, 1]>> to memref<64x2x64xf16, strided<[4096, 64, 1]>, "DDR">
// CHECK:           %[[DST_RC:.*]] = memref.reinterpret_cast %[[DST_MSC]] to offset: [0], sizes: [64, 2, 64], strides: [4096, 64, 1] : memref<64x2x64xf16, strided<[4096, 64, 1]>, "DDR"> to memref<64x2x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">

// CHECK:           scf.for %[[I1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             %[[ADDR_OFF:.*]] = arith.muli %[[I1]], %{{.*}} : index
// CHECK:             %[[ADDR_RC:.*]] = memref.reinterpret_cast %[[ADDR_MSC]] to offset: {{\[}}%[[ADDR_OFF]]], sizes: [32], strides: [1] : memref<2x32xindex, "DDR"> to memref<32xindex, strided<[1], offset: ?>, "DDR">
// CHECK:             scf.for %[[I2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:               %[[ROW_I1:.*]] = arith.muli %[[I1]], %{{.*}} : index
// CHECK:               %[[ROW:.*]] = arith.addi %[[ROW_I1]], %[[I2]] : index
// CHECK:               %[[SRC_OFF:.*]] = arith.muli %[[ROW]], %{{.*}} : index
// CHECK:               %[[SRC_RC:.*]] = memref.reinterpret_cast %[[SRC_MSC]] to offset: {{\[}}%[[SRC_OFF]]], sizes: [1, 64], strides: [64, 1] : memref<64x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK:               scf.for %[[T0:.*]] =
// CHECK:                 scf.for %[[T1:.*]] =
// CHECK:                   ktdf.pipeline {
// CHECK:                     %[[PRV:.*]]:5 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>,

// Stage 1 is an ordinary direct load: nothing indirect on the load side.
// CHECK:                     ktdf.stage depends_in(none) depends_out(%[[PRV]]#2) {
// CHECK-NEXT:                  ktdf.data_transfer from %[[SRC_RC]]{{\[}}%[[T0]], %[[T1]]] size [1, 64] to %[[PRV]]#0 size [1, 64] {{.*}}
// CHECK-NEXT:                }

// CHECK:                     ktdf.stage depends_in(%[[PRV]]#2) depends_out(%[[PRV]]#3) {
// CHECK:                       ktdf.write_to_fifo %{{.*}}, %[[PRV]]#1
// CHECK-NEXT:                } {applicable_units = ["SFU"]}

// Stage 3 carries the fill and the indirect store, in that order.
// CHECK-NEXT:                ktdf.stage depends_in(%[[PRV]]#3) depends_out(%[[PRV]]#4) {
// CHECK-NEXT:                  scf.if %{{.*}} {
// CHECK-NEXT:                    ktdf.data_transfer from %[[ADDR_RC]][0] size [32] to %[[IAB_RC]][0] size [32] {dataflow_scheduler.throttle = 1 : i64} : memref<32xindex, strided<[1], offset: ?>, "DDR">, memref<32xindex, strided<[1], offset: ?>, "IAB">
// CHECK-NEXT:                  }
// CHECK-NEXT:                  ktdf.ind_data_transfer
// CHECK-NEXT:                    ind_src = none
// CHECK-NEXT:                    dir_src = %[[PRV]]#1 size [1, 64]
// CHECK-NEXT:                    ind_dst = %[[IAB_RC]]{{\[}}%[[I2]]]
// CHECK-NEXT:                    dir_dst = %[[DST_RC]]{{\[}}%{{.*}}, %[[T0]] + %{{.*}}, %[[T1]]] size [1, 1, 64]
// CHECK-NEXT:                    {dataflow_scheduler.throttle = 64 : i64} : none, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<32xindex, strided<[1], offset: ?>, "IAB">, memref<64x2x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:                }

// CHECK-NOT: ktdp_lowering.construct_indirect_access_tile
// CHECK-NOT: ktdp.store

// -----------------------------------------------------------------------
// Scatter with pinned-outer-dims load: the source access tile has two
// size-1 pinned outer dims (1×1×2×64) that are collapsed before the
// linalg op.  The pass must traverse the collapse_shape to find the linalg
// indexing map and re-express it in the 4-D access-tile rank.
// Stage 1 reads the 4-D tile; stage 2 runs the element-wise op on the
// tiled (post-collapse, post-extract_slice) shape; stage 3 carries the IAB
// fill and the indirect scatter.
// -----------------------------------------------------------------------

// CHECK-LABEL:   func.func @scatter_pinned_src_pipeline() {
// CHECK:           %[[ADDR_MV:.*]] = ktdp.construct_memory_view {{.*}} : memref<2x32xindex, #ktdp.memory_space<global>>
// CHECK:           %[[IAB_MV:.*]] = ktdp_lowering.construct_memory_view {{.*}} : memref<32xindex, "IAB">
// CHECK:           %[[SRC_MV:.*]] = ktdp.construct_memory_view {{.*}} : memref<2x32x2x64xf16>
// CHECK:           %[[DST_MV:.*]] = ktdp.construct_memory_view {{.*}} : memref<64x2x64xf16>
// CHECK:           %[[IAB_RC:.*]] = memref.reinterpret_cast %[[IAB_MV]] to offset: [0], sizes: [32], strides: [1] : memref<32xindex, "IAB"> to memref<32xindex, strided<[1], offset: ?>, "IAB">
// CHECK:           %[[SRC_MSC:.*]] = memref.memory_space_cast %[[SRC_MV]] : memref<2x32x2x64xf16> to memref<2x32x2x64xf16, "DDR">
// CHECK:           %[[ADDR_MSC:.*]] = memref.memory_space_cast %[[ADDR_MV]] : memref<2x32xindex, #ktdp.memory_space<global>> to memref<2x32xindex, "DDR">
// CHECK:           %[[DST_MSC:.*]] = memref.memory_space_cast %[[DST_MV]] : memref<64x2x64xf16> to memref<64x2x64xf16, "DDR">
// CHECK:           %[[DST_RC:.*]] = memref.reinterpret_cast %[[DST_MSC]] to offset: [0], sizes: [64, 2, 64], strides: [4096, 64, 1] : memref<64x2x64xf16, "DDR"> to memref<64x2x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">

// CHECK:           scf.for %[[I1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:             %[[ADDR_OFF:.*]] = arith.muli %[[I1]], %{{.*}} : index
// CHECK:             %[[ADDR_RC:.*]] = memref.reinterpret_cast %[[ADDR_MSC]] to offset: {{\[}}%[[ADDR_OFF]]], sizes: [32], strides: [1] : memref<2x32xindex, "DDR"> to memref<32xindex, strided<[1], offset: ?>, "DDR">
// CHECK:             scf.for %[[I2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// The 4-D source tile is reinterpret_cast'd outside the tiling loops.
// CHECK:               %[[OFF0:.*]] = arith.muli %[[I1]], %{{.*}} : index
// CHECK:               %[[OFF1:.*]] = arith.muli %[[I2]], %{{.*}} : index
// CHECK:               %[[SRC_OFF:.*]] = arith.addi %[[OFF0]], %[[OFF1]] : index
// CHECK:               %[[SRC_RC:.*]] = memref.reinterpret_cast %[[SRC_MSC]] to offset: {{\[}}%[[SRC_OFF]]], sizes: [1, 1, 2, 64], strides: [4096, 128, 64, 1] : memref<2x32x2x64xf16, "DDR"> to memref<1x1x2x64xf16, strided<[4096, 128, 64, 1], offset: ?>, "DDR">
// Two tiling loops over the collapsed 2×64 output shape.
// CHECK:               scf.for %[[T0:.*]] =
// CHECK:                 scf.for %[[T1:.*]] =
// CHECK:                   ktdf.pipeline {
// CHECK:                     %[[PRV:.*]]:5 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>,

// Stage 1: ordinary direct load of a slice of the 4-D source tile.
// CHECK:                     ktdf.stage depends_in(none) depends_out(%[[PRV]]#2) {
// CHECK-NEXT:                  ktdf.data_transfer from %[[SRC_RC]][0, 0, %[[T0]], %[[T1]]] size [1, 1, 1, 64] to %[[PRV]]#0 size [1, 1, 1, 64] {{.*}}
// CHECK-NEXT:                }

// Stage 2: compute on the tiled 1×64 shape read from the FIFO.
// CHECK:                     ktdf.stage depends_in(%[[PRV]]#2) depends_out(%[[PRV]]#3) {
// CHECK:                       %[[RF:.*]] = ktdf.read_from_fifo %[[PRV]]#0 : <"DDR" -> "SFU", 64xf16> -> tensor<1x64xf16>
// CHECK:                       ktdf.write_to_fifo %{{.*}}, %[[PRV]]#1
// CHECK-NEXT:                } {applicable_units = ["SFU"]}

// Stage 3: IAB fill (conditional) then indirect scatter.
// CHECK-NEXT:                ktdf.stage depends_in(%[[PRV]]#3) depends_out(%[[PRV]]#4) {
// CHECK-NEXT:                  scf.if %{{.*}} {
// CHECK-NEXT:                    ktdf.data_transfer from %[[ADDR_RC]][0] size [32] to %[[IAB_RC]][0] size [32] {dataflow_scheduler.throttle = 1 : i64} : memref<32xindex, strided<[1], offset: ?>, "DDR">, memref<32xindex, strided<[1], offset: ?>, "IAB">
// CHECK-NEXT:                  }
// CHECK-NEXT:                  ktdf.ind_data_transfer
// CHECK-NEXT:                    ind_src = none
// CHECK-NEXT:                    dir_src = %[[PRV]]#1 size [1, 64]
// CHECK-NEXT:                    ind_dst = %[[IAB_RC]]{{\[}}%[[I2]]]
// CHECK-NEXT:                    dir_dst = %[[DST_RC]]{{\[}}%{{.*}}, %[[T0]] + %{{.*}}, %[[T1]]] size [1, 1, 64]
// CHECK-NEXT:                    {dataflow_scheduler.throttle = 64 : i64} : none, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>, memref<32xindex, strided<[1], offset: ?>, "IAB">, memref<64x2x64xf16, strided<[4096, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:                }

// CHECK-NOT: ktdp_lowering.construct_indirect_access_tile
// CHECK-NOT: ktdp.store

#set_addr_buf  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_addr_row  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_iab       = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_src       = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_src_tile  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_dst       = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// The variables space: 1x64, matching the SFU tile exactly.
#set_vars      = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>

// Additional sets/maps for @scatter_pinned_src_pipeline.
// 4-D source memref bounds (2×32×2×64).
#set_src_pinned     = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// 4-D access tile set: d0 and d1 pinned to 0, d2 in [0,1], d3 in [0,63].
#set_src_row_pinned = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// Variables space: 2×64 (matches the linalg output and the dst tile).
#set_vars2          = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>

#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @scatter_pipeline() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c2  = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c1000 = arith.constant 1000 : index
    %addr_buf_base = arith.constant 999 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_addr_buf, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>

    %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    scf.for %i1 = %c0 to %c2 step %c1 {
      scf.for %i2 = %c0 to %c32 step %c1 {
        %row_i1 = arith.muli %i1, %c32 : index
        %row = arith.addi %row_i1, %i2 : index

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

        // Direct source.
        %desc_src = ktdp.construct_memory_view %c1000, sizes: [64, 64], strides: [64, 1]
            {coordinate_set = #set_src, memory_space = #ktdp.memory_space<global>}
            : memref<64x64xf16>
        %src_tile = ktdp.construct_access_tile %desc_src[%row, %c0]
            {access_tile_order = #map2, access_tile_set = #set_src_tile}
            : memref<64x64xf16> -> !ktdp.access_tile<1x64xindex>
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

        // Indirect destination: buffer entry %i2 supplies the row address. The
        // original base is folded into each buffer entry, so the view's own base
        // offset is 0.
        %desc_dst = ktdp.construct_memory_view %c0, sizes: [64, 2, 64], strides: [4096, 64, 1]
            {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
            : memref<64x2x64xf16, strided<[4096, 64, 1]>>
        %dst_tile = ktdp_lowering.construct_indirect_access_tile
            intermediate_variables(%arg7, %arg8)
            base_ptr = %iab_mv[%i2]
            %desc_dst[%c0, %c0 + %arg7, %arg8]
            {variables_space_order = #map2, variables_space_set = #set_vars}
            : memref<64x2x64xf16, strided<[4096, 64, 1]>>, memref<32xindex, "IAB">
            -> !ktdp.access_tile<1x64xindex>
        ktdp.store %result, %dst_tile
            : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
      }
    }
    return
  }

  // Scatter with pinned-outer-dims load: source is 1×1×2×64, collapsed to
  // 2×64 before the linalg op, then tiled to 1×64 by the pass.
  func.func @scatter_pinned_src_pipeline() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c2  = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c1000 = arith.constant 1000 : index
    %addr_buf_base = arith.constant 999 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_addr_buf, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>

    %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    scf.for %i1 = %c0 to %c2 step %c1 {
      scf.for %i2 = %c0 to %c32 step %c1 {
        %eq0 = arith.cmpi eq, %i2, %c0 : index
        scf.if %eq0 {
          // Load one row of the address buffer and store into the IAB.
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

        // Source: 4-D memref with two size-1 pinned outer dims; the access
        // tile is 1×1×2×64 and is collapsed to 2×64 before the linalg op.
        %desc_src = ktdp.construct_memory_view %c1000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
            {coordinate_set = #set_src_pinned, memory_space = #ktdp.memory_space<global>}
            : memref<2x32x2x64xf16>
        %src_tile_row = ktdp.construct_access_tile %desc_src[%i1, %i2, %c0, %c0]
            {access_tile_order = #map3, access_tile_set = #set_src_row_pinned}
            : memref<2x32x2x64xf16> -> !ktdp.access_tile<1x1x2x64xindex>
        %src_stick = ktdp.load %src_tile_row
            : !ktdp.access_tile<1x1x2x64xindex> -> tensor<1x1x2x64xf16>
        // Collapse the two pinned outer dims before the element-wise compute.
        %src = tensor.collapse_shape %src_stick [[0, 1, 2], [3]]
            : tensor<1x1x2x64xf16> into tensor<2x64xf16>

        %empty = tensor.empty() : tensor<2x64xf16>
        %result = linalg.generic
            {indexing_maps = [#map2, #map2],
             iterator_types = ["parallel", "parallel"]}
            ins(%src : tensor<2x64xf16>) outs(%empty : tensor<2x64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %cf10 = arith.constant 1.000000e+01 : f16
            %sum = arith.addf %in, %cf10 : f16
            linalg.yield %sum : f16
        } -> tensor<2x64xf16>

        // Indirect destination.
        %desc_dst = ktdp.construct_memory_view %c0, sizes: [64, 2, 64], strides: [4096, 64, 1]
            {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
            : memref<64x2x64xf16>
        %dst_tile = ktdp_lowering.construct_indirect_access_tile
            intermediate_variables(%arg7, %arg8)
            base_ptr = %iab_mv[%i2]
            %desc_dst[(%c0), (%c0 + %arg7), (%arg8)]
            {variables_space_order = #map2, variables_space_set = #set_vars2}
            : memref<64x2x64xf16>, memref<32xindex, "IAB">
            -> !ktdp.access_tile<2x64xindex>
        ktdp.store %result, %dst_tile
            : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>
      }
    }
    return
  }
}
