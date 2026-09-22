// RUN: dataflow-scheduler-opt --construct-three-stage-pipeline %s | FileCheck %s

// Gather over a variables space with a reduction dimension, and with extent-1
// dimensions written as the equality `d0 == 0` -- the spelling
// buildIntegerSetFromSizes emits, and the one getSizesFromIntegerSet has to
// invert.
//
// The reduction dimension is not tiled (tile size 0), so it gets no loop and no
// IV: its subscript is pinned to the origin and its transfer size is the whole
// variables-space extent, which is the one place those extents are observable in
// the emitted sizes. The parallel dimension still goes through the tiled IV.

// CHECK-LABEL:   func.func @gather_reduction() {
// CHECK:           %[[BASE_RC:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [64, 2, 64], strides: [64, 4096, 1]
// CHECK:           scf.for %[[I1:.*]] =
// CHECK:             scf.for %[[I2:.*]] =
// Only the parallel dimension is tiled, so there is exactly one tiled loop.
// CHECK:               scf.for %[[T0:.*]] =
// CHECK-NOT:             scf.for
// CHECK:                 ktdf.pipeline {
// The load FIFO carries the whole reduction extent (64), the store FIFO one element.
// CHECK:                   %[[PRV:.*]]:5 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 1xf16>,
// CHECK:                   ktdf.stage depends_in(none) depends_out(%[[PRV]]#2) {
// CHECK-NEXT:                scf.if %{{.*}} {
// CHECK-NEXT:                  ktdf.data_transfer from %{{.*}}[0] size [32] to %{{.*}}[0] size [32] {dataflow_scheduler.throttle = 1 : i64}
// CHECK-NEXT:                }
// CHECK-NEXT:                ktdf.ind_data_transfer
// CHECK-NEXT:                  ind_src = %{{.*}}{{\[}}%[[I2]]]
// Base dim 1 carries the tiled IV; base dim 2 is the untiled reduction dim, so it
// is pinned to the origin and its size is the variables_space_set extent (64).
// CHECK-NEXT:                  dir_src = %[[BASE_RC]]{{\[}}%{{.*}}, %[[T0]] + %{{.*}}, 0] size [1, 1, 64]
// CHECK-NEXT:                  ind_dst = none
// CHECK-NEXT:                  dir_dst = %[[PRV]]#0 size [1, 64]
// CHECK-NEXT:                  {dataflow_scheduler.throttle = 64 : i64} : memref<32xindex, strided<[1], offset: ?>, "IAB">, memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">, none, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
// CHECK-NEXT:              }
// CHECK:                   ktdf.stage depends_in(%[[PRV]]#2) depends_out(%[[PRV]]#3) {
// CHECK:                     linalg.generic {{.*}}iterator_types = ["parallel", "reduction"]
// CHECK:                   ktdf.stage depends_in(%[[PRV]]#3) depends_out(%[[PRV]]#4) {
// CHECK-NEXT:                ktdf.data_transfer from %[[PRV]]#1 size [1] to %{{.*}}{{\[}}%[[T0]]] size [1]
// CHECK-NOT: ktdp_lowering.construct_indirect_access_tile


// The addr_buf row (%i1), the desc_2 slice (%i1, %i2) and the base tensor.
#set_addr_buf  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_addr_row  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_iab       = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_base      = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// The variables space: 1x64, matching the SFU tile exactly.
#set_vars      = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
#set_dst       = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_dst_tile  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>

#map1 = affine_map<(d0) -> (d0)>
#map_red = affine_map<(d0, d1) -> (d0)>
#set_dst_1d = affine_set<(d0) : (d0 >= 0, -d0 >= 0)>
#set_dst_full = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @gather_reduction() {
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

        // Reduce over the second dimension of the gathered tile.
        %empty = tensor.empty() : tensor<1xf16>
        %result = linalg.generic
            {indexing_maps = [#map2, #map_red],
             iterator_types = ["parallel", "reduction"]}
            ins(%src : tensor<1x64xf16>) outs(%empty : tensor<1xf16>) {
          ^bb0(%in: f16, %out: f16):
            %cf10 = arith.constant 1.000000e+01 : f16
            %sum = arith.addf %in, %cf10 : f16
            linalg.yield %sum : f16
        } -> tensor<1xf16>

        // Direct destination: one row per (%i1, %i2).
        %desc_dst = ktdp.construct_memory_view %c10000, sizes: [64], strides: [1]
            {coordinate_set = #set_dst_full, memory_space = #ktdp.memory_space<global>}
            : memref<64xf16>
        %dst_tile = ktdp.construct_access_tile %desc_dst[%row]
            {access_tile_order = #map1, access_tile_set = #set_dst_1d}
            : memref<64xf16> -> !ktdp.access_tile<1xindex>
        ktdp.store %result, %dst_tile
            : tensor<1xf16>, !ktdp.access_tile<1xindex>
      }
    }
    return
  }
}
