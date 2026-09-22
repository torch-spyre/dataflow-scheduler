// RUN: dataflow-scheduler-opt --construct-three-stage-pipeline %s | FileCheck %s

// Gather where tiling splits the variables space: the space is 2x64xf16 but SFU
// has 64 f16 lanes, so determineTileSizes yields [1, 64] and the pipeline runs
// under an extra two-trip loop. This is the regression test for the direct
// subscripts: each one must project the tiled loop IVs through the op's
// per-dimension subscript maps, and each size must be the tiled extent, not the
// variables-space extent. Reading the variables space literally would emit
// `dir_src = %base[0, 0, 0] size [1, 2, 64]` with a 128xf16 FIFO and silently
// load the same row twice while never loading the second one.

// CHECK-LABEL:   func.func @gather_pipeline_tiled() {
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[BASE_RC:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [64, 2, 64], strides: [64, 4096, 1]

// CHECK:           scf.for %[[I1:.*]] = %{{.*}} to %[[C2]] step %{{.*}} {
// CHECK:             scf.for %[[I2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {

// The tile splits the variables space, so dim 0 gets a two-trip tiled loop...
// CHECK:               %[[TC2:.*]] = arith.constant 2 : index
// CHECK:               scf.for %[[T0:.*]] = %{{.*}} to %[[TC2]] step %{{.*}} {
// CHECK:                 scf.for %[[T1:.*]] =

// ...and the FIFO carries one tile (64 elements), not the whole 2x64 space.
// CHECK:                   %[[PRV:.*]]:5 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 64xf16>,

// CHECK:                     ktdf.stage depends_in(none) depends_out(%[[PRV]]#2) {
// CHECK-NEXT:                  scf.if %{{.*}} {
// CHECK-NEXT:                    ktdf.data_transfer from %{{.*}}[0] size [32] to %{{.*}}[0] size [32] {dataflow_scheduler.throttle = 1 : i64}
// CHECK-NEXT:                  }
// CHECK-NEXT:                  ktdf.ind_data_transfer
// CHECK-NEXT:                    ind_src = %{{.*}}{{\[}}%[[I2]]]
// The regression: base dim 1 carries the tiled IV and its size is the tile
// extent (1), not the variables-space extent (2).
// CHECK-NEXT:                    dir_src = %[[BASE_RC]]{{\[}}%{{.*}}, %[[T0]] + %{{.*}}, %[[T1]]] size [1, 1, 64]
// CHECK-NEXT:                    ind_dst = none
// CHECK-NEXT:                    dir_dst = %[[PRV]]#0 size [1, 64]
// CHECK-NEXT:                    {dataflow_scheduler.throttle = 64 : i64} : memref<32xindex, strided<[1], offset: ?>, "IAB">, memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">, none, !ktdf.fifo.slot<"DDR" -> "SFU", 64xf16>
// CHECK-NEXT:                }

// The compute stage sees one tile too.
// CHECK:                  ktdf.read_from_fifo %[[PRV]]#0 : <"DDR" -> "SFU", 64xf16> -> tensor<1x64xf16>

// CHECK-NOT: ktdp_lowering.construct_indirect_access_tile

#set_addr_buf  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_addr_row  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_iab       = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_base      = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// The variables space: 2x64, which the 64-lane tile splits in two.
#set_vars      = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_dst       = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_dst_tile  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>

#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @gather_pipeline_tiled() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c2  = arith.constant 2 : index
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
        // Two destination rows per iteration.
        %entry_i1 = arith.muli %i1, %c32 : index
        %entry = arith.addi %entry_i1, %i2 : index
        %row = arith.muli %entry, %c2 : index

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

        %desc_src = ktdp.construct_memory_view %c0, sizes: [64, 2, 64], strides: [64, 4096, 1]
            {coordinate_set = #set_base, memory_space = #ktdp.memory_space<global>}
            : memref<64x2x64xf16, strided<[64, 4096, 1]>>
        %src_tile = ktdp_lowering.construct_indirect_access_tile
            intermediate_variables(%arg7, %arg8)
            base_ptr = %iab_mv[%i2]
            %desc_src[%c0, %c0 + %arg7, %arg8]
            {variables_space_order = #map2, variables_space_set = #set_vars}
            : memref<64x2x64xf16, strided<[64, 4096, 1]>>, memref<32xindex, "IAB">
            -> !ktdp.access_tile<2x64xindex>
        %src = ktdp.load %src_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>

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

        %desc_dst = ktdp.construct_memory_view %c10000, sizes: [128, 64], strides: [64, 1]
            {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
            : memref<128x64xf16>
        %dst_tile = ktdp.construct_access_tile %desc_dst[%row, %c0]
            {access_tile_order = #map2, access_tile_set = #set_dst_tile}
            : memref<128x64xf16> -> !ktdp.access_tile<2x64xindex>
        ktdp.store %result, %dst_tile
            : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>
      }
    }
    return
  }
}
