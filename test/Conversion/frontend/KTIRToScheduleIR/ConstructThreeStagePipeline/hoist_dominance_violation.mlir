// RUN: dataflow-scheduler-opt --construct-three-stage-pipeline --verify-diagnostics %s

// Negative test: a memory view whose base address is derived from a loop IV
// cannot be hoisted to the pipeline anchor.  The pass must reject it with a
// clear diagnostic rather than producing broken IR.

#set_addr_buf  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_addr_row  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_iab       = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_base      = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set_vars      = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_dst       = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_dst_tile  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @hoist_dominance_violation(%src_base: index) {
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

    %desc_dst = ktdp.construct_memory_view %c10000, sizes: [64, 64], strides: [64, 1]
        {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
        : memref<64x64xf16>

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

        %src_addr = arith.addi %src_base, %i1 : index
        %desc_src = ktdp.construct_memory_view %src_addr, sizes: [64, 2, 64], strides: [64, 4096, 1]
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
            {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]}
            ins(%src : tensor<1x64xf16>) outs(%empty : tensor<1x64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %cf10 = arith.constant 1.000000e+01 : f16
            %sum = arith.addf %in, %cf10 : f16
            linalg.yield %sum : f16
        } -> tensor<1x64xf16>

        // %dst_addr depends on the loop IV — %desc_dst cannot be hoisted above
        // the tiled loop anchor.
        %dst_addr = arith.addi %src_base, %i2 : index
        // expected-error @+1 {{cannot hoist op: an operand does not dominate the hoist anchor}}
        %desc_dst_iv = ktdp.construct_memory_view %dst_addr, sizes: [64, 64], strides: [64, 1]
            {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
            : memref<64x64xf16>
        %dst_tile = ktdp.construct_access_tile %desc_dst_iv[%row, %c0]
            {access_tile_order = #map2, access_tile_set = #set_dst_tile}
            : memref<64x64xf16> -> !ktdp.access_tile<1x64xindex>
        ktdp.store %result, %dst_tile
            : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
      }
    }
    return
  }
}
