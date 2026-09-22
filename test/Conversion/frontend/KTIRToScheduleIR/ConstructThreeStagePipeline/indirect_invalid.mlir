// RUN: dataflow-scheduler-opt --construct-three-stage-pipeline --verify-diagnostics %s

// Forms of indirect access this pass cannot lower. They are diagnosed rather
// than asserted: until the earlier steps of the indirect pipeline exist this IR
// is hand-written, and the op verifier checks none of these properties.

#set_addr_buf  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_addr_row  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_addr_all  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_iab       = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
#set_iab_2d    = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set_base      = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set_vars      = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// Not a box: the first constraint couples two dimensions.
#set_vars_skew = affine_set<(d0, d1) : (d0 + d1 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_dst       = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_dst_tile  = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>

#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // The fill is what tells the pass which load/store pair is not a pipeline
  // load/store. Without one there is nothing to co-locate the transfer with, and
  // the buffer would be dereferenced before it was ever populated.
  func.func @no_fill() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c10000 = arith.constant 10000 : index

    %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    scf.for %i2 = %c0 to %c32 step %c1 {
      %desc_src = ktdp.construct_memory_view %c0, sizes: [64, 2, 64], strides: [64, 4096, 1]
          {coordinate_set = #set_base, memory_space = #ktdp.memory_space<global>}
          : memref<64x2x64xf16, strided<[64, 4096, 1]>>
      // expected-error @+1 {{no indirect-address-buffer fill found for this indirect access tile}}
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
      %desc_dst = ktdp.construct_memory_view %c10000, sizes: [64, 64], strides: [64, 1]
          {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
          : memref<64x64xf16>
      %dst_tile = ktdp.construct_access_tile %desc_dst[%i2, %c0]
          {access_tile_order = #map2, access_tile_set = #set_dst_tile}
          : memref<64x64xf16> -> !ktdp.access_tile<1x64xindex>
      ktdp.store %result, %dst_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }
    return
  }

  // Two indirect tiles would need two buffer fills in one stage, and there is
  // only one buffer per unit.
  func.func @two_indirect_tiles() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 999 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_addr_buf, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>
    %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    scf.for %i2 = %c0 to %c32 step %c1 {
      %eq0 = arith.cmpi eq, %i2, %c0 : index
      scf.if %eq0 {
        %addr_row = ktdp.construct_access_tile %addr_buf[%c0, %c0]
            {access_tile_order = #map2, access_tile_set = #set_addr_row}
            : memref<2x32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<1x32xindex>
        %addr_vals_row = ktdp.load %addr_row : !ktdp.access_tile<1x32xindex> -> tensor<1x32xindex>
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
      %tile_a = ktdp_lowering.construct_indirect_access_tile
          intermediate_variables(%arg7, %arg8)
          base_ptr = %iab_mv[%i2]
          %desc_src[%c0, %c0 + %arg7, %arg8]
          {variables_space_order = #map2, variables_space_set = #set_vars}
          : memref<64x2x64xf16, strided<[64, 4096, 1]>>, memref<32xindex, "IAB">
          -> !ktdp.access_tile<1x64xindex>
      // expected-error @+1 {{more than one indirect access tile in a single pipeline is not supported}}
      %tile_b = ktdp_lowering.construct_indirect_access_tile
          intermediate_variables(%arg7, %arg8)
          base_ptr = %iab_mv[%i2]
          %desc_src[%c0, %c0 + %arg7, %arg8]
          {variables_space_order = #map2, variables_space_set = #set_vars}
          : memref<64x2x64xf16, strided<[64, 4096, 1]>>, memref<32xindex, "IAB">
          -> !ktdp.access_tile<1x64xindex>
      %a = ktdp.load %tile_a : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
      %b = ktdp.load %tile_b : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
      %empty = tensor.empty() : tensor<1x64xf16>
      %result = linalg.generic
          {indexing_maps = [#map2, #map2, #map2],
           iterator_types = ["parallel", "parallel"]}
          ins(%a, %b : tensor<1x64xf16>, tensor<1x64xf16>) outs(%empty : tensor<1x64xf16>) {
        ^bb0(%in0: f16, %in1: f16, %out: f16):
          %sum = arith.addf %in0, %in1 : f16
          linalg.yield %sum : f16
      } -> tensor<1x64xf16>
      %desc_dst = ktdp.construct_memory_view %c10000, sizes: [64, 64], strides: [64, 1]
          {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
          : memref<64x64xf16>
      %dst_tile = ktdp.construct_access_tile %desc_dst[%i2, %c0]
          {access_tile_order = #map2, access_tile_set = #set_dst_tile}
          : memref<64x64xf16> -> !ktdp.access_tile<1x64xindex>
      ktdp.store %result, %dst_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }
    return
  }

  // A fill whose source tile has a non-unit leading dimension cannot be
  // collapsed to rank-1.  The IAB and dest tile are 2D to keep the store
  // valid (shape must match); the check fires on the source tile.
  func.func @iab_fill_tile_not_collapsible() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 999 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_addr_buf, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>
    %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_iab_2d, memory_space = "IAB"} : memref<2x32xindex, "IAB">

    scf.for %i2 = %c0 to %c32 step %c1 {
      %eq0 = arith.cmpi eq, %i2, %c0 : index
      scf.if %eq0 {
        // expected-error @+1 {{indirect-address-buffer fill tile must collapse to rank-1 after dropping leading unit dimensions}}
        %addr_all = ktdp.construct_access_tile %addr_buf[%c0, %c0]
            {access_tile_order = #map2, access_tile_set = #set_addr_all}
            : memref<2x32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<2x32xindex>
        %addr_vals = ktdp.load %addr_all : !ktdp.access_tile<2x32xindex> -> tensor<2x32xindex>
        %iab_tile = ktdp.construct_access_tile %iab_mv[%c0, %c0]
            {access_tile_order = #map2, access_tile_set = #set_iab_2d}
            : memref<2x32xindex, "IAB"> -> !ktdp.access_tile<2x32xindex>
        ktdp.store %addr_vals, %iab_tile : tensor<2x32xindex>, !ktdp.access_tile<2x32xindex>
      }

      %desc_src = ktdp.construct_memory_view %c0, sizes: [64, 2, 64], strides: [64, 4096, 1]
          {coordinate_set = #set_base, memory_space = #ktdp.memory_space<global>}
          : memref<64x2x64xf16, strided<[64, 4096, 1]>>
      %src_tile = ktdp_lowering.construct_indirect_access_tile
          intermediate_variables(%arg7, %arg8)
          base_ptr = %iab_mv[%c0, %i2]
          %desc_src[%c0, %c0 + %arg7, %arg8]
          {variables_space_order = #map2, variables_space_set = #set_vars}
          : memref<64x2x64xf16, strided<[64, 4096, 1]>>, memref<2x32xindex, "IAB">
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
      %desc_dst = ktdp.construct_memory_view %c10000, sizes: [64, 64], strides: [64, 1]
          {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
          : memref<64x64xf16>
      %dst_tile = ktdp.construct_access_tile %desc_dst[%i2, %c0]
          {access_tile_order = #map2, access_tile_set = #set_dst_tile}
          : memref<64x64xf16> -> !ktdp.access_tile<1x64xindex>
      ktdp.store %result, %dst_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }
    return
  }

  // A rank-2 IAB memref means a whole window per transfer, which
  // ktdf.ind_data_transfer cannot express: it dereferences one entry at a time.
  func.func @iab_rank_two() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 999 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_addr_buf, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>
    %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_iab_2d, memory_space = "IAB"} : memref<2x32xindex, "IAB">

    scf.for %i2 = %c0 to %c32 step %c1 {
      %eq0 = arith.cmpi eq, %i2, %c0 : index
      scf.if %eq0 {
        %addr_row = ktdp.construct_access_tile %addr_buf[%c0, %c0]
            {access_tile_order = #map2, access_tile_set = #set_addr_row}
            : memref<2x32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<1x32xindex>
        %addr_vals = ktdp.load %addr_row : !ktdp.access_tile<1x32xindex> -> tensor<1x32xindex>
        %iab_tile = ktdp.construct_access_tile %iab_mv[%c0, %c0]
            {access_tile_order = #map2, access_tile_set = #set_iab_2d}
            : memref<2x32xindex, "IAB"> -> !ktdp.access_tile<1x32xindex>
        ktdp.store %addr_vals, %iab_tile : tensor<1x32xindex>, !ktdp.access_tile<1x32xindex>
      }

      %desc_src = ktdp.construct_memory_view %c0, sizes: [64, 2, 64], strides: [64, 4096, 1]
          {coordinate_set = #set_base, memory_space = #ktdp.memory_space<global>}
          : memref<64x2x64xf16, strided<[64, 4096, 1]>>
      // expected-error @+1 {{indirect address buffer must be rank 1 for ktdf.ind_data_transfer}}
      %src_tile = ktdp_lowering.construct_indirect_access_tile
          intermediate_variables(%arg7, %arg8)
          base_ptr = %iab_mv[%c0, %i2]
          %desc_src[%c0, %c0 + %arg7, %arg8]
          {variables_space_order = #map2, variables_space_set = #set_vars}
          : memref<64x2x64xf16, strided<[64, 4096, 1]>>, memref<2x32xindex, "IAB">
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
      %desc_dst = ktdp.construct_memory_view %c10000, sizes: [64, 64], strides: [64, 1]
          {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
          : memref<64x64xf16>
      %dst_tile = ktdp.construct_access_tile %desc_dst[%i2, %c0]
          {access_tile_order = #map2, access_tile_set = #set_dst_tile}
          : memref<64x64xf16> -> !ktdp.access_tile<1x64xindex>
      ktdp.store %result, %dst_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }
    return
  }

  // An intermediate variable as the buffer subscript is the same problem seen
  // from the other side: the tile spans every entry of the window.
  func.func @iab_subscript_is_intermediate() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 999 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_addr_buf, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>
    %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    scf.for %i2 = %c0 to %c32 step %c1 {
      %eq0 = arith.cmpi eq, %i2, %c0 : index
      scf.if %eq0 {
        %addr_row = ktdp.construct_access_tile %addr_buf[%c0, %c0]
            {access_tile_order = #map2, access_tile_set = #set_addr_row}
            : memref<2x32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<1x32xindex>
        %addr_vals_row = ktdp.load %addr_row : !ktdp.access_tile<1x32xindex> -> tensor<1x32xindex>
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
      // expected-error @+1 {{indirect address buffer subscript is an intermediate variable}}
      %src_tile = ktdp_lowering.construct_indirect_access_tile
          intermediate_variables(%arg7, %arg8)
          base_ptr = %iab_mv[%arg7]
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
      %desc_dst = ktdp.construct_memory_view %c10000, sizes: [64, 64], strides: [64, 1]
          {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
          : memref<64x64xf16>
      %dst_tile = ktdp.construct_access_tile %desc_dst[%i2, %c0]
          {access_tile_order = #map2, access_tile_set = #set_dst_tile}
          : memref<64x64xf16> -> !ktdp.access_tile<1x64xindex>
      ktdp.store %result, %dst_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }
    return
  }

  // The tile extents come from variables_space_set, so a set that is not a box
  // has no per-dimension extents to read.
  func.func @variables_space_not_box() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 999 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_addr_buf, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>
    %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    scf.for %i2 = %c0 to %c32 step %c1 {
      %eq0 = arith.cmpi eq, %i2, %c0 : index
      scf.if %eq0 {
        %addr_row = ktdp.construct_access_tile %addr_buf[%c0, %c0]
            {access_tile_order = #map2, access_tile_set = #set_addr_row}
            : memref<2x32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<1x32xindex>
        %addr_vals_row = ktdp.load %addr_row : !ktdp.access_tile<1x32xindex> -> tensor<1x32xindex>
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
      // expected-error @+1 {{variables_space_set is not in box form}}
      %src_tile = ktdp_lowering.construct_indirect_access_tile
          intermediate_variables(%arg7, %arg8)
          base_ptr = %iab_mv[%i2]
          %desc_src[%c0, %c0 + %arg7, %arg8]
          {variables_space_order = #map2, variables_space_set = #set_vars_skew}
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
      %desc_dst = ktdp.construct_memory_view %c10000, sizes: [64, 64], strides: [64, 1]
          {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
          : memref<64x64xf16>
      %dst_tile = ktdp.construct_access_tile %desc_dst[%i2, %c0]
          {access_tile_order = #map2, access_tile_set = #set_dst_tile}
          : memref<64x64xf16> -> !ktdp.access_tile<1x64xindex>
      ktdp.store %result, %dst_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }
    return
  }

  // variables_space_order has to name one intermediate variable per
  // variables-space dimension; a short map leaves dimensions unaccounted for.
  func.func @order_dim_count_mismatch() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 999 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_addr_buf, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>
    %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    scf.for %i2 = %c0 to %c32 step %c1 {
      %eq0 = arith.cmpi eq, %i2, %c0 : index
      scf.if %eq0 {
        %addr_row = ktdp.construct_access_tile %addr_buf[%c0, %c0]
            {access_tile_order = #map2, access_tile_set = #set_addr_row}
            : memref<2x32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<1x32xindex>
        %addr_vals_row = ktdp.load %addr_row : !ktdp.access_tile<1x32xindex> -> tensor<1x32xindex>
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
      // expected-error @+1 {{variables_space_order maps 1 intermediate variable(s) to 1 dimension(s)}}
      %src_tile = ktdp_lowering.construct_indirect_access_tile
          intermediate_variables(%arg7, %arg8)
          base_ptr = %iab_mv[%i2]
          %desc_src[%c0, %c0 + %arg7, %arg8]
          {variables_space_order = #map1, variables_space_set = #set_vars}
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
      %desc_dst = ktdp.construct_memory_view %c10000, sizes: [64, 64], strides: [64, 1]
          {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
          : memref<64x64xf16>
      %dst_tile = ktdp.construct_access_tile %desc_dst[%i2, %c0]
          {access_tile_order = #map2, access_tile_set = #set_dst_tile}
          : memref<64x64xf16> -> !ktdp.access_tile<1x64xindex>
      ktdp.store %result, %dst_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }
    return
  }

  // A base dimension walked by two intermediate variables has no single
  // contiguous extent, so there is no size to give the transfer.
  func.func @two_intermediates_in_one_dim() {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c10000 = arith.constant 10000 : index
    %addr_buf_base = arith.constant 999 : index

    %addr_buf = ktdp.construct_memory_view %addr_buf_base, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set_addr_buf, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>
    %iab_mv = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
        {coordinate_set = #set_iab, memory_space = "IAB"} : memref<32xindex, "IAB">

    scf.for %i2 = %c0 to %c32 step %c1 {
      %eq0 = arith.cmpi eq, %i2, %c0 : index
      scf.if %eq0 {
        %addr_row = ktdp.construct_access_tile %addr_buf[%c0, %c0]
            {access_tile_order = #map2, access_tile_set = #set_addr_row}
            : memref<2x32xindex, #ktdp.memory_space<global>> -> !ktdp.access_tile<1x32xindex>
        %addr_vals_row = ktdp.load %addr_row : !ktdp.access_tile<1x32xindex> -> tensor<1x32xindex>
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
      // expected-error @+1 {{references more than one intermediate variable}}
      %src_tile = ktdp_lowering.construct_indirect_access_tile
          intermediate_variables(%arg7, %arg8)
          base_ptr = %iab_mv[%i2]
          %desc_src[%c0, %arg7 + %arg8, %arg8]
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
      %desc_dst = ktdp.construct_memory_view %c10000, sizes: [64, 64], strides: [64, 1]
          {coordinate_set = #set_dst, memory_space = #ktdp.memory_space<global>}
          : memref<64x64xf16>
      %dst_tile = ktdp.construct_access_tile %desc_dst[%i2, %c0]
          {access_tile_order = #map2, access_tile_set = #set_dst_tile}
          : memref<64x64xf16> -> !ktdp.access_tile<1x64xindex>
      ktdp.store %result, %dst_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }
    return
  }
}
