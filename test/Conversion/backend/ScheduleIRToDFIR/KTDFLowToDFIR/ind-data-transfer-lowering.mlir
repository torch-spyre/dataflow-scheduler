// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (0, 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_4:.+]] = affine_map<() -> ()>
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0) -> (0)>
// CHECK: #[[$ATTR_6:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_7:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @ind_transfer_gather_to_memref() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-iab", type = "iab"} : index
// CHECK-NEXT:     %[[GET_UNIT_4:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-iab", type = "iab"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_3]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_4]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_0:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_2]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_1:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_0]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_1]]} : index, index, memref<32xindex>
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:       agen.composite_indirect_load_and_store indirect_src:%[[GET_LOGICAL_MEMORY_VIEW_1]]{{\[}}%[[CONSTANT_0]]] direct_src:%[[GET_LOGICAL_MEMORY_VIEW_0]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]] direct_dst:%[[ALLOC_0]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_1:.*]]:vector<64xf16>)
// CHECK-NEXT:        {load_direct_time_addr_map = #[[$ATTR_2]], load_indirect_time_addr_map = #[[$ATTR_1]], load_order = #[[$ATTR_3]], load_set = #[[$ATTR_6]], store_direct_time_addr_map = #[[$ATTR_2]], store_indirect_time_addr_map = #[[$ATTR_4]], store_order = #[[$ATTR_3]], store_set = #[[$ATTR_6]], time_order = #[[$ATTR_1]], time_set = #[[$ATTR_7]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<32xindex>, memref<64x64xf16>, memref<1x64xf16, "L1">
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @ind_transfer_gather_to_fifo() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:     %[[GET_UNIT_4:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"} : index
// CHECK-NEXT:     %[[GET_UNIT_5:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-iab", type = "iab"} : index
// CHECK-NEXT:     %[[GET_UNIT_6:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-iab", type = "iab"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_5]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_6]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_0:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_4]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_1:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_0]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_1]]} : index, index, memref<32xindex>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       agen.composite_indirect_load_and_store indirect_src:%[[GET_LOGICAL_MEMORY_VIEW_1]]{{\[}}%[[CONSTANT_0]]] direct_src:%[[GET_LOGICAL_MEMORY_VIEW_0]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]] direct_dst:%[[GET_LOGICAL_MEMORY_VIEW_0]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_1:.*]]:vector<64xf16>)
// CHECK-NEXT:        {load_direct_time_addr_map = #[[$ATTR_2]], load_indirect_time_addr_map = #[[$ATTR_1]], load_order = #[[$ATTR_3]], load_set = #[[$ATTR_6]], store_direct_time_addr_map = #[[$ATTR_5]], store_indirect_time_addr_map = #[[$ATTR_4]], store_order = #[[$ATTR_3]], store_set = #[[$ATTR_6]], time_order = #[[$ATTR_1]], time_set = #[[$ATTR_7]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         dataflow.send %[[QUERY_MAP_1]], %[[VAL_1]] : vector<64xf16>
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<32xindex>, memref<64x64xf16>, memref<64x64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_2:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @ind_transfer_scatter() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-iab", type = "iab"} : index
// CHECK-NEXT:     %[[GET_UNIT_4:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-iab", type = "iab"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_3]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_4]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_0:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_2]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_1:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_0]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_1]]} : index, index, memref<32xindex>
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:       agen.composite_indirect_load_and_store direct_src:%[[ALLOC_0]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]] indirect_dst:%[[GET_LOGICAL_MEMORY_VIEW_1]]{{\[}}%[[CONSTANT_0]]] direct_dst:%[[GET_LOGICAL_MEMORY_VIEW_0]]{{\[}}%[[CONSTANT_1]], %[[CONSTANT_1]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_1:.*]]:vector<64xf16>)
// CHECK-NEXT:        {load_direct_time_addr_map = #[[$ATTR_2]], load_indirect_time_addr_map = #[[$ATTR_4]], load_order = #[[$ATTR_3]], load_set = #[[$ATTR_6]], store_direct_time_addr_map = #[[$ATTR_2]], store_indirect_time_addr_map = #[[$ATTR_1]], store_order = #[[$ATTR_3]], store_set = #[[$ATTR_6]], time_order = #[[$ATTR_1]], time_set = #[[$ATTR_7]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<1x64xf16, "L1">, memref<32xindex>, memref<64x64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @ind_transfer_gather_loop_iab_index() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 32 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-iab", type = "iab"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]]) : {
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_2]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_0:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_1]], %[[CONSTANT_2]] {layout_map = #[[$ATTR_0]]} : index, index, memref<64x64xf16>
// CHECK-NEXT:       %[[GET_LOGICAL_MEMORY_VIEW_1:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_0]], %[[CONSTANT_2]] {layout_map = #[[$ATTR_1]]} : index, index, memref<32xindex>
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_1]] step %[[CONSTANT_0]] {
// CHECK-NEXT:         agen.composite_indirect_load_and_store indirect_src:%[[GET_LOGICAL_MEMORY_VIEW_1]]{{\[}}%[[VAL_1]]] direct_src:%[[GET_LOGICAL_MEMORY_VIEW_0]]{{\[}}%[[CONSTANT_2]], %[[CONSTANT_2]]] direct_dst:%[[ALLOC_0]]{{\[}}%[[CONSTANT_2]], %[[CONSTANT_2]]]
// CHECK-NEXT:          time_symbols(), load_iv(%[[VAL_2:.*]]:vector<64xf16>)
// CHECK-NEXT:          {load_direct_time_addr_map = #[[$ATTR_2]], load_indirect_time_addr_map = #[[$ATTR_1]], load_order = #[[$ATTR_3]], load_set = #[[$ATTR_6]], store_direct_time_addr_map = #[[$ATTR_2]], store_indirect_time_addr_map = #[[$ATTR_4]], store_order = #[[$ATTR_3]], store_set = #[[$ATTR_6]], time_order = #[[$ATTR_1]], time_set = #[[$ATTR_7]]}
// CHECK-NEXT:         {
// CHECK-NEXT:           agen.yield
// CHECK-NEXT:         } : memref<32xindex>, memref<64x64xf16>, memref<1x64xf16, "L1">
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// Verify that ktdf.ind_data_transfer is lowered to
// agen.composite_indirect_load_and_store for all four modes:
//   1. Gather → memref: IAB drives the source address; result goes into a
//      local staging buffer.  A self-sync (MNILU→MNILU) is emitted before the
//      indirect op.  No body beyond agen.yield.
//   2. Gather → FIFO: IAB drives the source address; result is forwarded
//      via dataflow.send in the body.  A self-sync is emitted before the
//      indirect op.
//   3. Scatter: IAB drives the destination address; source is a local buffer.
//      A self-sync (MNISU→MNISU) is emitted before the indirect op.  No body
//      beyond agen.yield.
//   4. Gather → memref with an scf.for induction variable as the IAB index:
//      the self-sync is placed before the enclosing scf.for loop.
//
// In all cases the original ktdf.ind_data_transfer must not survive.


#ddr_coord_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#iab_coord_set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // -------------------------------------------------------------------
  // Gather to memref: IAB entry drives the source base address;
  // a [1, 64] tile is loaded and written into a local staging buffer.
  // -------------------------------------------------------------------
  func.func @ind_transfer_gather_to_memref() attributes {grid = [2]} {
    %mnilu0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %mnilu1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_mnilu = uniform.def_immutable_mapping([%c0 -> %mnilu0], [%c1 -> %mnilu1]) : index
    %u_mnilu   = uniform.query_map(map:%map_mnilu, key:%tile_id) : index

    %data_mv  = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1]
                    {coordinate_set = #ddr_coord_set, memory_space = #ktdp.memory_space<global>}
                    : memref<64x64xf16>
    %data_msc = memref.memory_space_cast %data_mv : memref<64x64xf16> to memref<64x64xf16, "DDR">
    %data     = memref.reinterpret_cast %data_msc to offset: [%c0], sizes: [64, 64], strides: [64, 1]
                    : memref<64x64xf16, "DDR"> to memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">

    ktdf_lowering.execute_on %u_mnilu {
      %iab     = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
                     {coordinate_set = #iab_coord_set, memory_space = "IAB"}
                     : memref<32xindex, "IAB">
      %staging = memref.alloc() : memref<1x64xf16, "L1">
      ktdf.ind_data_transfer
          ind_src = %iab[%c1]
          dir_src = %data[%c0, %c0] size [1, 64]
          ind_dst = none
          dir_dst = %staging[%c0, %c0] size [1, 64]
          {dataflow_scheduler.access_granularity = 64 : i64}
          : memref<32xindex, "IAB">,
            memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">,
            none,
            memref<1x64xf16, "L1">
    }
    return
  }

  // -------------------------------------------------------------------
  // Gather to FIFO: IAB entry drives the source base address; the loaded
  // vector is forwarded to the SFU via dataflow.send in the body.
  // The outer execute_on spans both MNILU and SFU so the FIFO endpoint
  // ("SFU") can be resolved.
  // -------------------------------------------------------------------
  func.func @ind_transfer_gather_to_fifo() attributes {grid = [2]} {
    %mnilu0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %mnilu1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_mnilu = uniform.def_immutable_mapping([%c0 -> %mnilu0], [%c1 -> %mnilu1]) : index
    %u_mnilu   = uniform.query_map(map:%map_mnilu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNILU" -> "SFU", 64xf16>

    %data_mv  = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1]
                    {coordinate_set = #ddr_coord_set, memory_space = #ktdp.memory_space<global>}
                    : memref<64x64xf16>
    %data_msc = memref.memory_space_cast %data_mv : memref<64x64xf16> to memref<64x64xf16, "DDR">
    %data     = memref.reinterpret_cast %data_msc to offset: [%c0], sizes: [64, 64], strides: [64, 1]
                    : memref<64x64xf16, "DDR"> to memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">

    ktdf_lowering.execute_on %u_mnilu, %u_sfu {
      ktdf_lowering.execute_on %u_mnilu {
        %iab  = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
                    {coordinate_set = #iab_coord_set, memory_space = "IAB"}
                    : memref<32xindex, "IAB">
        ktdf.ind_data_transfer
            ind_src = %iab[%c1]
            dir_src = %data[%c0, %c0] size [1, 64]
            ind_dst = none
            dir_dst = %fifo           size [64]
            {dataflow_scheduler.access_granularity = 64 : i64}
            : memref<32xindex, "IAB">,
              memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">,
              none,
              !ktdf.fifo.slot<"MNILU" -> "SFU", 64xf16>
      }
    }
    return
  }

  // -------------------------------------------------------------------
  // Scatter: IAB entry drives the destination base address; a [1, 64]
  // tile is written from a local staging buffer to global memory.
  // -------------------------------------------------------------------
  func.func @ind_transfer_scatter() attributes {grid = [2]} {
    %mnisu0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
    %mnisu1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_mnisu = uniform.def_immutable_mapping([%c0 -> %mnisu0], [%c1 -> %mnisu1]) : index
    %u_mnisu   = uniform.query_map(map:%map_mnisu, key:%tile_id) : index

    %dst_mv  = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1]
                   {coordinate_set = #ddr_coord_set, memory_space = #ktdp.memory_space<global>}
                   : memref<64x64xf16>
    %dst_msc = memref.memory_space_cast %dst_mv : memref<64x64xf16> to memref<64x64xf16, "DDR">
    %dst     = memref.reinterpret_cast %dst_msc to offset: [%c0], sizes: [64, 64], strides: [64, 1]
                   : memref<64x64xf16, "DDR"> to memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">

    ktdf_lowering.execute_on %u_mnisu {
      %iab     = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
                     {coordinate_set = #iab_coord_set, memory_space = "IAB"}
                     : memref<32xindex, "IAB">
      %staging = memref.alloc() : memref<1x64xf16, "L1">
      ktdf.ind_data_transfer
          ind_src = none
          dir_src = %staging[%c0, %c0] size [1, 64]
          ind_dst = %iab[%c1]
          dir_dst = %dst[%c0, %c0]     size [1, 64]
          {dataflow_scheduler.access_granularity = 64 : i64}
          : none,
            memref<1x64xf16, "L1">,
            memref<32xindex, "IAB">,
            memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">
    }
    return
  }

  // -------------------------------------------------------------------
  // Gather to memref with loop iter arg as IAB index: the IAB entry
  // index is supplied by an scf.for induction variable rather than
  // a compile-time constant, verifying that the self-sync is placed
  // before the enclosing scf.for loop.
  // -------------------------------------------------------------------
  func.func @ind_transfer_gather_loop_iab_index() attributes {grid = [2]} {
    %mnilu0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0  = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c1  = arith.constant 1 : index
    %map_mnilu = uniform.def_immutable_mapping([%c0 -> %mnilu0]) : index
    %u_mnilu   = uniform.query_map(map:%map_mnilu, key:%tile_id) : index

    %data_mv  = ktdp.construct_memory_view %c0, sizes: [64, 64], strides: [64, 1]
                    {coordinate_set = #ddr_coord_set, memory_space = #ktdp.memory_space<global>}
                    : memref<64x64xf16>
    %data_msc = memref.memory_space_cast %data_mv : memref<64x64xf16> to memref<64x64xf16, "DDR">
    %data     = memref.reinterpret_cast %data_msc to offset: [%c0], sizes: [64, 64], strides: [64, 1]
                    : memref<64x64xf16, "DDR"> to memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">

    ktdf_lowering.execute_on %u_mnilu {
      %iab     = ktdp_lowering.construct_memory_view %c0, sizes: [32], strides: [1]
                     {coordinate_set = #iab_coord_set, memory_space = "IAB"}
                     : memref<32xindex, "IAB">
      %staging = memref.alloc() : memref<1x64xf16, "L1">
      scf.for %iv = %c0 to %c32 step %c1 {
        ktdf.ind_data_transfer
            ind_src = %iab[%iv]
            dir_src = %data[%c0, %c0] size [1, 64]
            ind_dst = none
            dir_dst = %staging[%c0, %c0] size [1, 64]
            {dataflow_scheduler.access_granularity = 64 : i64}
            : memref<32xindex, "IAB">,
              memref<64x64xf16, strided<[64, 1], offset: ?>, "DDR">,
              none,
              memref<1x64xf16, "L1">
      }
    }
    return
  }
}
