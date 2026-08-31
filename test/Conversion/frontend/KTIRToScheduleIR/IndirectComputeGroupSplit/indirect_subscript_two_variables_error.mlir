// Verify that an indirect subscript summing two live variables is rejected.
//
// `ind(%desc_0[%arg5 + %arg6, %arg6])` selects index element
// [arg5 + arg6, arg6].  $ind_addr_buf_dim_positions names one variable of the
// unified space and nothing else, so no position can stand for `arg5 + arg6`:
// position(%arg5) and position(%arg6) each select a different entry than the
// sum does.
//
// Unlike the offset and floordiv cases, picking one of the two variables is
// superficially plausible here — both are intermediate variables, so a rule
// phrased as "take the intermediate one" would silently accept this and drop
// the other term.  This test pins the rejection.
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split \
// RUN:   --verify-diagnostics %s

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  ktdf_arch.device @iab_device import("Inputs/iab_device.mlir")

  module {
    func.func @orchestrator() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }

  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0     = arith.constant 0     : index
      %c100   = arith.constant 100   : index
      %c1000  = arith.constant 1000  : index
      %c10000 = arith.constant 10000 : index

      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>
      %desc_1 = ktdp.construct_memory_view %c1000, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>
      %desc_2 = ktdp.construct_memory_view %c10000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
        coordinate_set = #set2,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32x2x64xf16>

      // Result 0 is `d1 + d2` — %arg5 + %arg6, two live variables.
      // expected-error @+1 {{indirect-compute-group-split: result 0 of per_dim_subscript_maps[0] (affine_map<(d0, d1, d2, d3, d4) -> (d1 + d2, d2)>) cannot be used as an address-buffer subscript: it must name exactly one variable, optionally added to captured variables that are literal zeros, because $ind_addr_buf_dim_positions encodes a variable position and no arithmetic}}
      %tile = ktdp.construct_indirect_access_tile intermediate_variables(%arg5, %arg6, %arg7, %arg8)
          %desc_1[ind(%desc_0[%arg5 + %arg6, %arg6]), (%c0 + %arg7), (%arg8)]
          {variables_space_order = #map, variables_space_set = #set2}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>
      %val = ktdp.load %tile : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>
      %at_out = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0] {
        access_tile_order = #map,
        access_tile_set = #set2
      } : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
      ktdp.store %val, %at_out : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
      return
    }
  }
}
