// Verify that a child-module function containing MORE THAN ONE
// ktdp.construct_indirect_access_tile triggers precondition P3:
// the pass emits an error on the FuncOp and signals failure.
//
// No ktdf_arch.device is needed — the P3 check fires before any device lookup.
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split \
// RUN:   --verify-diagnostics %s

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#set_idx  = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set_base = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>

module {
  module {
    func.func @main() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  module @local_schedule_0 {
    // expected-error @+1 {{indirect-compute-group-split: more than one indirect tensor in function 'local_schedule_0' is not supported}}
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0    = arith.constant 0    : index
      %c100  = arith.constant 100  : index
      %c1000 = arith.constant 1000 : index
      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>
      %desc_1 = ktdp.construct_memory_view %c1000, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set_base,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>
      // First indirect op.
      %tile_a = ktdp.construct_indirect_access_tile intermediate_variables(%arg0, %arg1, %arg2, %arg3)
          %desc_1[ind(%desc_0[%c0 + %arg0, %c0 + %arg1]), (%c0 + %arg2), (%arg3)]
          {variables_space_order = #map, variables_space_set = #set_idx}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>
      // Second indirect op — same structure, triggers P3.
      %tile_b = ktdp.construct_indirect_access_tile intermediate_variables(%arg4, %arg5, %arg6, %arg7)
          %desc_1[ind(%desc_0[%c0 + %arg4, %c0 + %arg5]), (%c0 + %arg6), (%arg7)]
          {variables_space_order = #map, variables_space_set = #set_idx}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>
      %loaded_a = ktdp.load %tile_a : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>
      %loaded_b = ktdp.load %tile_b : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>
      return
    }
  }
}
