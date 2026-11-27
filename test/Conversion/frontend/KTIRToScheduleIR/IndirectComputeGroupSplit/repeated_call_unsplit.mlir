// RUN: dataflow-scheduler-opt --indirect-compute-group-split %s | FileCheck %s

// The orchestrator calls @helper, a program the pass does not split, twice.
// Only a function the pass splits must have a single call site, since its
// address buffer is filled before that one call; repeated calls to any other
// are left as they are. The split of @local_schedule_0 is gather_basic.mlir's.

// CHECK-LABEL: func.func @orchestrator
// CHECK:         call @helper() : () -> ()
// CHECK-NEXT:    call @local_schedule_0_idx_to_addr() : () -> ()
// CHECK-NEXT:    call @local_schedule_0() : () -> ()
// CHECK-NEXT:    call @helper() : () -> ()
// CHECK-NEXT:    return
// CHECK-NOT:   @helper_idx_to_addr
// CHECK-LABEL: module @helper {
// CHECK-NOT:     ktdp_lowering.
// CHECK-LABEL: module @local_schedule_0_idx_to_addr
// CHECK-LABEL: module @local_schedule_0 {
// CHECK:         ktdp_lowering.construct_indirect_access_tile

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#set  = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>

module {
  // Inline device declaration with indirect_address_buffer feature (entry_type = si32).
  // The pass reads the IAB entry type from this node to derive computeType = i32.
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  module {
    // Orchestrator child module (no sym_name → identified as orchestrator).
    func.func @orchestrator() attributes {grid = [1]} {
      call @helper() : () -> ()
      call @local_schedule_0() : () -> ()
      call @helper() : () -> ()
      return
    }
    func.func private @local_schedule_0()
    func.func private @helper()
  }

  // A program with no indirect access, which the pass does not split.
  module @helper {
    func.func @helper() attributes {grid = [1]} {
      return
    }
  }

  module @local_schedule_0 {
    // Extracted child module: contains the canonical gather indirect op.
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0    = arith.constant 0     : index
      %c100  = arith.constant 100   : index
      %c1024 = arith.constant 1024  : index
      %c10000 = arith.constant 10000 : index

      // Index memref: 2×32 si32, base = %c100, stride[0] = 32, stride[1] = 1.
      %desc_0 = ktdp.construct_memory_view %c100, sizes: [2, 32], strides: [32, 1] {
        coordinate_set = #set,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32xsi32>

      // Data (embedding) memref: 64×2×64 f16, base = %c1024.
      // Indirect dimension D = 0 (stride[0] = 64 — the static stride for dim 0).
      %desc_1 = ktdp.construct_memory_view %c1024, sizes: [64, 2, 64], strides: [64, 4096, 1] {
        coordinate_set = #set1,
        memory_space = #ktdp.memory_space<global>
      } : memref<64x2x64xf16>

      // Output memref: 2×32×2×64 f16.
      %desc_2 = ktdp.construct_memory_view %c10000, sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1] {
        coordinate_set = #set2,
        memory_space = #ktdp.memory_space<global>
      } : memref<2x32x2x64xf16>

      // Indirect access tile: dim 0 of desc_1 is indirect (via desc_0).
      %tmp1 = ktdp.construct_indirect_access_tile intermediate_variables(%arg5, %arg6, %arg7, %arg8)
          %desc_1[ind(%desc_0[%c0 + %arg5, %c0 + %arg6]), (%c0 + %arg7), (%arg8)]
          {variables_space_order = #map, variables_space_set = #set2}
          : memref<64x2x64xf16>, memref<2x32xsi32> -> !ktdp.access_tile<2x32x2x64xindex>

      // Gather: load from the indirect tile.
      %tmp1_0 = ktdp.load %tmp1 : !ktdp.access_tile<2x32x2x64xindex> -> tensor<2x32x2x64xf16>

      // Element-wise compute.
      %12 = tensor.empty() : tensor<2x32x2x64xf16>
      %13 = linalg.generic {
          indexing_maps = [#map, #map],
          iterator_types = ["parallel", "parallel", "parallel", "parallel"]
        } ins(%tmp1_0 : tensor<2x32x2x64xf16>) outs(%12 : tensor<2x32x2x64xf16>) {
        ^bb0(%in: f16, %out: f16):
          %cf10 = arith.constant 10.000000e+00 : f16
          %14 = arith.addf %in, %cf10 : f16
          linalg.yield %14 : f16
      } -> tensor<2x32x2x64xf16>

      // Direct store to output.
      %at_out = ktdp.construct_access_tile %desc_2[%c0, %c0, %c0, %c0] {
        access_tile_order = #map,
        access_tile_set = #set2
      } : memref<2x32x2x64xf16> -> !ktdp.access_tile<2x32x2x64xindex>
      ktdp.store %13, %at_out : tensor<2x32x2x64xf16>, !ktdp.access_tile<2x32x2x64xindex>
      return
    }
  }

}
