// Verify that a child-module function containing NO ktdp.construct_indirect_access_tile
// passes through IndirectComputeGroupSplit completely unchanged (strict no-op path).
//
// The pass must not modify, add, or remove any op.  The two CHECK-LABEL lines confirm
// that both the orchestrator module and the extracted child module survive intact.
//
// Note: no ktdf_arch.device is present — the pass must NOT attempt a device lookup
// when there are no indirect ops (confirmed by the work.empty() early-return).
//
// RUN: dataflow-scheduler-opt --indirect-compute-group-split %s | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK:         call @local_schedule_0() : () -> ()
// CHECK:         return
// CHECK:         func.func private @local_schedule_0()

// CHECK-LABEL: module @local_schedule_0
// CHECK:         func.func @local_schedule_0
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C1024:.*]] = arith.constant 1024 : index
// CHECK:           %[[MV:.*]] = ktdp.construct_memory_view
// CHECK:           %[[AT:.*]] = ktdp.construct_access_tile
// CHECK:           %[[LD:.*]] = ktdp.load
// CHECK:           ktdp.store
// CHECK:           return

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set_full = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  module {
    // Orchestrator child module (no sym_name → identified as orchestrator).
    func.func @main() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  module @local_schedule_0 {
    // Extracted child module: direct (non-indirect) load/compute/store only.
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0   = arith.constant 0    : index
      %c1024 = arith.constant 1024 : index
      %c12288 = arith.constant 12288 : index
      %mv_a = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = #set_full, memory_space = #ktdp.memory_space<global>
      } : memref<96x64xf16>
      %at_a = ktdp.construct_access_tile %mv_a[%c0, %c0] {
        access_tile_set = #set, access_tile_order = #map
      } : memref<96x64xf16> -> !ktdp.access_tile<4x64xindex>
      %tile = ktdp.load %at_a : !ktdp.access_tile<4x64xindex> -> tensor<4x64xf16>
      %mv_c = ktdp.construct_memory_view %c12288, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = #set_full, memory_space = #ktdp.memory_space<global>
      } : memref<96x64xf16>
      %at_c = ktdp.construct_access_tile %mv_c[%c0, %c0] {
        access_tile_set = #set, access_tile_order = #map
      } : memref<96x64xf16> -> !ktdp.access_tile<4x64xindex>
      ktdp.store %tile, %at_c : tensor<4x64xf16>, !ktdp.access_tile<4x64xindex>
      return
    }
  }
}
