// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>

// The execute_on work region is lowered into a dataflow.program_unit. Inside
// it, the captured get_compute_tile_id is replaced by a core map+query keyed
// on the program_unit iter arg, and the unit-resolution query_map collapses
// into it. Rule A then folds the arith.divui/remui/addi addressing chain over
// that query into a single folded query_map (the map values become the folded
// per-core base offsets), so no tile-id use or arith op on it remains in the
// body after lowering.
// CHECK-LABEL: func.func @compute_tile_id_core_query
// CHECK:         dataflow.program_unit iter_arg : %{{.*}} -> (%{{.*}}) : {
// CHECK:           %{{.*}} = uniform.def_immutable_mapping
// CHECK:           %{{.*}} = uniform.query_map
// CHECK-NOT:       arith.divui
// CHECK-NOT:       arith.remui
// CHECK-NOT:       ktdp.get_compute_tile_id
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @compute_tile_id_core_query() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %tid = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %resolved = uniform.query_map(map:%map, key:%tid) : index
    %c2 = arith.constant 2 : index
    %c12 = arith.constant 12 : index
    %c64 = arith.constant 64 : index
    %c64000 = arith.constant 64000 : index
    %row = arith.divui %tid, %c2 : index
    %col = arith.remui %tid, %c2 : index
    %off = arith.addi %row, %col : index
    %base = arith.addi %off, %c64000 : index
    %mv = ktdp.construct_memory_view %base, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
    %msc = memref.memory_space_cast %mv : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
    %rc = memref.reinterpret_cast %msc to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
    %cast = memref.cast %rc : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
    scf.for %arg1 = %c0 to %c12 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c1 {
        ktdf_lowering.execute_on %resolved {
          %alloc = memref.alloc() : memref<12x1x64x64xf16, "L1">
          ktdf.data_transfer from %cast[%arg1, %c0, %arg2, %c0] size [1, 1, 1, 64] to %alloc[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.access_granularity = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}