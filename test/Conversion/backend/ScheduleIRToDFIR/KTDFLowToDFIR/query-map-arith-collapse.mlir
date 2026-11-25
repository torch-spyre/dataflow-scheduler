// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" --split-input-file %s | FileCheck %s

// Rule A: divui(query(map), 6) folds into a query over a map of v/6.
// map values 0,6,12,18 -> 0,1,2,3, and the arith.divui disappears.
// CHECK-LABEL: func.func @ra
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         dataflow.program_unit
// CHECK:           %[[M:.*]] = uniform.def_immutable_mapping([%{{.*}} -> %[[C0]]], [%{{.*}} -> %[[C1]]], [%{{.*}} -> %[[C2]]], [%{{.*}} -> %[[C3]]]):index
// CHECK:           uniform.query_map(map:%[[M]], key:%{{.*}}) : index
// CHECK-NOT:       arith.divui
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @ra() attributes {grid = [4]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %2 = dataflow.get_unit {core = 2 : i32, name = "C2-MNILU", type = "MNILU"} : index
    %3 = dataflow.get_unit {core = 3 : i32, name = "C3-MNILU", type = "MNILU"} : index
    %tid = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %umap = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1], [%c2 -> %2], [%c3 -> %3]):index
    %resolved = uniform.query_map(map:%umap, key:%tid) : index
    %c6 = arith.constant 6 : index
    %c12 = arith.constant 12 : index
    %c18 = arith.constant 18 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %amap = uniform.def_immutable_mapping([%0 -> %c0], [%1 -> %c6], [%2 -> %c12], [%3 -> %c18]):index
    %aq = uniform.query_map(map:%amap, key:%tid) : index
    %d = arith.divui %aq, %c6 : index
    %mv = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16, "DDR">
    %rc = memref.reinterpret_cast %mv to offset: [%d], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c64 {
        ktdf_lowering.execute_on %resolved {
          %alloc = memref.alloc() : memref<1x64xf16, "L1">
          ktdf.data_transfer from %rc[%arg1, %arg2] size [1, 64] to %alloc[%c0, %c0] size [1, 64] {dataflow_scheduler.access_granularity = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

// -----

// Rule A: subi with const on the LEFT (reversed operand: const - query).
// map values 0, 4 -> 10-0=10, 10-4=6, and the arith.subi disappears.
// CHECK-LABEL: func.func @rb
// CHECK:         %[[C6:.*]] = arith.constant 6 : index
// CHECK:         %[[C10:.*]] = arith.constant 10 : index
// CHECK:         dataflow.program_unit
// CHECK:           %[[M:.*]] = uniform.def_immutable_mapping([%{{.*}} -> %[[C10]]], [%{{.*}} -> %[[C6]]]):index
// CHECK:           uniform.query_map(map:%[[M]], key:%{{.*}}) : index
// CHECK-NOT:       arith.subi
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @rb() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %tid = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %umap = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %resolved = uniform.query_map(map:%umap, key:%tid) : index
    %c4 = arith.constant 4 : index
    %c10 = arith.constant 10 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %amap = uniform.def_immutable_mapping([%0 -> %c0], [%1 -> %c4]):index
    %aq = uniform.query_map(map:%amap, key:%tid) : index
    %d = arith.subi %c10, %aq : index
    %mv = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16, "DDR">
    %rc = memref.reinterpret_cast %mv to offset: [%d], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c64 {
        ktdf_lowering.execute_on %resolved {
          %alloc = memref.alloc() : memref<1x64xf16, "L1">
          ktdf.data_transfer from %rc[%arg1, %arg2] size [1, 64] to %alloc[%c0, %c0] size [1, 64] {dataflow_scheduler.access_granularity = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

// -----

// Rule A: remui(query, 4).
// map values 7, 9 -> 7%4=3, 9%4=1, and the arith.remui disappears.
// CHECK-LABEL: func.func @rc
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         dataflow.program_unit
// CHECK:           %[[M:.*]] = uniform.def_immutable_mapping([%{{.*}} -> %[[C3]]], [%{{.*}} -> %[[C1]]]):index
// CHECK:           uniform.query_map(map:%[[M]], key:%{{.*}}) : index
// CHECK-NOT:       arith.remui
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @rc() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %tid = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %umap = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %resolved = uniform.query_map(map:%umap, key:%tid) : index
    %c7 = arith.constant 7 : index
    %c9 = arith.constant 9 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %amap = uniform.def_immutable_mapping([%0 -> %c7], [%1 -> %c9]):index
    %aq = uniform.query_map(map:%amap, key:%tid) : index
    %d = arith.remui %aq, %c4 : index
    %mv = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16, "DDR">
    %rc = memref.reinterpret_cast %mv to offset: [%d], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c64 {
        ktdf_lowering.execute_on %resolved {
          %alloc = memref.alloc() : memref<1x64xf16, "L1">
          ktdf.data_transfer from %rc[%arg1, %arg2] size [1, 64] to %alloc[%c0, %c0] size [1, 64] {dataflow_scheduler.access_granularity = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

// -----

// Rule B: addi(query(mapA), query(mapB)) over the SAME key -> one query of a_i+b_i.
// mapA 10,20 + mapB 100,200 -> 110,220, and the arith.addi disappears.
// CHECK-LABEL: func.func @rd
// CHECK:         %[[C220:.*]] = arith.constant 220 : index
// CHECK:         %[[C110:.*]] = arith.constant 110 : index
// CHECK:         dataflow.program_unit
// CHECK:           %[[M:.*]] = uniform.def_immutable_mapping([%{{.*}} -> %[[C110]]], [%{{.*}} -> %[[C220]]]):index
// CHECK:           uniform.query_map(map:%[[M]], key:%{{.*}}) : index
// CHECK-NOT:       arith.addi
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @rd() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %tid = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %umap = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %resolved = uniform.query_map(map:%umap, key:%tid) : index
    %c10 = arith.constant 10 : index
    %c20 = arith.constant 20 : index
    %c100 = arith.constant 100 : index
    %c200 = arith.constant 200 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %ma = uniform.def_immutable_mapping([%0 -> %c10], [%1 -> %c20]):index
    %mb = uniform.def_immutable_mapping([%0 -> %c100], [%1 -> %c200]):index
    %qa = uniform.query_map(map:%ma, key:%tid) : index
    %qb = uniform.query_map(map:%mb, key:%tid) : index
    %d = arith.addi %qa, %qb : index
    %mv = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16, "DDR">
    %rc = memref.reinterpret_cast %mv to offset: [%d], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c64 {
        ktdf_lowering.execute_on %resolved {
          %alloc = memref.alloc() : memref<1x64xf16, "L1">
          ktdf.data_transfer from %rc[%arg1, %arg2] size [1, 64] to %alloc[%c0, %c0] size [1, 64] {dataflow_scheduler.access_granularity = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

// -----

// Full collapse: a chain of binops over one query collapses to ONE query.
// map 0,8 ; row=q/2 ; col=q%2 ; a=row*4 ; b=col*5 ; sum=a+b.
// q=0 -> 0 ; q=8 -> (8/2)*4 + (8%2)*5 = 16 + 0 = 16. No arith binops remain.
// CHECK-LABEL: func.func @chain
// CHECK:         %[[C16:.*]] = arith.constant 16 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         dataflow.program_unit
// CHECK:           %[[M:.*]] = uniform.def_immutable_mapping([%{{.*}} -> %[[C0]]], [%{{.*}} -> %[[C16]]]):index
// CHECK:           uniform.query_map(map:%[[M]], key:%{{.*}}) : index
// CHECK-NOT:       arith.divui
// CHECK-NOT:       arith.remui
// CHECK-NOT:       arith.muli
// CHECK-NOT:       arith.addi
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @chain() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %tid = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %umap = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %resolved = uniform.query_map(map:%umap, key:%tid) : index
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %m = uniform.def_immutable_mapping([%0 -> %c0], [%1 -> %c8]):index
    %q = uniform.query_map(map:%m, key:%tid) : index
    %row = arith.divui %q, %c2 : index
    %col = arith.remui %q, %c2 : index
    %a = arith.muli %row, %c4 : index
    %b = arith.muli %col, %c5 : index
    %sum = arith.addi %a, %b : index
    %mv = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16, "DDR">
    %rc = memref.reinterpret_cast %mv to offset: [%sum], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c64 {
        ktdf_lowering.execute_on %resolved {
          %alloc = memref.alloc() : memref<1x64xf16, "L1">
          ktdf.data_transfer from %rc[%arg1, %arg2] size [1, 64] to %alloc[%c0, %c0] size [1, 64] {dataflow_scheduler.access_granularity = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

// -----

// Rule B negative: queries over DIFFERENT keys are NOT merged; arith.addi survives.
// CHECK-LABEL: func.func @rb_neg
// CHECK:         uniform.query_map
// CHECK:         uniform.query_map
// CHECK:         arith.addi
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @rb_neg(%arg0: index) attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %tid = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %umap = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %resolved = uniform.query_map(map:%umap, key:%tid) : index
    %c10 = arith.constant 10 : index
    %c20 = arith.constant 20 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index
    %ma = uniform.def_immutable_mapping([%0 -> %c10], [%1 -> %c20]):index
    %qa = uniform.query_map(map:%ma, key:%tid) : index
    %qb = uniform.query_map(map:%ma, key:%arg0) : index
    %d = arith.addi %qa, %qb : index
    %mv = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16, "DDR">
    %rc = memref.reinterpret_cast %mv to offset: [%d], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    scf.for %arg1 = %c0 to %c1 step %c1 {
      scf.for %arg2 = %c0 to %c64 step %c64 {
        ktdf_lowering.execute_on %resolved {
          %alloc = memref.alloc() : memref<1x64xf16, "L1">
          ktdf.data_transfer from %rc[%arg1, %arg2] size [1, 64] to %alloc[%c0, %c0] size [1, 64] {dataflow_scheduler.access_granularity = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}