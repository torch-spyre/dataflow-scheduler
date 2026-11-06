// RUN: dataflow-scheduler-opt -split-input-file -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// A kernel's loop variable as a root of the run's symbols, the way a kernel
// argument is. The kernel launches its programs from its loops, so what a
// program is passed from one is a number only the run knows, as much as an
// address the run hands in: it gets an id of its own, after the kernel's
// arguments, declared once at the top of the loop's body, and whatever a program
// computes from it is a symbol derived from that one.
//
// A loop variable no call reads, directly or through arithmetic the kernel does
// on it, keeps its id but is not declared.
//
// The loop variable is never folded or enumerated, though every trip count here
// is known, so what this says holds whether a later stage keeps the loop or
// unrolls it.

// A constant base, walked through the tensor by the loop variable. The kernel
// has no arguments, so the loop variable is the first root.
// CHECK-LABEL:   func.func @test()
// CHECK:           scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        symbol.create_id %[[I]] {symbol_id = -1 : si64} : index
// CHECK-NEXT:        func.call @sched_0(%[[I]])

// The program declares its argument as the loop variable it is passed, and the
// address is derived from it: the rows it is walked by, one symbol, then the
// base the compiler picked, over the word size.
// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index) attributes {grid = [2]} {
// CHECK:           symbol.create_id %arg0 {symbol_id = -1 : si64} : index
// CHECK-NEXT:      %[[ROWS:.*]] = arith.muli %arg0, %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[ROWS]] {symbol_id = -2 : si64} : index
// CHECK-NEXT:      %[[BYTES:.*]] = arith.addi %[[ROWS]], %{{.*}} : index
// CHECK-NEXT:      %[[WORDS:.*]] = arith.divsi %[[BYTES]], %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[WORDS]] {symbol_id = -3 : si64} : index
// CHECK:           symbol.create_symbol {SymbolId = -3 : i64} : index
// CHECK-NOT:       symbol.create_id

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test() attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %i = %c0 to %c3 step %c1 {
        func.call @sched_0(%i) : (index) -> ()
      }
      return
    }
    func.func private @sched_0(index)
  }
  module @sched_0 {
    func.func private @sched_0(%i: index) attributes {grid = [2]} {
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %6 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %7 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c4096 = arith.constant 4096 : index
      %c64000 = arith.constant 64000 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        %rows = arith.muli %i, %c4096 : index
        %base = arith.addi %c64000, %rows : index
        %v0 = ktdp.construct_memory_view %base, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc0 = memref.memory_space_cast %v0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc0 = memref.reinterpret_cast %msc0 to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %r = %c0 to %c12 step %c1 {
          scf.for %s = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %rc0[%r, %c0, %s, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1 : memref<12x1x64x64xf16, "L1">
      }
      dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%6, %7) : {
      }
      return
    }
  }
}

// -----

// The loop variable through an access-tile index, beside the grid element's own
// slab: `(tile * 6 + i * 3) * 4096` rows into a constant base. Only one root,
// and the grid element's share is a per-grid term, so this is a symbol per grid
// element over the loop variable.
// CHECK-LABEL:   func.func @test()
// CHECK:           symbol.create_id %{{.*}} {symbol_id = -1 : si64} : index
// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index) attributes {grid = [2]} {
// CHECK:           symbol.create_id %arg0 {symbol_id = -1 : si64} : index
// The first grid element's slab starts at the base, so all it adds is the
// loop variable's share.
// CHECK:           %[[WORDS_0:.*]] = arith.divsi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[WORDS_0]] {symbol_id = -6 : si64} : index
// The second's is 6 rows further on.
// CHECK:           arith.addi %{{.*}}, %c6 : index
// CHECK:           %[[WORDS_1:.*]] = arith.divsi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[WORDS_1]] {symbol_id = -12 : si64} : index
// CHECK:           %[[SYM_0:.*]] = symbol.create_symbol {SymbolId = -6 : i64} : index
// CHECK-NEXT:      %[[SYM_1:.*]] = symbol.create_symbol {SymbolId = -12 : i64} : index
// CHECK-NEXT:      %[[MAP:.*]] = uniform.def_immutable_mapping({{\[}}%{{.*}} -> %[[SYM_0]]], {{\[}}%{{.*}} -> %[[SYM_1]]]):index
// CHECK-NEXT:      %[[Q:.*]] = uniform.query_map(map:%[[MAP]], key:%{{.*}}) : index
// CHECK-NEXT:      dataflow.get_logical_memory_view %{{.*}}, %[[Q]]

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test() attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %i = %c0 to %c3 step %c1 {
        func.call @sched_0(%i) : (index) -> ()
      }
      return
    }
    func.func private @sched_0(index)
  }
  module @sched_0 {
    func.func private @sched_0(%i: index) attributes {grid = [2]} {
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %6 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %7 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c4096 = arith.constant 4096 : index
      %c64000 = arith.constant 64000 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        %tilemap = uniform.def_immutable_mapping([%0 -> %c0], [%1 -> %c1]):index
        %tile = uniform.query_map(map:%tilemap, key:%arg0) : index
        %slab = arith.muli %tile, %c6 : index
        %step = arith.muli %i, %c3 : index
        %row = arith.addi %slab, %step : index
        %off = arith.muli %row, %c4096 : index
        %v0 = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc0 = memref.memory_space_cast %v0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc0 = memref.reinterpret_cast %msc0 to offset: [%off], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %r = %c0 to %c12 step %c1 {
          scf.for %s = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %rc0[%r, %c0, %s, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1 : memref<12x1x64x64xf16, "L1">
      }
      dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%6, %7) : {
      }
      return
    }
  }
}

// -----

// Nested loops, only the inner variable passed. Every root of the kernel is
// numbered, its arguments and then its loops in pre-order -- but a root is only
// declared once a call is found to read it. The argument and the outer loop
// variable are read by no program, so they keep -1 and -2 reserved and say
// nothing, and nothing is derived from them. A loop no call sits in is numbered
// too, -4, and says nothing either.
// CHECK-LABEL:   func.func @test(
// CHECK-SAME:      %{{.*}}: index)
// CHECK-NOT:       symbol.create_id
// CHECK:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:          symbol.create_id %[[I]] {symbol_id = -3 : si64} : index
// CHECK-NEXT:          func.call @sched_0(%[[I]])
// CHECK-NOT:       symbol.create_id
// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index) attributes {grid = [2]} {
// CHECK:           symbol.create_id %arg0 {symbol_id = -3 : si64} : index
// CHECK-NEXT:      %[[ROWS:.*]] = arith.muli %arg0, %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[ROWS]] {symbol_id = -5 : si64} : index
// CHECK-NEXT:      %[[BYTES:.*]] = arith.addi %[[ROWS]], %{{.*}} : index
// CHECK-NEXT:      %[[WORDS:.*]] = arith.divsi %[[BYTES]], %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[WORDS]] {symbol_id = -6 : si64} : index
// CHECK-NOT:       symbol_id = -2
// CHECK-NOT:       symbol_id = -4 :

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index) attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      scf.for %o = %c0 to %c2 step %c1 {
        scf.for %i = %c0 to %c3 step %c1 {
          func.call @sched_0(%i) : (index) -> ()
        }
      }
      scf.for %w = %c0 to %c2 step %c1 {
        %x = arith.addi %w, %c1 : index
      }
      return
    }
    func.func private @sched_0(index)
  }
  module @sched_0 {
    func.func private @sched_0(%i: index) attributes {grid = [2]} {
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %6 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %7 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c4096 = arith.constant 4096 : index
      %c64000 = arith.constant 64000 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        %rows = arith.muli %i, %c4096 : index
        %base = arith.addi %rows, %c64000 : index
        %v0 = ktdp.construct_memory_view %base, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc0 = memref.memory_space_cast %v0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc0 = memref.reinterpret_cast %msc0 to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %r = %c0 to %c12 step %c1 {
          scf.for %s = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %rc0[%r, %c0, %s, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1 : memref<12x1x64x64xf16, "L1">
      }
      dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%6, %7) : {
      }
      return
    }
  }
}

