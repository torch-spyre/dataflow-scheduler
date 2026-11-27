// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// A kernel passing its programs values it computed itself from its roots -- its
// arguments and loop variables -- rather than the roots themselves. Each such
// value is a symbol derived from the roots, declared where the kernel computes
// it: the arithmetic is already there, next to what it reads, so nothing is
// rebuilt. A definition is one operator over symbols and constants, so a value
// computed in several steps is a symbol per step, and a step one value shares
// with another keeps the one declaration.
//
// From a program's side, its argument is then what the kernel passed, and the
// program declares it with that symbol's id, the way it would a root's.

// CHECK-LABEL:   func.func @test(
// CHECK-SAME:      %[[A:[^:]+]]: index)
// CHECK:           symbol.create_id %[[A]] {symbol_id = -1 : si64} : index
// CHECK:           scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        symbol.create_id %[[I]] {symbol_id = -2 : si64} : index
// `i * 2`, passed as it is...
// CHECK-NEXT:        %[[X:.*]] = arith.muli %[[I]], %{{.*}} : index
// CHECK-NEXT:        symbol.create_id %[[X]] {symbol_id = -3 : si64} : index
// ... and `a + i * 2`, which reads it, so -3 is its step and only the sum is new.
// CHECK-NEXT:        %[[Y:.*]] = arith.addi %[[A]], %[[X]] : index
// CHECK-NEXT:        symbol.create_id %[[Y]] {symbol_id = -4 : si64} : index
// CHECK-NEXT:        func.call @sched_0(%[[X]], %[[Y]])
// CHECK-NOT:       symbol.create_id

// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index, %{{.*}}: index) attributes {grid = [2]} {
// CHECK:           symbol.create_id %arg0 {symbol_id = -3 : si64} : index
// CHECK-NEXT:      symbol.create_id %arg1 {symbol_id = -4 : si64} : index
// What the program computes over them is derived from them in turn.
// CHECK-NEXT:      %[[XB:.*]] = arith.addi %arg0, %{{.*}} : index
// CHECK-NEXT:      %[[XB_WORDS:.*]] = arith.divsi %[[XB]], %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[XB_WORDS]] {symbol_id = -5 : si64} : index
// CHECK-NEXT:      %[[Y_WORDS:.*]] = arith.divsi %arg1, %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[Y_WORDS]] {symbol_id = -6 : si64} : index
// CHECK:           symbol.create_symbol {SymbolId = -5 : i64} : index
// CHECK:           symbol.create_symbol {SymbolId = -6 : i64} : index

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index) attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      scf.for %i = %c0 to %c3 step %c1 {
        %x = arith.muli %i, %c2 : index
        %y = arith.addi %A, %x : index
        func.call @sched_0(%x, %y) : (index, index) -> ()
      }
      return
    }
    func.func private @sched_0(index, index)
  }
  module @sched_0 {
    func.func private @sched_0(%x: index, %y: index) attributes {grid = [2]} {
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
        %xb = arith.addi %x, %c64000 : index
        %v0 = ktdp.construct_memory_view %xb, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc0 = memref.memory_space_cast %v0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc0 = memref.reinterpret_cast %msc0 to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %v1 = ktdp.construct_memory_view %y, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc1 = memref.memory_space_cast %v1 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc1 = memref.reinterpret_cast %msc1 to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %r = %c0 to %c12 step %c1 {
          scf.for %s = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %rc0[%r, %c0, %s, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
            ktdf.data_transfer from %rc1[%r, %c0, %s, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
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

