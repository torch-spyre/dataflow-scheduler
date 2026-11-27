// RUN: dataflow-scheduler-opt -split-input-file -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// An address computed from more than one value a symbol stands for: a base the
// run hands in, advanced by the kernel's loop variable, or two inputs combined.
// It is one symbol, defined over all of them -- a step at a time, since a
// definition is one operator over symbols and constants, and a symbol can be
// either operand of it.
//
// The kernel takes two inputs and loops once, so its arguments are -1 and -2
// and the loop variable -3; what is derived starts at -4.

// A run input advanced by the loop variable, `a + i * 128`, read in words of 64
// bytes: the rows the loop variable walks are a symbol of their own, read by the
// sum, and the sum goes unnamed into the division the view reads.
// CHECK-LABEL:   func.func @test(
// CHECK-SAME:      %[[A:[^:]+]]: index, %[[B:[^:]+]]: index)
// CHECK:           symbol.create_id %[[A]] {symbol_id = -1 : si64} : index
// CHECK-NEXT:      symbol.create_id %[[B]] {symbol_id = -2 : si64} : index
// CHECK:           scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        symbol.create_id %[[I]] {symbol_id = -3 : si64} : index
// CHECK-NEXT:        func.call @sched_0(%[[A]], %[[B]], %[[I]])
// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) attributes {grid = [2]} {
// CHECK:           symbol.create_id %arg0 {symbol_id = -1 : si64} : index
// CHECK-NEXT:      symbol.create_id %arg1 {symbol_id = -2 : si64} : index
// CHECK-NEXT:      symbol.create_id %arg2 {symbol_id = -3 : si64} : index
// CHECK-NEXT:      %[[ROWS:.*]] = arith.muli %arg2, %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[ROWS]] {symbol_id = -4 : si64} : index
// CHECK-NEXT:      %[[BYTES:.*]] = arith.addi %arg0, %[[ROWS]] : index
// CHECK-NEXT:      %[[WORDS:.*]] = arith.divsi %[[BYTES]], %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[WORDS]] {symbol_id = -5 : si64} : index
// CHECK:           symbol.create_symbol {SymbolId = -5 : i64} : index

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index, %B: index) attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %i = %c0 to %c3 step %c1 {
        func.call @sched_0(%A, %B, %i) : (index, index, index) -> ()
      }
      return
    }
    func.func private @sched_0(index, index, index)
  }
  module @sched_0 {
    func.func private @sched_0(%a: index, %b: index, %i: index) attributes {grid = [2]} {
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
        %rows = arith.muli %i, %c128 : index
        %base = arith.addi %a, %rows : index
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

// `a * 4 + b` and `a + b * 4`: the same operators over the same two inputs, and
// two different addresses. What tells them apart is which input each operand is,
// so each is a symbol of its own.
// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) attributes {grid = [2]} {
// CHECK:           %[[A4:.*]] = arith.muli %arg0, %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[A4]] {symbol_id = -4 : si64} : index
// CHECK-NEXT:      %[[X:.*]] = arith.addi %[[A4]], %arg1 : index
// CHECK-NEXT:      %[[X_WORDS:.*]] = arith.divsi %[[X]], %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[X_WORDS]] {symbol_id = -5 : si64} : index
// CHECK-NEXT:      %[[B4:.*]] = arith.muli %arg1, %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[B4]] {symbol_id = -6 : si64} : index
// CHECK-NEXT:      %[[Y:.*]] = arith.addi %arg0, %[[B4]] : index
// CHECK-NEXT:      %[[Y_WORDS:.*]] = arith.divsi %[[Y]], %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[Y_WORDS]] {symbol_id = -7 : si64} : index
// CHECK:           symbol.create_symbol {SymbolId = -5 : i64} : index
// CHECK:           symbol.create_symbol {SymbolId = -7 : i64} : index

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index, %B: index) attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %i = %c0 to %c3 step %c1 {
        func.call @sched_0(%A, %B, %i) : (index, index, index) -> ()
      }
      return
    }
    func.func private @sched_0(index, index, index)
  }
  module @sched_0 {
    func.func private @sched_0(%a: index, %b: index, %i: index) attributes {grid = [2]} {
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
        %a4 = arith.muli %a, %c4 : index
        %x = arith.addi %a4, %b : index
        %b4 = arith.muli %b, %c4 : index
        %y = arith.addi %a, %b4 : index
        %v0 = ktdp.construct_memory_view %x, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
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

// -----

// Over one root, `(i * 2) + (i * 3)` and `((i * 2) * 3) + i` are also the same
// operators in the same order, and also two different addresses.
//
// And `(t * 2) + t` for `t = a - 1` reads `t` twice. It is rebuilt before
// either of the steps reading it, so each of them reads the rebuild rather than
// the program's own `t`.
// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) attributes {grid = [2]} {
// CHECK:           %[[X_WORDS:.*]] = arith.divsi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[X_WORDS]] {symbol_id = -6 : si64} : index
// CHECK:           %[[Y_WORDS:.*]] = arith.divsi %{{.*}}, %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[Y_WORDS]] {symbol_id = -8 : si64} : index
// CHECK-NEXT:      %[[T:.*]] = arith.subi %arg0, %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[T]] {symbol_id = -9 : si64} : index
// CHECK-NEXT:      %[[T2:.*]] = arith.muli %[[T]], %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[T2]] {symbol_id = -10 : si64} : index
// CHECK-NEXT:      %[[Z:.*]] = arith.addi %[[T2]], %[[T]] : index
// CHECK-NEXT:      %[[Z_WORDS:.*]] = arith.divsi %[[Z]], %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[Z_WORDS]] {symbol_id = -11 : si64} : index
// CHECK:           symbol.create_symbol {SymbolId = -6 : i64} : index
// CHECK:           symbol.create_symbol {SymbolId = -8 : i64} : index
// CHECK:           symbol.create_symbol {SymbolId = -11 : i64} : index

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index, %B: index) attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %i = %c0 to %c3 step %c1 {
        func.call @sched_0(%A, %B, %i) : (index, index, index) -> ()
      }
      return
    }
    func.func private @sched_0(index, index, index)
  }
  module @sched_0 {
    func.func private @sched_0(%a: index, %b: index, %i: index) attributes {grid = [2]} {
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
        %i2 = arith.muli %i, %c2 : index
        %i3 = arith.muli %i, %c3 : index
        %x = arith.addi %i2, %i3 : index
        %i23 = arith.muli %i2, %c3 : index
        %y = arith.addi %i23, %i : index
        %t = arith.subi %a, %c1 : index
        %t2 = arith.muli %t, %c2 : index
        %z = arith.addi %t2, %t : index
        %v0 = ktdp.construct_memory_view %x, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc0 = memref.memory_space_cast %v0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc0 = memref.reinterpret_cast %msc0 to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %v1 = ktdp.construct_memory_view %y, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc1 = memref.memory_space_cast %v1 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc1 = memref.reinterpret_cast %msc1 to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %v2 = ktdp.construct_memory_view %z, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc2 = memref.memory_space_cast %v2 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc2 = memref.reinterpret_cast %msc2 to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %r = %c0 to %c12 step %c1 {
          scf.for %s = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %rc0[%r, %c0, %s, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
            ktdf.data_transfer from %rc1[%r, %c0, %s, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
            ktdf.data_transfer from %rc2[%r, %c0, %s, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
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

// The same address over two inputs, read in two program units: one symbol, and
// declared once.
// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) attributes {grid = [2]} {
// CHECK:           symbol.create_id %{{.*}} {symbol_id = -4 : si64} : index
// CHECK:           symbol.create_id %{{.*}} {symbol_id = -5 : si64} : index
// CHECK-NOT:       symbol.create_id
// CHECK:           symbol.create_symbol {SymbolId = -5 : i64} : index
// CHECK-NOT:       symbol.create_id
// CHECK:           symbol.create_symbol {SymbolId = -5 : i64} : index

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index, %B: index) attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %i = %c0 to %c3 step %c1 {
        func.call @sched_0(%A, %B, %i) : (index, index, index) -> ()
      }
      return
    }
    func.func private @sched_0(index, index, index)
  }
  module @sched_0 {
    func.func private @sched_0(%a: index, %b: index, %i: index) attributes {grid = [2]} {
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
        %rows = arith.muli %i, %c128 : index
        %base = arith.addi %a, %rows : index
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
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        %rows = arith.muli %i, %c128 : index
        %base = arith.addi %a, %rows : index
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

// `a + tile * i`, per grid element. The first grid element's tile is 0, so its
// share folds away and what is left is the input alone, over the word size; the
// second's is the input plus the loop variable.
// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index) attributes {grid = [2]} {
// CHECK:           symbol.create_id %arg2 {symbol_id = -3 : si64} : index
// CHECK-NEXT:      %[[WORDS_0:.*]] = arith.divsi %arg0, %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[WORDS_0]] {symbol_id = -4 : si64} : index
// CHECK-NEXT:      %[[BYTES_1:.*]] = arith.addi %arg0, %arg2 : index
// CHECK-NEXT:      %[[WORDS_1:.*]] = arith.divsi %[[BYTES_1]], %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[WORDS_1]] {symbol_id = -5 : si64} : index
// CHECK:           %[[SYM_0:.*]] = symbol.create_symbol {SymbolId = -4 : i64} : index
// CHECK-NEXT:      %[[SYM_1:.*]] = symbol.create_symbol {SymbolId = -5 : i64} : index
// CHECK-NEXT:      uniform.def_immutable_mapping({{\[}}%{{.*}} -> %[[SYM_0]]], {{\[}}%{{.*}} -> %[[SYM_1]]]):index

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index, %B: index) attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %i = %c0 to %c3 step %c1 {
        func.call @sched_0(%A, %B, %i) : (index, index, index) -> ()
      }
      return
    }
    func.func private @sched_0(index, index, index)
  }
  module @sched_0 {
    func.func private @sched_0(%a: index, %b: index, %i: index) attributes {grid = [2]} {
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
        %share = arith.muli %tile, %i : index
        %base = arith.addi %a, %share : index
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

