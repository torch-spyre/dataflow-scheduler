// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// A tensor's start address that arrives as an input to the kernel rather than as
// a constant. The kernel takes two of them and hands one to each of its two
// programs, so each program reads its base as its own first argument -- and the
// two mean different tensors, which is what makes the kernel's signature, not
// the program's, the thing that says which symbol an address is.
//
// What the lowering emits is symbol.create_symbol carrying that id, standing
// alone as the view's start address: the symbol is a placeholder whatever runs
// the program writes over, so it has to be the whole operand.
//
// The unit addresses in words of 64 bytes and the run hands the address over in
// bytes, so the symbol a program reads is a symbol of its own, defined as the
// input divided by that. The input keeps its own id, which is what the host
// resolves the division from.
//
// The second program is the mixed case: one base an input, one a constant. A
// constant base is untouched -- there is nothing symbolic about it.

// The kernel and the calls are as they were: what an address is stays said in
// the signature, and the argument is left in place even where the program body
// no longer reads it -- dropping it would mean rewriting every call as well,
// and it is still what says the program is about that input.
// CHECK-LABEL:   func.func @test(
// CHECK-SAME:      %[[BASE_0:.*]]: index, %[[BASE_1:.*]]: index)
// The kernel's signature is what says which symbol each input is: its arguments
// in order, so the first is -1 and the second -2. Everything downstream reads
// this rather than working the numbering out again.
// CHECK-NEXT:      symbol.create_id %[[BASE_0]] {symbol_id = -1 : si64} : index
// CHECK-NEXT:      symbol.create_id %[[BASE_1]] {symbol_id = -2 : si64} : index
// CHECK-NEXT:      call @sched_0(%[[BASE_0]])
// CHECK-NEXT:      call @sched_1(%[[BASE_1]])

// The kernel's first input, so symbol -1, and the whole start address of the
// view rather than a term in it.
// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index) attributes {grid = [2]} {
// CHECK:           %[[DDR_0:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"}
// The program says which input its own argument is, so what corrects it can be
// written against what it declares rather than against the kernel's signature.
// CHECK:           symbol.create_id %arg0 {symbol_id = -1 : si64} : index
// The address in words, which is what the view reads: symbol -3, defined as the
// input over the unit's word size.
// CHECK-NEXT:      %[[WORDS_0:.*]] = arith.divsi %arg0, %[[C64:.*]] : index
// CHECK-NEXT:      symbol.create_id %[[WORDS_0]] {symbol_id = -3 : si64} : index
// CHECK:           %[[SYM_0:.*]] = symbol.create_symbol {SymbolId = -3 : i64} : index
// CHECK-NEXT:      dataflow.get_logical_memory_view %[[DDR_0]], %[[SYM_0]]
// CHECK-NOT:       @sched_1
// CHECK-NOT:       SymbolId = -2

// The kernel's second input is a different tensor, so a different symbol -- and
// it is the position in the kernel's signature that says so, not the position in
// this program's, where both are argument 0.
// CHECK-LABEL:   func.func private @sched_1(%{{.*}}: index) attributes {grid = [2]} {
// CHECK:           %[[C64000:.*]] = arith.constant 64000 : index
// CHECK:           %[[DDR_1:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"}
// CHECK:           symbol.create_id %arg0 {symbol_id = -2 : si64} : index
// CHECK-NEXT:      %[[WORDS_1:.*]] = arith.divsi %arg0, %{{.*}} : index
// CHECK-NEXT:      symbol.create_id %[[WORDS_1]] {symbol_id = -4 : si64} : index
// CHECK:           %[[SYM_1:.*]] = symbol.create_symbol {SymbolId = -4 : i64} : index
// CHECK-NEXT:      dataflow.get_logical_memory_view %[[DDR_1]], %[[SYM_1]]
// The constant base beside it is still a constant.
// CHECK-NEXT:      dataflow.get_logical_memory_view %[[DDR_1]], %[[C64000]]

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%base_0: index, %base_1: index) attributes {grid = [2]} {
      call @sched_0(%base_0) : (index) -> ()
      call @sched_1(%base_1) : (index) -> ()
      return
    }
    func.func private @sched_0(index)
    func.func private @sched_1(index)
  }
  module {
    func.func private @sched_0(%base: index) attributes {grid = [2]} {
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
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        %10 = ktdp.construct_memory_view %base, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc = memref.memory_space_cast %10 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc = memref.reinterpret_cast %msc to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
        %cast = memref.cast %rc : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %arg1 = %c0 to %c12 step %c1 {
          scf.for %arg2 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %cast[%arg1, %c0, %arg2, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
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
  module {
    func.func private @sched_1(%base: index) attributes {grid = [2]} {
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
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c64000 = arith.constant 64000 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        // The input base ...
        %10 = ktdp.construct_memory_view %base, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc = memref.memory_space_cast %10 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc = memref.reinterpret_cast %msc to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
        %cast = memref.cast %rc : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        // ... and a constant one beside it.
        %20 = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc2 = memref.memory_space_cast %20 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc2 = memref.reinterpret_cast %msc2 to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
        %cast2 = memref.cast %rc2 : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %arg1 = %c0 to %c12 step %c1 {
          scf.for %arg2 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %cast[%arg1, %c0, %arg2, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
            ktdf.data_transfer from %cast2[%arg1, %c0, %arg2, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
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
