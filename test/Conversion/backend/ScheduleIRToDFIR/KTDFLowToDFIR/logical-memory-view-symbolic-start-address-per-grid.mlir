// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// A start address that varies with the grid element: each core reads its own slab
// of the tensor, at the input's address plus its share. So there is no one
// address for the program and no one symbol -- there is a symbol per core, and
// the start address the view reads is the choice between them.
//
// The core whose slab starts at the tensor's own base adds nothing, so its
// address *is* the input and the input's own symbol stands for it. Only the
// others are symbols of their own, and what each is computed from is said as
// arithmetic beside the program: nothing reads those values, they are there so
// that whatever resolves the symbols can read what each one is.

// CHECK-LABEL:   func.func @test(
// CHECK-SAME:      %[[BASE:.*]]: index)
// CHECK-NEXT:      symbol.create_id %[[BASE]] {symbol_id = -1 : si64} : index

// The definition of the second core's address, at function level -- outside the
// region that is one core's program, because symbols belong to the run rather
// than to any one core.
// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index) attributes {grid = [2]} {
// CHECK:           %[[C24576:.*]] = arith.constant 24576 : index
// CHECK:           %[[DDR:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"}
// CHECK:           symbol.create_id %arg0 {symbol_id = -1 : si64} : index
// CHECK-NEXT:      %[[DERIVED:.*]] = arith.addi %arg0, %[[C24576]] : index
// CHECK-NEXT:      symbol.create_id %[[DERIVED]] {symbol_id = -2 : si64} : index

// And in the program, a symbol per core gathered into a map keyed on the unit:
// core 0 reads the input's own symbol, core 1 the derived one.
// CHECK:           %[[SYM_0:.*]] = symbol.create_symbol {SymbolId = -1 : i64} : index
// CHECK-NEXT:      %[[SYM_1:.*]] = symbol.create_symbol {SymbolId = -2 : i64} : index
// CHECK-NEXT:      %[[MAP:.*]] = uniform.def_immutable_mapping({{\[}}%{{.*}} -> %[[SYM_0]]], {{\[}}%{{.*}} -> %[[SYM_1]]]):index
// CHECK-NEXT:      %[[Q:.*]] = uniform.query_map(map:%[[MAP]], key:%{{.*}}) : index
// CHECK-NEXT:      dataflow.get_logical_memory_view %[[DDR]], %[[Q]]

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%base_0: index) attributes {grid = [2]} {
      call @sched_0(%base_0) : (index) -> ()
      return
    }
    func.func private @sched_0(index)
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
      %c24576 = arith.constant 24576 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        // What ktdp.get_compute_tile_id has become by this point: one constant
        // per core, chosen by the unit the program is running on.
        %tilemap = uniform.def_immutable_mapping([%0 -> %c0], [%1 -> %c1]):index
        %tile = uniform.query_map(map:%tilemap, key:%arg0) : index
        %shift = arith.muli %tile, %c24576 : index
        %shifted = arith.addi %base, %shift : index
        %10 = ktdp.construct_memory_view %shifted, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
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
}
