// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// The same address per grid element as
// logical-memory-view-symbolic-start-address-per-grid, arrived at from the other
// side: the view is of the whole tensor and sits at the input itself, and what
// each core reads is a tile *within* that view -- its own slab, at an offset the
// reinterpret_cast carries. Which is what a KTIR access tile split along the
// outer dimension becomes.
//
// So the address a core reads is the input plus an offset that differs per core,
// which gives a symbol per core exactly as arithmetic on the view's own address
// would have. The offset becomes a term of each symbol's definition rather than
// arithmetic left at the view, because the run has to write over the whole
// operand.

// CHECK-LABEL:   func.func @test(
// CHECK-SAME:      %[[BASE:.*]]: index)
// CHECK-NEXT:      symbol.create_id %[[BASE]] {symbol_id = -1 : si64} : index

// The second core's address, defined at function level: the input plus the row
// its tile starts at. Nothing reads it; it is there so that whatever resolves
// the symbols can read the definition of -3.
//
// In bytes, because the input is a byte address: the reinterpret_cast offset
// counts elements -- six rows of 4096 is 24576 elements -- and an f16 is two
// bytes wide, so the distance is 24576 * 2 = 49152 bytes.
// CHECK-LABEL:   func.func private @sched_0(%{{.*}}: index) attributes {grid = [2]} {
// CHECK:           %[[C49152:.*]] = arith.constant 49152 : index
// CHECK:           %[[DDR:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"}
// CHECK:           symbol.create_id %arg0 {symbol_id = -1 : si64} : index
// The first core's address in words, and the second core's: the input, plus that
// core's share of the tensor where there is one, over the unit's word size.
// CHECK-NEXT:      %[[WORDS_0:.*]] = arith.divsi %arg0, %[[C64:.*]] : index
// CHECK-NEXT:      symbol.create_id %[[WORDS_0]] {symbol_id = -2 : si64} : index
// CHECK-NEXT:      %[[SHIFTED:.*]] = arith.addi %arg0, %[[C49152]] : index
// CHECK-NEXT:      %[[WORDS_1:.*]] = arith.divsi %[[SHIFTED]], %[[C64]] : index
// CHECK-NEXT:      symbol.create_id %[[WORDS_1]] {symbol_id = -3 : si64} : index

// And in the program, a symbol per core gathered into a map keyed on the unit:
// core 0's tile starts at the tensor's own base and core 1's a slab in, and each
// reads the symbol its own address in words is. Neither the offset arithmetic nor the
// reinterpret_cast it fed survives -- the whole start address is the symbol.
// CHECK:           %[[SYM_0:.*]] = symbol.create_symbol {SymbolId = -2 : i64} : index
// CHECK-NEXT:      %[[SYM_1:.*]] = symbol.create_symbol {SymbolId = -3 : i64} : index
// CHECK-NEXT:      %[[MAP:.*]] = uniform.def_immutable_mapping({{\[}}%{{.*}} -> %[[SYM_0]]], {{\[}}%{{.*}} -> %[[SYM_1]]]):index
// CHECK-NEXT:      %[[Q:.*]] = uniform.query_map(map:%[[MAP]], key:%{{.*}}) : index
// CHECK-NEXT:      dataflow.get_logical_memory_view %[[DDR]], %[[Q]]
// CHECK-NOT:       memref.reinterpret_cast

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
        // The tile's offset into the view, as lowering an access tile leaves it:
        // the core's own row times the outer stride.
        %shift = arith.muli %tile, %c24576 : index
        // The view itself is of the whole tensor, at the input and nothing else.
        %10 = ktdp.construct_memory_view %base, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc = memref.memory_space_cast %10 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc = memref.reinterpret_cast %msc to offset: [%shift], sizes: [6, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<6x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<6x1x64x64xf16, "L1">
        scf.for %arg1 = %c0 to %c12 step %c1 {
          scf.for %arg2 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %rc[%arg1, %c0, %arg2, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] : memref<6x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<6x1x64x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1 : memref<6x1x64x64xf16, "L1">
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
