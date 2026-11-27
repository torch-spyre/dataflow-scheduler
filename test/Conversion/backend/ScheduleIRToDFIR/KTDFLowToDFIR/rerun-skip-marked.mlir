// RUN: dataflow-scheduler-opt --ktdflowering-to-dfir %s | FileCheck %s

// A re-run, as ktdflowering-to-dfir finds it: @old is a program
// an earlier run scheduled and marked, already DataflowIR with its five
// program units; @new is the same program still in ktdf_lowering form.
//
// The pass leaves @old as it is. Without the skip it would wrap every one of
// @old's program units in another, five becoming thirty. It lowers @new as on
// a first run, to five program units, none nested in another.

// CHECK-LABEL: module @old attributes {dataflow_scheduler.scheduled}
// CHECK:         func.func @old_body()
// CHECK-COUNT-5:   dataflow.program_unit
// CHECK-NOT:       dataflow.program_unit
// CHECK-LABEL: module @new
// CHECK-NOT:     ktdf_lowering.
// CHECK-COUNT-5: dataflow.program_unit
// CHECK-NOT:     dataflow.program_unit
// CHECK-NOT:     ktdf_lowering.

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @kernel(%arg0: index) {
      symbol.create_id %arg0 {symbol_id = -1 : si64} : index
      call @old(%arg0) : (index) -> ()
      call @new(%arg0) : (index) -> ()
      return
    }
    func.func private @old(index)
    func.func private @new(index)
  }
  module @old attributes {dataflow_scheduler.scheduled} {
    func.func private @old_body()
    func.func @old(%arg0: index) {
      %c2 = arith.constant 2 : index
      %c1024 = arith.constant 1024 : index
      %c3 = arith.constant 3 : index
      %c12288 = arith.constant 12288 : index
      %c18432 = arith.constant 18432 : index
      %c64 = arith.constant 64 : index
      symbol.create_id %arg0 {symbol_id = -1 : si64} : index
      %0 = arith.muli %arg0, %c64 : index
      symbol.create_id %0 {symbol_id = -2 : si64} : index
      %1 = arith.addi %0, %c1024 : index
      symbol.create_id %1 {symbol_id = -3 : si64} : index
      %2 = arith.muli %1, %c2 : index
      symbol.create_id %2 {symbol_id = -4 : si64} : index
      %3 = arith.divsi %2, %c64 : index
      symbol.create_id %3 {symbol_id = -5 : si64} : index
      %4 = arith.addi %arg0, %c3 : index
      symbol.create_id %4 {symbol_id = -6 : si64} : index
      %5 = arith.muli %4, %c64 : index
      symbol.create_id %5 {symbol_id = -7 : si64} : index
      %6 = arith.addi %5, %c1024 : index
      symbol.create_id %6 {symbol_id = -8 : si64} : index
      %7 = arith.muli %6, %c2 : index
      symbol.create_id %7 {symbol_id = -9 : si64} : index
      %8 = arith.divsi %7, %c64 : index
      symbol.create_id %8 {symbol_id = -10 : si64} : index
      %9 = arith.muli %arg0, %c64 : index
      symbol.create_id %9 {symbol_id = -2 : si64} : index
      %10 = arith.addi %9, %c12288 : index
      symbol.create_id %10 {symbol_id = -11 : si64} : index
      %11 = arith.muli %10, %c2 : index
      symbol.create_id %11 {symbol_id = -12 : si64} : index
      %12 = arith.divsi %11, %c64 : index
      symbol.create_id %12 {symbol_id = -13 : si64} : index
      %13 = arith.addi %arg0, %c3 : index
      symbol.create_id %13 {symbol_id = -6 : si64} : index
      %14 = arith.muli %13, %c64 : index
      symbol.create_id %14 {symbol_id = -7 : si64} : index
      %15 = arith.addi %14, %c12288 : index
      symbol.create_id %15 {symbol_id = -14 : si64} : index
      %16 = arith.muli %15, %c2 : index
      symbol.create_id %16 {symbol_id = -15 : si64} : index
      %17 = arith.divsi %16, %c64 : index
      symbol.create_id %17 {symbol_id = -16 : si64} : index
      %18 = arith.muli %arg0, %c64 : index
      symbol.create_id %18 {symbol_id = -2 : si64} : index
      %19 = arith.addi %18, %c18432 : index
      symbol.create_id %19 {symbol_id = -17 : si64} : index
      %20 = arith.muli %19, %c2 : index
      symbol.create_id %20 {symbol_id = -18 : si64} : index
      %21 = arith.divsi %20, %c64 : index
      symbol.create_id %21 {symbol_id = -19 : si64} : index
      %22 = arith.addi %arg0, %c3 : index
      symbol.create_id %22 {symbol_id = -6 : si64} : index
      %23 = arith.muli %22, %c64 : index
      symbol.create_id %23 {symbol_id = -7 : si64} : index
      %24 = arith.addi %23, %c18432 : index
      symbol.create_id %24 {symbol_id = -20 : si64} : index
      %25 = arith.muli %24, %c2 : index
      symbol.create_id %25 {symbol_id = -21 : si64} : index
      %26 = arith.divsi %25, %c64 : index
      symbol.create_id %26 {symbol_id = -22 : si64} : index
      call @old_body() : () -> ()
      return
    }
    module {
      func.func @old_body() attributes {grid = [2]} {
        %c128 = arith.constant 128 : index
        %c64 = arith.constant 64 : index
        %c0 = arith.constant 0 : index
        %0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnilu", type = "mnilu"} : index
        %1 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
        %2 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
        %3 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1lu", type = "l1lu"} : index
        %4 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
        %5 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-sfu", type = "sfu"} : index
        %6 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
        %7 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1su", type = "l1su"} : index
        %8 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
        %9 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnisu", type = "mnisu"} : index
        %10 = dataflow.get_unit {name = "ddr", type = "ddr"} : index
        %11 = dataflow.get_unit {core = 0 : i32, name = "C0-l1", type = "l1"} : index
        %12 = dataflow.get_unit {core = 1 : i32, name = "C1-l1", type = "l1"} : index
        dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
          %13 = uniform.def_immutable_mapping([%0 -> %11], [%1 -> %12]):index
          %14 = uniform.query_map(map:%13, key:%arg0) : index
          %15 = symbol.create_symbol {SymbolId = -5 : i64} : index
          %16 = symbol.create_symbol {SymbolId = -10 : i64} : index
          %17 = uniform.def_immutable_mapping([%0 -> %15], [%1 -> %16]):index
          %18 = uniform.query_map(map:%17, key:%arg0) : index
          %19 = dataflow.get_logical_memory_view %10, %18 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<96x64xf16>
          %20 = symbol.create_symbol {SymbolId = -13 : i64} : index
          %21 = symbol.create_symbol {SymbolId = -16 : i64} : index
          %22 = uniform.def_immutable_mapping([%0 -> %20], [%1 -> %21]):index
          %23 = uniform.query_map(map:%22, key:%arg0) : index
          %24 = dataflow.get_logical_memory_view %10, %23 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<96x64xf16>
          %25 = dataflow.get_logical_memory_view %14, %c0 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<1x64xf16>
          %26 = dataflow.get_logical_memory_view %14, %c64 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<1x64xf16>
          agen.composite_load_and_store src:%19[0, %c0 * 64] dst:%25[0, 0]
           time_symbols(), load_iv(%arg1:vector<64xf16>)
           {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, load_time_addr_map = affine_map<(d0) -> (0, 0)>, store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, store_time_addr_map = affine_map<(d0) -> (0, 0)>, time_order = affine_map<(d0) -> (d0)>, time_set = affine_set<(d0) : (d0 == 0)>}
          {
            agen.yield
          } : memref<96x64xf16>, memref<1x64xf16>
          agen.composite_load_and_store src:%24[0, %c0 * 64] dst:%26[0, 0]
           time_symbols(), load_iv(%arg1:vector<64xf16>)
           {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, load_time_addr_map = affine_map<(d0) -> (0, 0)>, store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, store_time_addr_map = affine_map<(d0) -> (0, 0)>, time_order = affine_map<(d0) -> (d0)>, time_set = affine_set<(d0) : (d0 == 0)>}
          {
            agen.yield
          } : memref<96x64xf16>, memref<1x64xf16>
          %27 = uniform.def_immutable_mapping([%0 -> %2], [%1 -> %3]):index
          %28 = uniform.query_map(map:%27, key:%arg0) : index
          dataflow.sync_send %28 {wait_immediately_for_async_transfers = true} : index
          dataflow.sync_recv %28 : index
        }
        dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
          %13 = uniform.def_immutable_mapping([%2 -> %11], [%3 -> %12]):index
          %14 = uniform.query_map(map:%13, key:%arg0) : index
          %15 = dataflow.get_logical_memory_view %14, %c0 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<1x64xf16>
          %16 = dataflow.get_logical_memory_view %14, %c64 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<1x64xf16>
          %17 = uniform.def_immutable_mapping([%2 -> %0], [%3 -> %1]):index
          %18 = uniform.query_map(map:%17, key:%arg0) : index
          dataflow.sync_send %18 {wait_immediately_for_async_transfers = true} : index
          dataflow.sync_recv %18 : index
          %19 = agen.vector_load %15[0, 0] {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<1x64xf16>, vector<64xf16>
          %20 = uniform.def_immutable_mapping([%2 -> %4], [%3 -> %5]):index
          %21 = uniform.query_map(map:%20, key:%arg0) : index
          dataflow.send %21, %19 : vector<64xf16>
          %22 = agen.vector_load %16[0, 0] {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<1x64xf16>, vector<64xf16>
          %23 = uniform.def_immutable_mapping([%2 -> %4], [%3 -> %5]):index
          %24 = uniform.query_map(map:%23, key:%arg0) : index
          dataflow.send %24, %22 : vector<64xf16>
        }
        dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
          %13 = uniform.def_immutable_mapping([%4 -> %2], [%5 -> %3]):index
          %14 = uniform.query_map(map:%13, key:%arg0) : index
          %15 = dataflow.receive %14 : vector<64xf16>
          %16 = uniform.def_immutable_mapping([%4 -> %2], [%5 -> %3]):index
          %17 = uniform.query_map(map:%16, key:%arg0) : index
          %18 = dataflow.receive %17 : vector<64xf16>
          %19 = vectorchain.binary %15, %18 {binary_op = #vectorchain<binary_operator add>, op_specific_map = affine_map<(d0) -> (d0)>} : vector<64xf16>, vector<64xf16>, vector<64xf16>
          %20 = uniform.def_immutable_mapping([%4 -> %6], [%5 -> %7]):index
          %21 = uniform.query_map(map:%20, key:%arg0) : index
          dataflow.send %21, %19 : vector<64xf16>
        }
        dataflow.program_unit iter_arg : %arg0 -> (%6, %7) : {
          %13 = uniform.def_immutable_mapping([%6 -> %11], [%7 -> %12]):index
          %14 = uniform.query_map(map:%13, key:%arg0) : index
          %15 = dataflow.get_logical_memory_view %14, %c128 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<1x64xf16>
          %16 = uniform.def_immutable_mapping([%6 -> %4], [%7 -> %5]):index
          %17 = uniform.query_map(map:%16, key:%arg0) : index
          %18 = dataflow.receive %17 : vector<64xf16>
          agen.vector_store %18, %15[0, 0] {store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<1x64xf16>, vector<64xf16>
          %19 = uniform.def_immutable_mapping([%6 -> %8], [%7 -> %9]):index
          %20 = uniform.query_map(map:%19, key:%arg0) : index
          dataflow.sync_send %20 {wait_immediately_for_async_transfers = true} : index
          dataflow.sync_recv %20 : index
        }
        dataflow.program_unit iter_arg : %arg0 -> (%8, %9) : {
          %13 = uniform.def_immutable_mapping([%8 -> %11], [%9 -> %12]):index
          %14 = uniform.query_map(map:%13, key:%arg0) : index
          %15 = symbol.create_symbol {SymbolId = -19 : i64} : index
          %16 = symbol.create_symbol {SymbolId = -22 : i64} : index
          %17 = uniform.def_immutable_mapping([%8 -> %15], [%9 -> %16]):index
          %18 = uniform.query_map(map:%17, key:%arg0) : index
          %19 = dataflow.get_logical_memory_view %10, %18 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<96x64xf16>
          %20 = dataflow.get_logical_memory_view %14, %c128 {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<1x64xf16>
          %21 = uniform.def_immutable_mapping([%8 -> %6], [%9 -> %7]):index
          %22 = uniform.query_map(map:%21, key:%arg0) : index
          dataflow.sync_send %22 {wait_immediately_for_async_transfers = true} : index
          dataflow.sync_recv %22 : index
          agen.composite_load_and_store src:%20[0, 0] dst:%19[0, %c0 * 64]
           time_symbols(), load_iv(%arg1:vector<64xf16>)
           {load_order = affine_map<(d0, d1) -> (d0, d1)>, load_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, load_time_addr_map = affine_map<(d0) -> (0, 0)>, store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>, store_time_addr_map = affine_map<(d0) -> (0, 0)>, time_order = affine_map<(d0) -> (d0)>, time_set = affine_set<(d0) : (d0 == 0)>}
          {
            agen.yield
          } : memref<1x64xf16>, memref<96x64xf16>
        }
        return
      }
    }
  }
  module @new {
    func.func @new(%arg0: index) attributes {grid = [2]} {
      %c128 = arith.constant 128 : index
      %c1024 = arith.constant 1024 : index
      %c3 = arith.constant 3 : index
      %c12288 = arith.constant 12288 : index
      %c18432 = arith.constant 18432 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnilu", type = "mnilu"} : index
      %1 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
      %2 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
      %3 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1lu", type = "l1lu"} : index
      %4 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
      %5 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-sfu", type = "sfu"} : index
      %6 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
      %7 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1su", type = "l1su"} : index
      %8 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
      %9 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnisu", type = "mnisu"} : index
      %10 = ktdp.get_compute_tile_id : index
      %11 = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
      %12 = uniform.query_map(map:%11, key:%10) : index
      %13 = uniform.def_immutable_mapping([%c0 -> %2], [%c1 -> %3]):index
      %14 = uniform.query_map(map:%13, key:%10) : index
      %15 = uniform.def_immutable_mapping([%c0 -> %4], [%c1 -> %5]):index
      %16 = uniform.query_map(map:%15, key:%10) : index
      %17 = uniform.def_immutable_mapping([%c0 -> %6], [%c1 -> %7]):index
      %18 = uniform.query_map(map:%17, key:%10) : index
      %19 = uniform.def_immutable_mapping([%c0 -> %8], [%c1 -> %9]):index
      %20 = uniform.query_map(map:%19, key:%10) : index
      %21 = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
      %22 = ktdp.get_compute_tile_id : index
      %23 = arith.muli %22, %c3 : index
      %24 = arith.addi %23, %arg0 : index
      %25 = ktdp.construct_memory_view %c12288, sizes: [96, 64], strides: [64, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
      %26 = ktdp.construct_memory_view %c18432, sizes: [96, 64], strides: [64, 1] {coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
      %27 = arith.muli %24, %c64 : index
      %memspacecast = memref.memory_space_cast %21 : memref<96x64xf16> to memref<96x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [%27], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
      %28 = arith.muli %24, %c64 : index
      %memspacecast_0 = memref.memory_space_cast %25 : memref<96x64xf16> to memref<96x64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [%28], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
      %29 = tensor.empty() : tensor<64xf16>
      %30 = arith.muli %24, %c64 : index
      %memspacecast_2 = memref.memory_space_cast %26 : memref<96x64xf16> to memref<96x64xf16, "DDR">
      %reinterpret_cast_3 = memref.reinterpret_cast %memspacecast_2 to offset: [%30], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
      ktdf_lowering.execute_on %12, %14, %16, %18, %20 {
        %31 = builtin.unrealized_conversion_cast %c0 : index to memref<1x64xf16, "L1">
        %32 = builtin.unrealized_conversion_cast %c64 : index to memref<1x64xf16, "L1">
        %33:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        %34 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
        %35 = builtin.unrealized_conversion_cast %c128 : index to memref<1x64xf16, "L1">
        ktdf_lowering.execute_on %12 {
          ktdf.data_transfer from %reinterpret_cast[0, %c0 * 64] size [1, 64] to %31[0, 0] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
          ktdf.data_transfer from %reinterpret_cast_1[0, %c0 * 64] size [1, 64] to %32[0, 0] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
        }
        ktdf_lowering.signal %12, %14
        ktdf_lowering.execute_on %14 {
          ktdf.data_transfer from %31[0, 0] size [1, 64] to %33#0 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
          ktdf.data_transfer from %32[0, 0] size [1, 64] to %33#1 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        }
        ktdf_lowering.execute_on %16 {
          %36 = ktdf.read_from_fifo %33#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
          %37 = ktdf.read_from_fifo %33#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
          %38 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%36, %37 : tensor<64xf16>, tensor<64xf16>) outs(%29 : tensor<64xf16>) attrs =  {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
          ^bb0(%in: f16, %in_4: f16, %out: f16):
            %39 = arith.addf %in, %in_4 : f16
            linalg.yield %39 : f16
          } -> tensor<64xf16>
          ktdf.write_to_fifo %38, %34 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
        }
        ktdf_lowering.execute_on %18 {
          ktdf.data_transfer from %34 size [64] to %35[0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
        }
        ktdf_lowering.signal %18, %20
        ktdf_lowering.execute_on %20 {
          ktdf.data_transfer from %35[0, 0] size [1, 64] to %reinterpret_cast_3[0, %c0 * 64] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, "L1">, memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
        }
      }
      return
    }
  }
}
