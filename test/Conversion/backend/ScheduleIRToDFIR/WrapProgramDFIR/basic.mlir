// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(wrap-program-dfir)" %s | FileCheck %s

// A program of the run read two ways: what it is, and what it is compiled from.
//
// What it is keeps the program's own name and signature -- which of the run's
// inputs it is about -- and says which symbol each of those is, and what any
// address derived from one is computed from. What it is compiled from goes in a
// module of its own, in a function taking no arguments, because an argument is a
// value only a caller has and there is no caller on a device.
//
// So the symbol declarations move out of the DataflowIR and the arguments go with
// them, which is what makes the inner function something code generation can be
// handed.

// The declaration module is untouched: it holds the kernel, not a program.
// CHECK:      func.func @kernel(%[[ARG:.*]]: index)
// CHECK-NEXT:   call @prog(%[[ARG]])

// CHECK:      module @prog {
// CHECK-NEXT:   func.func private @prog_body()
// CHECK-NEXT:   func.func @prog(%[[BASE:.*]]: index) {
// CHECK-NEXT:     %[[C4096:.*]] = arith.constant 4096 : index
// CHECK-NEXT:     symbol.create_id %[[BASE]] {symbol_id = -1 : si64} : index
// CHECK-NEXT:     %[[DERIVED:.*]] = arith.addi %[[BASE]], %[[C4096]] : index
// CHECK-NEXT:     symbol.create_id %[[DERIVED]] {symbol_id = -2 : si64} : index
// CHECK-NEXT:     call @prog_body() : () -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT:   module {
// CHECK-NEXT:     func.func @prog_body() attributes {grid = [2]} {
// The DataflowIR keeps only what it can be compiled from: the symbols it reads
// are placeholders, not the argument, so nothing here needed one.
// CHECK-NOT:        symbol.create_id
// CHECK:            symbol.create_symbol {SymbolId = -1 : i64} : index
// CHECK:            symbol.create_symbol {SymbolId = -2 : i64} : index

module {
  module {
    func.func @kernel(%base: index) {
      call @prog(%base) : (index) -> ()
      return
    }
    func.func private @prog(index)
  }
  module @prog {
    func.func @prog(%base: index) attributes {grid = [2]} {
      %c4096 = arith.constant 4096 : index
      symbol.create_id %base {symbol_id = -1 : si64} : index
      %derived = arith.addi %base, %c4096 : index
      symbol.create_id %derived {symbol_id = -2 : si64} : index
      %unit = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      dataflow.program_unit iter_arg : %arg0 -> (%unit) : {
        %sym0 = symbol.create_symbol {SymbolId = -1 : i64} : index
        %sym1 = symbol.create_symbol {SymbolId = -2 : i64} : index
        %hbm = dataflow.get_unit {name = "hbm", type = "hbm"} : index
        %v0 = dataflow.get_logical_memory_view %hbm, %sym0 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
        %v1 = dataflow.get_logical_memory_view %hbm, %sym1 {layout_map = affine_map<(d0) -> (d0)>} : index, index, memref<64xf16>
      }
      return
    }
  }
}
