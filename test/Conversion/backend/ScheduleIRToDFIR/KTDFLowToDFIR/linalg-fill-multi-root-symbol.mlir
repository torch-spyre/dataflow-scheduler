// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// A register filled with a value computed from two symbols: a run input and the
// kernel's loop variable. The fill carries one symbol, defined a step at a time
// over both, next to the program.

// CHECK-LABEL:   func.func @test(
// CHECK-SAME:      %[[A:[^:]+]]: index)
// CHECK:           symbol.create_id %[[A]] {symbol_id = -1 : si64} : index
// CHECK:           scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:        symbol.create_id %[[I]] {symbol_id = -2 : si64} : index

// CHECK-LABEL:   func.func private @sched_0(
// CHECK:           vectorchain.constant_bitstream {is_symbol = true, value = {{\[}}[[ID:-[0-9]+]]]} : vector<1xf16>
// CHECK:         symbol.create_id %arg0 {symbol_id = -1 : si64} : index
// CHECK-NEXT:    symbol.create_id %arg1 {symbol_id = -2 : si64} : index
// CHECK-NEXT:    %[[SUM:.+]] = arith.addi %arg0, %arg1 : index
// CHECK-NEXT:    symbol.create_id %[[SUM]] {symbol_id = -3 : si64} : index
// CHECK-NEXT:    %[[NARROW:.+]] = arith.index_castui %[[SUM]] : index to i32
// CHECK-NEXT:    symbol.create_id %[[NARROW]] {symbol_id = -4 : si64} : i32
// CHECK-NEXT:    %[[WIDE:.+]] = arith.uitofp %[[NARROW]] : i32 to f16
// CHECK-NEXT:    symbol.create_id %[[WIDE]] {symbol_id = [[ID]] : si64} : f16

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index) attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %i = %c0 to %c3 step %c1 {
        func.call @sched_0(%A, %i) : (index, index) -> ()
      }
      return
    }
    func.func private @sched_0(index, index)
  }
  module @sched_0 {
    func.func private @sched_0(%a: index, %i: index) attributes {grid = [2]} {
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %tile_id = ktdp.get_compute_tile_id : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
      %unit = uniform.query_map(map:%map, key:%tile_id) : index
      ktdf_lowering.execute_on %unit {
        %alloc = memref.alloc() : memref<1x64xf16, "SFU_REG">
        %sum = arith.addi %a, %i : index
        %narrow = arith.index_castui %sum : index to i32
        %wide = arith.uitofp %narrow : i32 to f16
        linalg.fill ins(%wide : f16) outs(%alloc : memref<1x64xf16, "SFU_REG">)
      }
      return
    }
  }
}
