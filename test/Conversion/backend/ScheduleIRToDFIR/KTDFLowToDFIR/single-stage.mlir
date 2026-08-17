// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (0, 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK-LABEL:   func.func @single_stage_pipeline(
// CHECK-SAME:      %[[ARG0:.*]]: index) attributes {grid = [2]} {
// CHECK-NEXT:     %[[C3:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[C1024:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[C64:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[MNILU_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[MNILU_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[ARG_MNILU:.*]] -> (%[[MNILU_0]], %[[MNILU_1]]) : {
// The argument is one of the run's inputs, so which symbol it is gets said here.
// This tensor's own address is a constant, so nothing about the view is symbolic
// -- the input only displaces within it, which stays plain arithmetic.
// CHECK-NEXT:       symbol.create_id %[[ARG0]] {symbol_id = -1 : si64} : index
// CHECK-NEXT:       %[[MAP_CORE:.*]] = uniform.def_immutable_mapping({{\[}}%[[MNILU_0]] -> %[[C0]]], {{\[}}%[[MNILU_1]] -> %[[C3]]]):index
// CHECK-NEXT:       %[[CORE_IDX:.*]] = uniform.query_map(map:%[[MAP_CORE]], key:%[[ARG_MNILU]]) : index
// CHECK-NEXT:       %[[ADDI_0:.*]] = arith.addi %[[CORE_IDX]], %[[ARG0]] : index
// CHECK-NEXT:       %[[CMV_0:.*]] = ktdp.construct_memory_view %[[C1024]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_3]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16, "DDR">
// CHECK-NEXT:       %[[MULI_0:.*]] = arith.muli %[[ADDI_0]], %[[C64]] : index
// CHECK-NEXT:       %[[RC_0:.*]] = memref.reinterpret_cast %[[CMV_0]] to offset: {{\[}}%[[MULI_0]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:       agen.composite_load_and_store src:%[[RC_0]]{{\[}}%[[C0]], %[[C0]]] dst:%[[ALLOC_0]]{{\[}}%[[C0]], %[[C0]]]
// CHECK-NEXT:        time_symbols(), load_iv(%[[TIV_0:.*]]:vector<64xf16>)
// CHECK-NEXT:        {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_4]], load_time_addr_map = #[[$ATTR_1]], store_order = #[[$ATTR_0]], store_set = #[[$ATTR_4]], store_time_addr_map = #[[$ATTR_1]], time_order = #[[$ATTR_2]], time_set = #[[$ATTR_5]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }




// One single-unit-type pipeline. Expect exactly one dataflow.program_unit
// (for MNILU), and no remaining ktdf_lowering.* ops.


#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @single_stage_pipeline(%arg0: index) attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %2 = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %3 = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %4 = uniform.query_map(map:%3, key:%2) : index
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c1_2 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c3 = arith.constant 3 : index
    %c1024 = arith.constant 1024 : index
    %5 = ktdp.get_compute_tile_id : index
    %6 = arith.muli %5, %c3 : index
    %7 = arith.addi %6, %arg0 : index
    %c0_3 = arith.constant 0 : index
    %c1_4 = arith.constant 1 : index
    %c0_5 = arith.constant 0 : index
    %c64_6 = arith.constant 64 : index
    %8 = ktdp.construct_memory_view %c1024, sizes: [96, 64], strides: [64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<96x64xf16, "DDR">
    %c64_7 = arith.constant 64 : index
    %9 = arith.muli %7, %c64_7 : index
    %reinterpret_cast = memref.reinterpret_cast %8 to offset: [%9], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
    scf.for %arg1 = %c0_3 to %c1_4 step %c1_2 {
      scf.for %arg2 = %c0_5 to %c64_6 step %c64 {
        ktdf_lowering.execute_on %4 {
          %alloc = memref.alloc() : memref<1x64xf16, "L1">
          %10 = ktdf.create_token : !ktdf.token
          ktdf_lowering.execute_on %4 {
            ktdf.data_transfer from %reinterpret_cast[%arg1, %arg2] size [1, 64] to %alloc[%c0_0, %c0_0] size [1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
          }
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
