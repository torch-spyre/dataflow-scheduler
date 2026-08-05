// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// CHECK lines should be minimized and named to reflect the test’s intent.
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2 * 64 + d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (0, 0, 0, 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0 * 4096 + d1 * 4096 + d2 * 4096 + d3 * 64 + d4)>
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// The 49152-element transfer below walks its two non-unit outer dims (12 and 64)
// over the time axis, so ATTR_6 gains a second time dim and ATTR_9 narrows to a
// single vector. ATTR_10/ATTR_11 are its 2-dim time_order and time_set.
// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0, d1) -> (d0, 0, 0, d1, 0)>
// CHECK: #[[$ATTR_10:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_7:.+]] = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 == 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$ATTR_8:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK: #[[$ATTR_9:.+]] = affine_set<(d0, d1, d2, d3, d4) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0, d4 >= 0, -d4 + 63 >= 0)>
// CHECK: #[[$ATTR_11:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK:   module {
// CHECK:     module {
// CHECK:     func.func @test() attributes {grid = [2]} {
// CHECK-NEXT:       call @"local-schedule-0"() : () -> ()
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func private @"local-schedule-0"()
// CHECK-NEXT:   }

// CHECK:     module {
// CHECK:     func.func private @"local-schedule-0"() attributes {grid = [2]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 196736 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 128 : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 113216 : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 64000 : index
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 64 : index
// CHECK-NEXT:       %[[CONSTANT_5:.*]] = arith.constant 12 : index
// CHECK-NEXT:       %[[CONSTANT_6:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_7:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:       %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-NEXT:       %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:       %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:       %[[GET_UNIT_4:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:       %[[GET_UNIT_5:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:       %[[GET_UNIT_6:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
// CHECK-NEXT:       %[[GET_UNIT_7:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
// CHECK-NEXT:       %[[GET_UNIT_8:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
// CHECK-NEXT:       %[[GET_UNIT_9:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
// CHECK-NEXT:       %[[GET_UNIT_10:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"} : index
// CHECK-NEXT:       %[[GET_UNIT_11:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-l1", type = "l1"} : index
// CHECK-NEXT:       %[[GET_UNIT_12:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-l1", type = "l1"} : index
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_1:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_1]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_0:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_10]], %[[CONSTANT_3]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_1:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_0]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_2:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_5]] step %[[CONSTANT_6]] {
// CHECK-NEXT:             scf.for %[[VAL_3:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_4]] step %[[CONSTANT_6]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_0]]{{\[}}%[[VAL_2]], %[[CONSTANT_7]], %[[VAL_3]], %[[CONSTANT_7]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_1]]{{\[}}%[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_4:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_7]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_7]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_8]]}
// CHECK-NEXT:               {
// CHECK-NEXT:                 agen.yield
// CHECK-NEXT:               } : memref<12x1x64x64xf16>, memref<12x1x64x64xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_5:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_6:.*]] -> (%[[GET_UNIT_4]], %[[GET_UNIT_5]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_7:.*]] -> (%[[GET_UNIT_6]], %[[GET_UNIT_7]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_8:.*]] -> (%[[GET_UNIT_8]], %[[GET_UNIT_9]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_8]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_9]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[VAL_8]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_2:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_10]], %[[CONSTANT_2]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_3:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_1]], %[[CONSTANT_0]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_9:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_5]] step %[[CONSTANT_6]] {
// CHECK-NEXT:             scf.for %[[VAL_10:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_4]] step %[[CONSTANT_6]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_3]]{{\[}}%[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_2]]{{\[}}%[[VAL_9]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[VAL_10]], %[[CONSTANT_7]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_11:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_5]], load_set = #[[$ATTR_9]], load_time_addr_map = #[[$ATTR_6]], store_order = #[[$ATTR_5]], store_set = #[[$ATTR_9]], store_time_addr_map = #[[$ATTR_6]], time_order = #[[$ATTR_10]], time_set = #[[$ATTR_11]]}
// CHECK-NEXT:               {
// CHECK-NEXT:                 agen.yield
// CHECK-NEXT:               } : memref<12x1x1x64x64xf16>, memref<12x1x1x64x64xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_12:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_13:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_2:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_2]], key:%[[VAL_13]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_4:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_10]], %[[CONSTANT_3]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_5:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_2]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_14:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_5]] step %[[CONSTANT_6]] {
// CHECK-NEXT:             scf.for %[[VAL_15:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_4]] step %[[CONSTANT_6]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_4]]{{\[}}%[[VAL_14]], %[[CONSTANT_7]], %[[VAL_15]], %[[CONSTANT_7]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_5]]{{\[}}%[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_16:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_7]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_7]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_8]]}
// CHECK-NEXT:               {
// CHECK-NEXT:                 agen.yield
// CHECK-NEXT:               } : memref<12x1x64x64xf16>, memref<12x1x64x64xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_17:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_18:.*]] -> (%[[GET_UNIT_4]], %[[GET_UNIT_5]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_19:.*]] -> (%[[GET_UNIT_6]], %[[GET_UNIT_7]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_20:.*]] -> (%[[GET_UNIT_8]], %[[GET_UNIT_9]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_3:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_8]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_9]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_3:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_3]], key:%[[VAL_20]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_6:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_10]], %[[CONSTANT_2]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_7:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_3]], %[[CONSTANT_0]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_21:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_5]] step %[[CONSTANT_6]] {
// CHECK-NEXT:             scf.for %[[VAL_22:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_4]] step %[[CONSTANT_6]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_7]]{{\[}}%[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_6]]{{\[}}%[[VAL_21]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[VAL_22]], %[[CONSTANT_7]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_23:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_5]], load_set = #[[$ATTR_9]], load_time_addr_map = #[[$ATTR_6]], store_order = #[[$ATTR_5]], store_set = #[[$ATTR_9]], store_time_addr_map = #[[$ATTR_6]], time_order = #[[$ATTR_10]], time_set = #[[$ATTR_11]]}
// CHECK-NEXT:               {
// CHECK-NEXT:                 agen.yield
// CHECK-NEXT:               } : memref<12x1x1x64x64xf16>, memref<12x1x1x64x64xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_24:.*]] -> (%[[GET_UNIT_4]], %[[GET_UNIT_5]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_25:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_4:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_4:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_4]], key:%[[VAL_25]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_8:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_10]], %[[CONSTANT_3]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_9:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_4]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_26:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_5]] step %[[CONSTANT_6]] {
// CHECK-NEXT:             scf.for %[[VAL_27:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_4]] step %[[CONSTANT_6]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_8]]{{\[}}%[[VAL_26]], %[[CONSTANT_7]], %[[VAL_27]], %[[CONSTANT_7]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_9]]{{\[}}%[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_28:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_7]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_7]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_8]]}
// CHECK-NEXT:               {
// CHECK-NEXT:                 agen.yield
// CHECK-NEXT:               } : memref<12x1x64x64xf16>, memref<12x1x64x64xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_29:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_30:.*]] -> (%[[GET_UNIT_4]], %[[GET_UNIT_5]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_31:.*]] -> (%[[GET_UNIT_6]], %[[GET_UNIT_7]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_32:.*]] -> (%[[GET_UNIT_8]], %[[GET_UNIT_9]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_5:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_8]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_9]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_5:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_5]], key:%[[VAL_32]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_10:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_10]], %[[CONSTANT_2]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_11:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_5]], %[[CONSTANT_0]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_33:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_5]] step %[[CONSTANT_6]] {
// CHECK-NEXT:             scf.for %[[VAL_34:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_4]] step %[[CONSTANT_6]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_11]]{{\[}}%[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_10]]{{\[}}%[[VAL_33]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[VAL_34]], %[[CONSTANT_7]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_35:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_5]], load_set = #[[$ATTR_9]], load_time_addr_map = #[[$ATTR_6]], store_order = #[[$ATTR_5]], store_set = #[[$ATTR_9]], store_time_addr_map = #[[$ATTR_6]], time_order = #[[$ATTR_10]], time_set = #[[$ATTR_11]]}
// CHECK-NEXT:               {
// CHECK-NEXT:                 agen.yield
// CHECK-NEXT:               } : memref<12x1x1x64x64xf16>, memref<12x1x1x64x64xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_36:.*]] -> (%[[GET_UNIT_6]], %[[GET_UNIT_7]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_37:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_6:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_6:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_6]], key:%[[VAL_37]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_12:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_10]], %[[CONSTANT_3]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_13:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_6]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_38:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_5]] step %[[CONSTANT_6]] {
// CHECK-NEXT:             scf.for %[[VAL_39:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_4]] step %[[CONSTANT_6]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_12]]{{\[}}%[[VAL_38]], %[[CONSTANT_7]], %[[VAL_39]], %[[CONSTANT_7]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_13]]{{\[}}%[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_40:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_7]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_7]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_8]]}
// CHECK-NEXT:               {
// CHECK-NEXT:                 agen.yield
// CHECK-NEXT:               } : memref<12x1x64x64xf16>, memref<12x1x64x64xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_41:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_42:.*]] -> (%[[GET_UNIT_4]], %[[GET_UNIT_5]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_43:.*]] -> (%[[GET_UNIT_6]], %[[GET_UNIT_7]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_44:.*]] -> (%[[GET_UNIT_8]], %[[GET_UNIT_9]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_7:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_8]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_9]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_7:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_7]], key:%[[VAL_44]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_14:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_10]], %[[CONSTANT_2]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_15:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_7]], %[[CONSTANT_0]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_45:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_5]] step %[[CONSTANT_6]] {
// CHECK-NEXT:             scf.for %[[VAL_46:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_4]] step %[[CONSTANT_6]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_15]]{{\[}}%[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_14]]{{\[}}%[[VAL_45]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[VAL_46]], %[[CONSTANT_7]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_47:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_5]], load_set = #[[$ATTR_9]], load_time_addr_map = #[[$ATTR_6]], store_order = #[[$ATTR_5]], store_set = #[[$ATTR_9]], store_time_addr_map = #[[$ATTR_6]], time_order = #[[$ATTR_10]], time_set = #[[$ATTR_11]]}
// CHECK-NEXT:               {
// CHECK-NEXT:                 agen.yield
// CHECK-NEXT:               } : memref<12x1x1x64x64xf16>, memref<12x1x1x64x64xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_48:.*]] -> (%[[GET_UNIT_8]], %[[GET_UNIT_9]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_49:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_8:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_8:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_8]], key:%[[VAL_49]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_16:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_10]], %[[CONSTANT_3]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_17:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_8]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_50:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_5]] step %[[CONSTANT_6]] {
// CHECK-NEXT:             scf.for %[[VAL_51:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_4]] step %[[CONSTANT_6]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_16]]{{\[}}%[[VAL_50]], %[[CONSTANT_7]], %[[VAL_51]], %[[CONSTANT_7]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_17]]{{\[}}%[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_52:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_7]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_7]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_8]]}
// CHECK-NEXT:               {
// CHECK-NEXT:                 agen.yield
// CHECK-NEXT:               } : memref<12x1x64x64xf16>, memref<12x1x64x64xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_53:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_54:.*]] -> (%[[GET_UNIT_4]], %[[GET_UNIT_5]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_55:.*]] -> (%[[GET_UNIT_6]], %[[GET_UNIT_7]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_56:.*]] -> (%[[GET_UNIT_8]], %[[GET_UNIT_9]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_9:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_8]] -> %[[GET_UNIT_11]]], {{\[}}%[[GET_UNIT_9]] -> %[[GET_UNIT_12]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_9:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_9]], key:%[[VAL_56]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_18:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_10]], %[[CONSTANT_2]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_19:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_9]], %[[CONSTANT_0]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_57:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_5]] step %[[CONSTANT_6]] {
// CHECK-NEXT:             scf.for %[[VAL_58:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_4]] step %[[CONSTANT_6]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_19]]{{\[}}%[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[CONSTANT_7]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_18]]{{\[}}%[[VAL_57]], %[[CONSTANT_7]], %[[CONSTANT_7]], %[[VAL_58]], %[[CONSTANT_7]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_59:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_5]], load_set = #[[$ATTR_9]], load_time_addr_map = #[[$ATTR_6]], store_order = #[[$ATTR_5]], store_set = #[[$ATTR_9]], store_time_addr_map = #[[$ATTR_6]], time_order = #[[$ATTR_10]], time_set = #[[$ATTR_11]]}
// CHECK-NEXT:               {
// CHECK-NEXT:                 agen.yield
// CHECK-NEXT:               } : memref<12x1x1x64x64xf16>, memref<12x1x1x64x64xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }



// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/






//
// Test: Multiple program_units in one function — MNILU (DDR source + L1 dest)
// and L1SU (L1 source + DDR dest). Checks:
//   - DDR get_unit emitted once at func level (singleton)
//   - L1 get_unit ops emitted once at func level (per core)
//   - Each program_unit gets its own uniform map + query for L1
//   - DDR used directly (no uniform map) in both units
//   - Source A and B lowered correctly in both units

// Func-level memory units — DDR once, L1 per core

// MNILU program_unit: DDR source (Source A), L1 dest (Source B)

// MNISU program_unit: L1 source (Source B), DDR dest (Source A)

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test() attributes {grid = [2]} {
      call @"local-schedule-0"() : () -> ()
      return
    }
    func.func private @"local-schedule-0"()
  }
  module {
    func.func private @"local-schedule-0"() attributes {grid = [2]} {
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %6 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %7 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %8 = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
      %9 = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c64000 = arith.constant 64000 : index
      %c113216 = arith.constant 113216 : index
      %c128 = arith.constant 128 : index
      %c196736 = arith.constant 196736 : index

      // MNILU: DDR -> L1
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        %10 = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<12x1x64x64xf16>
        %msc = memref.memory_space_cast %10 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc = memref.reinterpret_cast %msc to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
        %cast = memref.cast %rc : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1_dst = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %arg1 = %c0 to %c12 step %c1 {
          scf.for %arg2 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %cast[%arg1, %c0, %arg2, %c0] size [1, 1, 1, 64] to %l1_dst[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1_dst : memref<12x1x64x64xf16, "L1">
      }
      dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%6, %7) : {
      }

      // MNISU: L1 -> DDR
      dataflow.program_unit iter_arg : %arg0 -> (%8, %9) : {
        %10 = ktdp.construct_memory_view %c113216, sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<12x1x1x64x64xf16>
        %msc = memref.memory_space_cast %10 : memref<12x1x1x64x64xf16> to memref<12x1x1x64x64xf16, "DDR">
        %rc = memref.reinterpret_cast %msc to offset: [0], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] : memref<12x1x1x64x64xf16, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR">
        %cast = memref.cast %rc : memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1_src = builtin.unrealized_conversion_cast %c196736 : index to memref<12x1x1x64x64xf16, "L1">
        scf.for %arg1 = %c0 to %c12 step %c1 {
          scf.for %arg2 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %l1_src[%c0, %c0, %c0, %c0, %c0] size [12, 1, 1, 64, 64] to %cast[%arg1, %c0, %c0, %arg2, %c0] size [12, 1, 1, 64, 64] : memref<12x1x1x64x64xf16, "L1">, memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1_src : memref<12x1x1x64x64xf16, "L1">
      }
      return
    }
  }
}
