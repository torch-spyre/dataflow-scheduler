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
// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0, d1) -> (d0, 0, 0, d1, 0)>
// CHECK: #[[$ATTR_7:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_8:.+]] = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 == 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$ATTR_9:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK: #[[$ATTR_10:.+]] = affine_set<(d0, d1, d2, d3, d4) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0, d4 >= 0, -d4 + 63 >= 0)>
// CHECK: #[[$ATTR_11:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK:         module {
// CHECK:           ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK:           module {
// CHECK:             func.func @test() attributes {grid = [2]} {
// CHECK:               call @"local-schedule-0"() : () -> ()
// CHECK:               return
// CHECK:             }
// CHECK:             func.func private @"local-schedule-0"()
// CHECK:           }

// CHECK:           module {
// CHECK:             func.func private @"local-schedule-0"() attributes {grid = [2]} {
// CHECK:             %[[VAL_0:.*]] = arith.constant 196736 : index
// CHECK:             %[[VAL_1:.*]] = arith.constant 128 : index
// CHECK:             %[[VAL_2:.*]] = arith.constant 113216 : index
// CHECK:             %[[VAL_3:.*]] = arith.constant 64000 : index
// CHECK:             %[[VAL_4:.*]] = arith.constant 64 : index
// CHECK:             %[[VAL_5:.*]] = arith.constant 12 : index
// CHECK:             %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:             %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_8:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK:             %[[VAL_9:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK:             %[[VAL_10:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK:             %[[VAL_11:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK:             %[[VAL_12:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK:             %[[VAL_13:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK:             %[[VAL_14:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
// CHECK:             %[[VAL_15:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
// CHECK:             %[[VAL_16:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
// CHECK:             %[[VAL_17:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
// CHECK:             %[[VAL_18:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"} : index
// CHECK:             %[[VAL_19:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-l1", type = "l1"} : index
// CHECK:             %[[VAL_20:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-l1", type = "l1"} : index
// CHECK:             dataflow.program_unit iter_arg : %[[VAL_21:.*]] -> (%[[VAL_8]], %[[VAL_9]]) : {
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_22:.*]] -> (%[[VAL_8]], %[[VAL_9]]) : {
// CHECK:                 %[[VAL_23:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_8]] -> %[[VAL_19]]], {{\[}}%[[VAL_9]] -> %[[VAL_20]]]):index
// CHECK:                 %[[VAL_24:.*]] = uniform.query_map(map:%[[VAL_23]], key:%[[VAL_22]]) : index
// CHECK:                 %[[VAL_25:.*]] = dataflow.get_logical_memory_view %[[VAL_18]], %[[VAL_3]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK:                 %[[VAL_26:.*]] = dataflow.get_logical_memory_view %[[VAL_24]], %[[VAL_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK:                 scf.for %[[VAL_27:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_6]] {
// CHECK:                   scf.for %[[VAL_28:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:                     agen.composite_load_and_store src:%[[VAL_25]]{{\[}}%[[VAL_27]], %[[VAL_7]], %[[VAL_28]], %[[VAL_7]]] dst:%[[VAL_26]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]]]
// CHECK:                      time_symbols(), load_iv(%[[VAL_29:.*]]:vector<64xf16>)
// CHECK:                      {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_8]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_8]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_9]]}
// CHECK:                     {
// CHECK:                       agen.yield
// CHECK:                     } : memref<12x1x64x64xf16>, memref<12x1x64x64xf16>
// CHECK:                   } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_30:.*]] -> (%[[VAL_10]], %[[VAL_11]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_31:.*]] -> (%[[VAL_12]], %[[VAL_13]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_32:.*]] -> (%[[VAL_14]], %[[VAL_15]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_33:.*]] -> (%[[VAL_16]], %[[VAL_17]]) : {
// CHECK:                 %[[VAL_34:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_16]] -> %[[VAL_19]]], {{\[}}%[[VAL_17]] -> %[[VAL_20]]]):index
// CHECK:                 %[[VAL_35:.*]] = uniform.query_map(map:%[[VAL_34]], key:%[[VAL_33]]) : index
// CHECK:                 %[[VAL_36:.*]] = dataflow.get_logical_memory_view %[[VAL_18]], %[[VAL_2]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK:                 %[[VAL_37:.*]] = dataflow.get_logical_memory_view %[[VAL_35]], %[[VAL_0]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK:                 scf.for %[[VAL_38:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_6]] {
// CHECK:                   scf.for %[[VAL_39:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:                     agen.composite_load_and_store src:%[[VAL_37]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]]] dst:%[[VAL_36]]{{\[}}%[[VAL_38]], %[[VAL_7]], %[[VAL_7]], %[[VAL_39]], %[[VAL_7]]]
// CHECK:                      time_symbols(), load_iv(%[[VAL_40:.*]]:vector<64xf16>)
// CHECK:                      {load_order = #[[$ATTR_5]], load_set = #[[$ATTR_10]], load_time_addr_map = #[[$ATTR_6]], store_order = #[[$ATTR_5]], store_set = #[[$ATTR_10]], store_time_addr_map = #[[$ATTR_6]], time_order = #[[$ATTR_7]], time_set = #[[$ATTR_11]]}
// CHECK:                     {
// CHECK:                       agen.yield
// CHECK:                     } : memref<12x1x1x64x64xf16>, memref<12x1x1x64x64xf16>
// CHECK:                   } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               }
// CHECK:             }
// CHECK:             dataflow.program_unit iter_arg : %[[VAL_41:.*]] -> (%[[VAL_10]], %[[VAL_11]]) : {
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_42:.*]] -> (%[[VAL_8]], %[[VAL_9]]) : {
// CHECK:                 %[[VAL_43:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_8]] -> %[[VAL_19]]], {{\[}}%[[VAL_9]] -> %[[VAL_20]]]):index
// CHECK:                 %[[VAL_44:.*]] = uniform.query_map(map:%[[VAL_43]], key:%[[VAL_42]]) : index
// CHECK:                 %[[VAL_45:.*]] = dataflow.get_logical_memory_view %[[VAL_18]], %[[VAL_3]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK:                 %[[VAL_46:.*]] = dataflow.get_logical_memory_view %[[VAL_44]], %[[VAL_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK:                 scf.for %[[VAL_47:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_6]] {
// CHECK:                   scf.for %[[VAL_48:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:                     agen.composite_load_and_store src:%[[VAL_45]]{{\[}}%[[VAL_47]], %[[VAL_7]], %[[VAL_48]], %[[VAL_7]]] dst:%[[VAL_46]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]]]
// CHECK:                      time_symbols(), load_iv(%[[VAL_49:.*]]:vector<64xf16>)
// CHECK:                      {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_8]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_8]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_9]]}
// CHECK:                     {
// CHECK:                       agen.yield
// CHECK:                     } : memref<12x1x64x64xf16>, memref<12x1x64x64xf16>
// CHECK:                   } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_50:.*]] -> (%[[VAL_10]], %[[VAL_11]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_51:.*]] -> (%[[VAL_12]], %[[VAL_13]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_52:.*]] -> (%[[VAL_14]], %[[VAL_15]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_53:.*]] -> (%[[VAL_16]], %[[VAL_17]]) : {
// CHECK:                 %[[VAL_54:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_16]] -> %[[VAL_19]]], {{\[}}%[[VAL_17]] -> %[[VAL_20]]]):index
// CHECK:                 %[[VAL_55:.*]] = uniform.query_map(map:%[[VAL_54]], key:%[[VAL_53]]) : index
// CHECK:                 %[[VAL_56:.*]] = dataflow.get_logical_memory_view %[[VAL_18]], %[[VAL_2]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK:                 %[[VAL_57:.*]] = dataflow.get_logical_memory_view %[[VAL_55]], %[[VAL_0]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK:                 scf.for %[[VAL_58:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_6]] {
// CHECK:                   scf.for %[[VAL_59:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:                     agen.composite_load_and_store src:%[[VAL_57]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]]] dst:%[[VAL_56]]{{\[}}%[[VAL_58]], %[[VAL_7]], %[[VAL_7]], %[[VAL_59]], %[[VAL_7]]]
// CHECK:                      time_symbols(), load_iv(%[[VAL_60:.*]]:vector<64xf16>)
// CHECK:                      {load_order = #[[$ATTR_5]], load_set = #[[$ATTR_10]], load_time_addr_map = #[[$ATTR_6]], store_order = #[[$ATTR_5]], store_set = #[[$ATTR_10]], store_time_addr_map = #[[$ATTR_6]], time_order = #[[$ATTR_7]], time_set = #[[$ATTR_11]]}
// CHECK:                     {
// CHECK:                       agen.yield
// CHECK:                     } : memref<12x1x1x64x64xf16>, memref<12x1x1x64x64xf16>
// CHECK:                   } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               }
// CHECK:             }
// CHECK:             dataflow.program_unit iter_arg : %[[VAL_61:.*]] -> (%[[VAL_12]], %[[VAL_13]]) : {
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_62:.*]] -> (%[[VAL_8]], %[[VAL_9]]) : {
// CHECK:                 %[[VAL_63:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_8]] -> %[[VAL_19]]], {{\[}}%[[VAL_9]] -> %[[VAL_20]]]):index
// CHECK:                 %[[VAL_64:.*]] = uniform.query_map(map:%[[VAL_63]], key:%[[VAL_62]]) : index
// CHECK:                 %[[VAL_65:.*]] = dataflow.get_logical_memory_view %[[VAL_18]], %[[VAL_3]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK:                 %[[VAL_66:.*]] = dataflow.get_logical_memory_view %[[VAL_64]], %[[VAL_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK:                 scf.for %[[VAL_67:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_6]] {
// CHECK:                   scf.for %[[VAL_68:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:                     agen.composite_load_and_store src:%[[VAL_65]]{{\[}}%[[VAL_67]], %[[VAL_7]], %[[VAL_68]], %[[VAL_7]]] dst:%[[VAL_66]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]]]
// CHECK:                      time_symbols(), load_iv(%[[VAL_69:.*]]:vector<64xf16>)
// CHECK:                      {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_8]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_8]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_9]]}
// CHECK:                     {
// CHECK:                       agen.yield
// CHECK:                     } : memref<12x1x64x64xf16>, memref<12x1x64x64xf16>
// CHECK:                   } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_70:.*]] -> (%[[VAL_10]], %[[VAL_11]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_71:.*]] -> (%[[VAL_12]], %[[VAL_13]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_72:.*]] -> (%[[VAL_14]], %[[VAL_15]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_73:.*]] -> (%[[VAL_16]], %[[VAL_17]]) : {
// CHECK:                 %[[VAL_74:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_16]] -> %[[VAL_19]]], {{\[}}%[[VAL_17]] -> %[[VAL_20]]]):index
// CHECK:                 %[[VAL_75:.*]] = uniform.query_map(map:%[[VAL_74]], key:%[[VAL_73]]) : index
// CHECK:                 %[[VAL_76:.*]] = dataflow.get_logical_memory_view %[[VAL_18]], %[[VAL_2]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK:                 %[[VAL_77:.*]] = dataflow.get_logical_memory_view %[[VAL_75]], %[[VAL_0]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK:                 scf.for %[[VAL_78:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_6]] {
// CHECK:                   scf.for %[[VAL_79:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:                     agen.composite_load_and_store src:%[[VAL_77]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]]] dst:%[[VAL_76]]{{\[}}%[[VAL_78]], %[[VAL_7]], %[[VAL_7]], %[[VAL_79]], %[[VAL_7]]]
// CHECK:                      time_symbols(), load_iv(%[[VAL_80:.*]]:vector<64xf16>)
// CHECK:                      {load_order = #[[$ATTR_5]], load_set = #[[$ATTR_10]], load_time_addr_map = #[[$ATTR_6]], store_order = #[[$ATTR_5]], store_set = #[[$ATTR_10]], store_time_addr_map = #[[$ATTR_6]], time_order = #[[$ATTR_7]], time_set = #[[$ATTR_11]]}
// CHECK:                     {
// CHECK:                       agen.yield
// CHECK:                     } : memref<12x1x1x64x64xf16>, memref<12x1x1x64x64xf16>
// CHECK:                   } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               }
// CHECK:             }
// CHECK:             dataflow.program_unit iter_arg : %[[VAL_81:.*]] -> (%[[VAL_14]], %[[VAL_15]]) : {
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_82:.*]] -> (%[[VAL_8]], %[[VAL_9]]) : {
// CHECK:                 %[[VAL_83:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_8]] -> %[[VAL_19]]], {{\[}}%[[VAL_9]] -> %[[VAL_20]]]):index
// CHECK:                 %[[VAL_84:.*]] = uniform.query_map(map:%[[VAL_83]], key:%[[VAL_82]]) : index
// CHECK:                 %[[VAL_85:.*]] = dataflow.get_logical_memory_view %[[VAL_18]], %[[VAL_3]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK:                 %[[VAL_86:.*]] = dataflow.get_logical_memory_view %[[VAL_84]], %[[VAL_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK:                 scf.for %[[VAL_87:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_6]] {
// CHECK:                   scf.for %[[VAL_88:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:                     agen.composite_load_and_store src:%[[VAL_85]]{{\[}}%[[VAL_87]], %[[VAL_7]], %[[VAL_88]], %[[VAL_7]]] dst:%[[VAL_86]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]]]
// CHECK:                      time_symbols(), load_iv(%[[VAL_89:.*]]:vector<64xf16>)
// CHECK:                      {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_8]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_8]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_9]]}
// CHECK:                     {
// CHECK:                       agen.yield
// CHECK:                     } : memref<12x1x64x64xf16>, memref<12x1x64x64xf16>
// CHECK:                   } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_90:.*]] -> (%[[VAL_10]], %[[VAL_11]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_91:.*]] -> (%[[VAL_12]], %[[VAL_13]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_92:.*]] -> (%[[VAL_14]], %[[VAL_15]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_93:.*]] -> (%[[VAL_16]], %[[VAL_17]]) : {
// CHECK:                 %[[VAL_94:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_16]] -> %[[VAL_19]]], {{\[}}%[[VAL_17]] -> %[[VAL_20]]]):index
// CHECK:                 %[[VAL_95:.*]] = uniform.query_map(map:%[[VAL_94]], key:%[[VAL_93]]) : index
// CHECK:                 %[[VAL_96:.*]] = dataflow.get_logical_memory_view %[[VAL_18]], %[[VAL_2]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK:                 %[[VAL_97:.*]] = dataflow.get_logical_memory_view %[[VAL_95]], %[[VAL_0]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK:                 scf.for %[[VAL_98:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_6]] {
// CHECK:                   scf.for %[[VAL_99:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:                     agen.composite_load_and_store src:%[[VAL_97]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]]] dst:%[[VAL_96]]{{\[}}%[[VAL_98]], %[[VAL_7]], %[[VAL_7]], %[[VAL_99]], %[[VAL_7]]]
// CHECK:                      time_symbols(), load_iv(%[[VAL_100:.*]]:vector<64xf16>)
// CHECK:                      {load_order = #[[$ATTR_5]], load_set = #[[$ATTR_10]], load_time_addr_map = #[[$ATTR_6]], store_order = #[[$ATTR_5]], store_set = #[[$ATTR_10]], store_time_addr_map = #[[$ATTR_6]], time_order = #[[$ATTR_7]], time_set = #[[$ATTR_11]]}
// CHECK:                     {
// CHECK:                       agen.yield
// CHECK:                     } : memref<12x1x1x64x64xf16>, memref<12x1x1x64x64xf16>
// CHECK:                   } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               }
// CHECK:             }
// CHECK:             dataflow.program_unit iter_arg : %[[VAL_101:.*]] -> (%[[VAL_16]], %[[VAL_17]]) : {
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_102:.*]] -> (%[[VAL_8]], %[[VAL_9]]) : {
// CHECK:                 %[[VAL_103:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_8]] -> %[[VAL_19]]], {{\[}}%[[VAL_9]] -> %[[VAL_20]]]):index
// CHECK:                 %[[VAL_104:.*]] = uniform.query_map(map:%[[VAL_103]], key:%[[VAL_102]]) : index
// CHECK:                 %[[VAL_105:.*]] = dataflow.get_logical_memory_view %[[VAL_18]], %[[VAL_3]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK:                 %[[VAL_106:.*]] = dataflow.get_logical_memory_view %[[VAL_104]], %[[VAL_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK:                 scf.for %[[VAL_107:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_6]] {
// CHECK:                   scf.for %[[VAL_108:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:                     agen.composite_load_and_store src:%[[VAL_105]]{{\[}}%[[VAL_107]], %[[VAL_7]], %[[VAL_108]], %[[VAL_7]]] dst:%[[VAL_106]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]]]
// CHECK:                      time_symbols(), load_iv(%[[VAL_109:.*]]:vector<64xf16>)
// CHECK:                      {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_8]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_8]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_9]]}
// CHECK:                     {
// CHECK:                       agen.yield
// CHECK:                     } : memref<12x1x64x64xf16>, memref<12x1x64x64xf16>
// CHECK:                   } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_110:.*]] -> (%[[VAL_10]], %[[VAL_11]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_111:.*]] -> (%[[VAL_12]], %[[VAL_13]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_112:.*]] -> (%[[VAL_14]], %[[VAL_15]]) : {
// CHECK:               }
// CHECK:               dataflow.program_unit iter_arg : %[[VAL_113:.*]] -> (%[[VAL_16]], %[[VAL_17]]) : {
// CHECK:                 %[[VAL_114:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_16]] -> %[[VAL_19]]], {{\[}}%[[VAL_17]] -> %[[VAL_20]]]):index
// CHECK:                 %[[VAL_115:.*]] = uniform.query_map(map:%[[VAL_114]], key:%[[VAL_113]]) : index
// CHECK:                 %[[VAL_116:.*]] = dataflow.get_logical_memory_view %[[VAL_18]], %[[VAL_2]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK:                 %[[VAL_117:.*]] = dataflow.get_logical_memory_view %[[VAL_115]], %[[VAL_0]] {layout_map = #[[$ATTR_4]]} : index, index, memref<12x1x1x64x64xf16>
// CHECK:                 scf.for %[[VAL_118:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_6]] {
// CHECK:                   scf.for %[[VAL_119:.*]] = %[[VAL_7]] to %[[VAL_4]] step %[[VAL_6]] {
// CHECK:                     agen.composite_load_and_store src:%[[VAL_117]]{{\[}}%[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]], %[[VAL_7]]] dst:%[[VAL_116]]{{\[}}%[[VAL_118]], %[[VAL_7]], %[[VAL_7]], %[[VAL_119]], %[[VAL_7]]]
// CHECK:                      time_symbols(), load_iv(%[[VAL_120:.*]]:vector<64xf16>)
// CHECK:                      {load_order = #[[$ATTR_5]], load_set = #[[$ATTR_10]], load_time_addr_map = #[[$ATTR_6]], store_order = #[[$ATTR_5]], store_set = #[[$ATTR_10]], store_time_addr_map = #[[$ATTR_6]], time_order = #[[$ATTR_7]], time_set = #[[$ATTR_11]]}
// CHECK:                     {
// CHECK:                       agen.yield
// CHECK:                     } : memref<12x1x1x64x64xf16>, memref<12x1x1x64x64xf16>
// CHECK:                   } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:                 } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK:               }
// CHECK:             }
// CHECK:             return
// CHECK:           }
// CHECK:         }



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
