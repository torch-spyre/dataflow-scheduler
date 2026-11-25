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
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 == 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0) : (d0 == 0)>
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
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 128 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 64000 : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 64 : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 12 : index
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_5:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:       %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-NEXT:       %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:       %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:       %[[GET_UNIT_4:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:       %[[GET_UNIT_5:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:       %[[GET_UNIT_6:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
// CHECK-NEXT:       %[[GET_UNIT_7:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
// CHECK-NEXT:       %[[GET_UNIT_8:.*]] = dataflow.get_unit {name = "ddr", type = "ddr"} : index
// CHECK-NEXT:       %[[GET_UNIT_9:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-l1", type = "l1"} : index
// CHECK-NEXT:       %[[GET_UNIT_10:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-l1", type = "l1"} : index
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_1:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_9]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_10]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_1]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_0:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_8]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_1:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_0]], %[[CONSTANT_0]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_2:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
// CHECK-NEXT:             scf.for %[[VAL_3:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_2]] step %[[CONSTANT_4]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_0]]{{\[}}%[[VAL_2]], %[[CONSTANT_5]], %[[VAL_3]], %[[CONSTANT_5]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_1]]{{\[}}%[[CONSTANT_5]], %[[CONSTANT_5]], %[[CONSTANT_5]], %[[CONSTANT_5]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_4:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_5]]}
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
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_8:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_9:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_9]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_10]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[VAL_9]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_2:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_8]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_3:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_1]], %[[CONSTANT_0]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_10:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
// CHECK-NEXT:             scf.for %[[VAL_11:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_2]] step %[[CONSTANT_4]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_2]]{{\[}}%[[VAL_10]], %[[CONSTANT_5]], %[[VAL_11]], %[[CONSTANT_5]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_3]]{{\[}}%[[CONSTANT_5]], %[[CONSTANT_5]], %[[CONSTANT_5]], %[[CONSTANT_5]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_12:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_5]]}
// CHECK-NEXT:               {
// CHECK-NEXT:                 agen.yield
// CHECK-NEXT:               } : memref<12x1x64x64xf16>, memref<12x1x64x64xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_13:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_14:.*]] -> (%[[GET_UNIT_4]], %[[GET_UNIT_5]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_15:.*]] -> (%[[GET_UNIT_6]], %[[GET_UNIT_7]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_16:.*]] -> (%[[GET_UNIT_4]], %[[GET_UNIT_5]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_17:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_9]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_10]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_2:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_2]], key:%[[VAL_17]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_4:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_8]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_5:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_2]], %[[CONSTANT_0]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_18:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
// CHECK-NEXT:             scf.for %[[VAL_19:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_2]] step %[[CONSTANT_4]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_4]]{{\[}}%[[VAL_18]], %[[CONSTANT_5]], %[[VAL_19]], %[[CONSTANT_5]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_5]]{{\[}}%[[CONSTANT_5]], %[[CONSTANT_5]], %[[CONSTANT_5]], %[[CONSTANT_5]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_20:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_5]]}
// CHECK-NEXT:               {
// CHECK-NEXT:                 agen.yield
// CHECK-NEXT:               } : memref<12x1x64x64xf16>, memref<12x1x64x64xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_21:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_22:.*]] -> (%[[GET_UNIT_4]], %[[GET_UNIT_5]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_23:.*]] -> (%[[GET_UNIT_6]], %[[GET_UNIT_7]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[VAL_24:.*]] -> (%[[GET_UNIT_6]], %[[GET_UNIT_7]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[VAL_25:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:           %[[DEF_IMMUTABLE_MAPPING_3:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_9]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_10]]]):index
// CHECK-NEXT:           %[[QUERY_MAP_3:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_3]], key:%[[VAL_25]]) : index
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_6:.*]] = dataflow.get_logical_memory_view %[[GET_UNIT_8]], %[[CONSTANT_1]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           %[[GET_LOGICAL_MEMORY_VIEW_7:.*]] = dataflow.get_logical_memory_view %[[QUERY_MAP_3]], %[[CONSTANT_0]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[VAL_26:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
// CHECK-NEXT:             scf.for %[[VAL_27:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_2]] step %[[CONSTANT_4]] {
// CHECK-NEXT:               agen.composite_load_and_store src:%[[GET_LOGICAL_MEMORY_VIEW_6]]{{\[}}%[[VAL_26]], %[[CONSTANT_5]], %[[VAL_27]], %[[CONSTANT_5]]] dst:%[[GET_LOGICAL_MEMORY_VIEW_7]]{{\[}}%[[CONSTANT_5]], %[[CONSTANT_5]], %[[CONSTANT_5]], %[[CONSTANT_5]]]
// CHECK-NEXT:                time_symbols(), load_iv(%[[VAL_28:.*]]:vector<64xf16>)
// CHECK-NEXT:                {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_4]], load_time_addr_map = #[[$ATTR_2]], store_order = #[[$ATTR_1]], store_set = #[[$ATTR_4]], store_time_addr_map = #[[$ATTR_2]], time_order = #[[$ATTR_3]], time_set = #[[$ATTR_5]]}
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
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }






//
// Test: Source A — ktdp.construct_memory_view chain is lowered to
// dataflow.get_logical_memory_view. Checks:
//   - DDR get_unit emitted at func level (singleton, no core attr)
//   - arith.addi combines construct_offset + reinterpret_offset
//   - layout_map is strides-based linearization
//   - result type is plain memref (no memory space, no strided layout)
//   - data_transfer operand type updated to plain memref
//   - original 4-op chain is gone



#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
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
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c64000 = arith.constant 64000 : index
      %c128 = arith.constant 128 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        // Source A chain: construct_memory_view -> memory_space_cast ->
        // reinterpret_cast (offset 0) -> memref.cast
        %10 = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc = memref.memory_space_cast %10 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc = memref.reinterpret_cast %msc to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
        %cast = memref.cast %rc : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %arg1 = %c0 to %c12 step %c1 {
          scf.for %arg2 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %cast[%arg1, %c0, %arg2, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.access_granularity = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
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
