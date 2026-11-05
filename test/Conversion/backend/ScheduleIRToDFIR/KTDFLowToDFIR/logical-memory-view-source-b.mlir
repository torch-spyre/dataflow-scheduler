// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2 * 64 + d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
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
// CHECK-NEXT:       %[[C128:.*]] = arith.constant 128 : index
// CHECK-NEXT:       %[[C64:.*]] = arith.constant 64 : index
// CHECK-NEXT:       %[[C12:.*]] = arith.constant 12 : index
// CHECK-NEXT:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[L1LU_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:       %[[L1LU_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:       %[[SFU_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:       %[[SFU_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:       %[[L1SU_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
// CHECK-NEXT:       %[[L1SU_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
// CHECK-NEXT:       %[[L1_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-l1", type = "l1"} : index
// CHECK-NEXT:       %[[L1_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-l1", type = "l1"} : index
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[ARG0:.*]] -> (%[[L1LU_0]], %[[L1LU_1]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG1:.*]] -> (%[[L1LU_0]], %[[L1LU_1]]) : {
// CHECK-NEXT:           %[[MAP_L1_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU_0]] -> %[[L1_0]]], {{\[}}%[[L1LU_1]] -> %[[L1_1]]]):index
// CHECK-NEXT:           %[[L1_TGT_0:.*]] = uniform.query_map(map:%[[MAP_L1_0]], key:%[[ARG1]]) : index
// CHECK-NEXT:           %[[LMV_0:.*]] = dataflow.get_logical_memory_view %[[L1_TGT_0]], %[[C128]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[I0:.*]] = %[[C0]] to %[[C12]] step %[[C1]] {
// CHECK-NEXT:             scf.for %[[J0:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:               %[[MAP_SFU_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU_0]] -> %[[SFU_0]]], {{\[}}%[[L1LU_1]] -> %[[SFU_1]]]):index
// CHECK-NEXT:               %[[SFU_TGT_0:.*]] = uniform.query_map(map:%[[MAP_SFU_0]], key:%[[ARG1]]) : index
// CHECK-NEXT:               %[[VL_0:.*]] = agen.vector_load %[[LMV_0]]{{\[}}%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_2]]} : memref<12x1x64x64xf16>, vector<49152xf16>
// CHECK-NEXT:               dataflow.send %[[SFU_TGT_0]], %[[VL_0]] : vector<49152xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG2:.*]] -> (%[[SFU_0]], %[[SFU_1]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG3:.*]] -> (%[[L1SU_0]], %[[L1SU_1]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[ARG4:.*]] -> (%[[SFU_0]], %[[SFU_1]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG5:.*]] -> (%[[L1LU_0]], %[[L1LU_1]]) : {
// CHECK-NEXT:           %[[MAP_L1_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU_0]] -> %[[L1_0]]], {{\[}}%[[L1LU_1]] -> %[[L1_1]]]):index
// CHECK-NEXT:           %[[L1_TGT_1:.*]] = uniform.query_map(map:%[[MAP_L1_1]], key:%[[ARG5]]) : index
// CHECK-NEXT:           %[[LMV_1:.*]] = dataflow.get_logical_memory_view %[[L1_TGT_1]], %[[C128]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[I1:.*]] = %[[C0]] to %[[C12]] step %[[C1]] {
// CHECK-NEXT:             scf.for %[[J1:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:               %[[MAP_SFU_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU_0]] -> %[[SFU_0]]], {{\[}}%[[L1LU_1]] -> %[[SFU_1]]]):index
// CHECK-NEXT:               %[[SFU_TGT_1:.*]] = uniform.query_map(map:%[[MAP_SFU_1]], key:%[[ARG5]]) : index
// CHECK-NEXT:               %[[VL_1:.*]] = agen.vector_load %[[LMV_1]]{{\[}}%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_2]]} : memref<12x1x64x64xf16>, vector<49152xf16>
// CHECK-NEXT:               dataflow.send %[[SFU_TGT_1]], %[[VL_1]] : vector<49152xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG6:.*]] -> (%[[SFU_0]], %[[SFU_1]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG7:.*]] -> (%[[L1SU_0]], %[[L1SU_1]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[ARG8:.*]] -> (%[[L1SU_0]], %[[L1SU_1]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG9:.*]] -> (%[[L1LU_0]], %[[L1LU_1]]) : {
// CHECK-NEXT:           %[[MAP_L1_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU_0]] -> %[[L1_0]]], {{\[}}%[[L1LU_1]] -> %[[L1_1]]]):index
// CHECK-NEXT:           %[[L1_TGT_2:.*]] = uniform.query_map(map:%[[MAP_L1_2]], key:%[[ARG9]]) : index
// CHECK-NEXT:           %[[LMV_2:.*]] = dataflow.get_logical_memory_view %[[L1_TGT_2]], %[[C128]] {layout_map = #[[$ATTR_0]]} : index, index, memref<12x1x64x64xf16>
// CHECK-NEXT:           scf.for %[[I2:.*]] = %[[C0]] to %[[C12]] step %[[C1]] {
// CHECK-NEXT:             scf.for %[[J2:.*]] = %[[C0]] to %[[C64]] step %[[C1]] {
// CHECK-NEXT:               %[[MAP_SFU_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU_0]] -> %[[SFU_0]]], {{\[}}%[[L1LU_1]] -> %[[SFU_1]]]):index
// CHECK-NEXT:               %[[SFU_TGT_2:.*]] = uniform.query_map(map:%[[MAP_SFU_2]], key:%[[ARG9]]) : index
// CHECK-NEXT:               %[[VL_2:.*]] = agen.vector_load %[[LMV_2]]{{\[}}%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {load_order = #[[$ATTR_1]], load_set = #[[$ATTR_2]]} : memref<12x1x64x64xf16>, vector<49152xf16>
// CHECK-NEXT:               dataflow.send %[[SFU_TGT_2]], %[[VL_2]] : vector<49152xf16>
// CHECK-NEXT:             } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG10:.*]] -> (%[[SFU_0]], %[[SFU_1]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG11:.*]] -> (%[[L1SU_0]], %[[L1SU_1]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }






//
// Test: Source B — unrealized_conversion_cast from index to L1 memref is
// lowered to dataflow.get_logical_memory_view. Checks:
//   - L1 get_unit ops emitted at func level (one per core)
//   - uniform.def_immutable_mapping + uniform.query_map emitted inside program_unit
//   - index operand of cast becomes start_address
//   - layout_map is row-major linearization from static shape
//   - result type is plain memref (no memory space)
//   - data_transfer operand type updated to plain memref
//   - original unrealized_conversion_cast is gone



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
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        // Source B: index -> L1 memref cast with non-dealloc users
        %l1_view = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        %fifo:1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        scf.for %arg1 = %c0 to %c12 step %c1 {
          scf.for %arg2 = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %l1_view[%c0, %c0, %c0, %c0] size [12, 1, 64, 64] to %fifo#0 size [12, 1, 64, 64] : memref<12x1x64x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1_view : memref<12x1x64x64xf16, "L1">
      }
      dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
      }
      return
    }
  }
}
