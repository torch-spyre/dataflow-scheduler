// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
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
// CHECK-NEXT:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[SFU_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:       %[[SFU_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:       %[[L1LU_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:       %[[L1LU_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:       %[[L1SU_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
// CHECK-NEXT:       %[[L1SU_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[ARG0:.*]] -> (%[[L1LU_0]], %[[L1LU_1]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG1:.*]] -> (%[[L1LU_0]], %[[L1LU_1]]) : {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:           %[[MAP_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU_0]] -> %[[SFU_0]]], {{\[}}%[[L1LU_1]] -> %[[SFU_1]]]):index
// CHECK-NEXT:           %[[TGT_0:.*]] = uniform.query_map(map:%[[MAP_0]], key:%[[ARG1]]) : index
// CHECK-NEXT:           %[[VL_0:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[C0]]] {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_1]]} : memref<64xf16, "L1">, vector<64xf16>
// CHECK-NEXT:           dataflow.send %[[TGT_0]], %[[VL_0]] : vector<64xf16>
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG2:.*]] -> (%[[SFU_0]], %[[SFU_1]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG3:.*]] -> (%[[L1SU_0]], %[[L1SU_1]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[ARG4:.*]] -> (%[[SFU_0]], %[[SFU_1]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG5:.*]] -> (%[[L1LU_0]], %[[L1LU_1]]) : {
// CHECK-NEXT:           %[[ALLOC_1:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:           %[[MAP_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU_0]] -> %[[SFU_0]]], {{\[}}%[[L1LU_1]] -> %[[SFU_1]]]):index
// CHECK-NEXT:           %[[TGT_1:.*]] = uniform.query_map(map:%[[MAP_1]], key:%[[ARG5]]) : index
// CHECK-NEXT:           %[[VL_1:.*]] = agen.vector_load %[[ALLOC_1]]{{\[}}%[[C0]]] {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_1]]} : memref<64xf16, "L1">, vector<64xf16>
// CHECK-NEXT:           dataflow.send %[[TGT_1]], %[[VL_1]] : vector<64xf16>
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG6:.*]] -> (%[[SFU_0]], %[[SFU_1]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG7:.*]] -> (%[[L1SU_0]], %[[L1SU_1]]) : {
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       dataflow.program_unit iter_arg : %[[ARG8:.*]] -> (%[[L1SU_0]], %[[L1SU_1]]) : {
// CHECK-NEXT:         dataflow.program_unit iter_arg : %[[ARG9:.*]] -> (%[[L1LU_0]], %[[L1LU_1]]) : {
// CHECK-NEXT:           %[[ALLOC_2:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:           %[[MAP_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[L1LU_0]] -> %[[SFU_0]]], {{\[}}%[[L1LU_1]] -> %[[SFU_1]]]):index
// CHECK-NEXT:           %[[TGT_2:.*]] = uniform.query_map(map:%[[MAP_2]], key:%[[ARG9]]) : index
// CHECK-NEXT:           %[[VL_2:.*]] = agen.vector_load %[[ALLOC_2]]{{\[}}%[[C0]]] {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_1]]} : memref<64xf16, "L1">, vector<64xf16>
// CHECK-NEXT:           dataflow.send %[[TGT_2]], %[[VL_2]] : vector<64xf16>
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
// Test: Source B pruning — if all uses of unrealized_conversion_cast are
// memref.dealloc, both the cast and the dealloc are deleted without emitting
// a get_logical_memory_view. Checks:
//   - No get_logical_memory_view emitted for pruned casts
//   - No memref.dealloc remains for the pruned casts
//   - No unrealized_conversion_cast remains
//   - program_unit body otherwise intact



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
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %c128 = arith.constant 128 : index
      %c98432 = arith.constant 98432 : index
      dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
        // Source B casts with ONLY dealloc users — should be pruned entirely
        %l10 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        %l11 = builtin.unrealized_conversion_cast %c98432 : index to memref<12x1x64x64xf16, "L1">
        %c0 = arith.constant 0 : index
        %l1 = memref.alloc() : memref<64xf16, "L1">
        %fifo:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        ktdf.data_transfer from %l1[%c0] size [64] to %fifo#0 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        memref.dealloc %l10 : memref<12x1x64x64xf16, "L1">
        memref.dealloc %l11 : memref<12x1x64x64xf16, "L1">
      }
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
      }
      return
    }
  }
}
