// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -allow-unregistered-dialect %s | FileCheck %s

// This script is intended to make adding checks to a test case quick and easy.
// It is *not* authoritative about what constitutes a good test. After using the
// script, be sure to review and refine the generated checks. For example,
// CHECK lines should be minimized and named to reflect the test’s intent.
// For comprehensive guidelines, see:
//   * https://mlir.llvm.org/getting_started/TestingGuide/



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK-LABEL:   func.func @nested_pipelines(
// CHECK-SAME:      %[[ARG0:.*]]: memref<?xf16, "DDR">,
// CHECK-SAME:      %[[ARG1:.*]]: index,
// CHECK-SAME:      %[[ARG2:.*]]: index) attributes {grid = [2]} {
// CHECK-DAG:      %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK-DAG:      %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK-DAG:      %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-DAG:      %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-DAG:      %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK-DAG:      %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK:          scf.for %[[VAL_1:.*]] = %[[CONSTANT_1]] to %[[ARG1]] step %[[CONSTANT_0]] {
// CHECK-DAG:          %[[ALLOC_0:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:         agen.composite_load_and_store src:%[[ARG0]]{{\[}}%[[VAL_1]]] dst:%[[ALLOC_0]]{{\[}}%[[CONSTANT_1]]]
// CHECK-NEXT:          time_symbols(), load_iv(%[[VAL_2:.*]]:vector<64xf16>)
// CHECK-NEXT:          {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_2]], load_time_addr_map = #[[$ATTR_1]], store_order = #[[$ATTR_0]], store_set = #[[$ATTR_2]], store_time_addr_map = #[[$ATTR_1]], time_order = #[[$ATTR_0]], time_set = #[[$ATTR_3]]}
// CHECK-NEXT:         {
// CHECK-NEXT:           agen.yield
// CHECK-NEXT:         } : memref<?xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_3:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK:          scf.for %[[VAL_4:.*]] = %[[CONSTANT_1]] to %[[ARG1]] step %[[CONSTANT_0]] {
// CHECK-DAG:          %[[ALLOC_1:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK:              scf.for %[[VAL_5:.*]] = %[[CONSTANT_1]] to %[[ARG2]] step %[[CONSTANT_0]] {
// CHECK:                "test.use"(%[[ALLOC_1]]) : (memref<64xf16, "L1">) -> ()
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// Two sibling pipelines using overlapping component types. Each program_unit
// receives a self-contained clone of the work region with type-irrelevant
// inner ktdf_lowering.execute_on ops removed.


module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @nested_pipelines(%arg0: memref<?xf16, "DDR">, %arg1: index, %arg2: index) attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %4 = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %5 = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %6 = uniform.query_map(map:%5, key:%4) : index
    %c0_0 = arith.constant 0 : index
    %c1_1 = arith.constant 1 : index
    %7 = uniform.def_immutable_mapping([%c0_0 -> %2], [%c1_1 -> %3]):index
    %8 = uniform.query_map(map:%7, key:%4) : index
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    ktdf_lowering.execute_on %6, %8 {
      %9 = ktdf.create_token : !ktdf.token
      ktdf_lowering.execute_on %6, %8 {
        scf.for %arg3 = %c0_2 to %arg1 step %c1_3 {
          %alloc = memref.alloc() : memref<64xf16, "L1">
          ktdf_lowering.execute_on %6 {
            ktdf_lowering.execute_on %6 {
              ktdf.data_transfer from %arg0[%arg3] size [64] to %alloc[%c0_2] size [64] {dataflow_scheduler.access_granularity = 64 : i64} : memref<?xf16, "DDR">, memref<64xf16, "L1">
            }
          }
          scf.for %arg4 = %c0_2 to %arg2 step %c1_3 {
            ktdf_lowering.execute_on %8 {
              %alloc_4 = memref.alloc() : memref<64xf16, "L1">
              %10 = ktdf.create_token : !ktdf.token
              ktdf_lowering.execute_on %8 {
                "test.use"(%alloc) : (memref<64xf16, "L1">) -> ()
              }
            }
          }
        }
      }
    }
    return
  }
}

