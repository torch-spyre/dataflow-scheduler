// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// A FIFO endpoint is resolved to a concrete target unit by matching candidate
// units on their `core` attribute. On a core with several corelets, every
// corelet's unit of the required type is a candidate, so the first candidate won
// for every key and corelet 1's FIFO resolved to corelet 0's destination.
// Each source corelet must map to the destination unit on its OWN corelet.

// CHECK-LABEL: func.func @fifo_endpoint_per_corelet
// CHECK-DAG:     %[[LU_CL0:.+]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-L1LU-CL0"
// CHECK-DAG:     %[[LU_CL1:.+]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-L1LU-CL1"
// CHECK-DAG:     %[[SFU_CL0:.+]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-SFU-CL0"
// CHECK-DAG:     %[[SFU_CL1:.+]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-SFU-CL1"
// The L1LU -> SFU FIFO endpoint map must pair corelet 0 with corelet 0 and
// corelet 1 with corelet 1.
// CHECK:         uniform.def_immutable_mapping([%[[LU_CL0]] -> %[[SFU_CL0]]], [%[[LU_CL1]] -> %[[SFU_CL1]]])

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @fifo_endpoint_per_corelet() attributes {grid = [1]} {
    %l1lu_cl0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-L1LU-CL0", type = "L1LU"} : index
    %l1lu_cl1 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-L1LU-CL1", type = "L1LU"} : index
    %sfu_cl0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-SFU-CL0", type = "SFU"} : index
    %sfu_cl1 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-SFU-CL1", type = "SFU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index

    // grid = [1]: one core, so each per-corelet map has a single key.
    %map_lu_cl0 = uniform.def_immutable_mapping([%c0 -> %l1lu_cl0]):index
    %u_lu_cl0 = uniform.query_map(map:%map_lu_cl0, key:%tile_id) : index
    %map_lu_cl1 = uniform.def_immutable_mapping([%c0 -> %l1lu_cl1]):index
    %u_lu_cl1 = uniform.query_map(map:%map_lu_cl1, key:%tile_id) : index
    %map_sfu_cl0 = uniform.def_immutable_mapping([%c0 -> %sfu_cl0]):index
    %u_sfu_cl0 = uniform.query_map(map:%map_sfu_cl0, key:%tile_id) : index
    %map_sfu_cl1 = uniform.def_immutable_mapping([%c0 -> %sfu_cl1]):index
    %u_sfu_cl1 = uniform.query_map(map:%map_sfu_cl1, key:%tile_id) : index

    ktdf_lowering.execute_on %u_lu_cl0, %u_lu_cl1, %u_sfu_cl0, %u_sfu_cl1 {
      %alloc = memref.alloc() : memref<128xf16, "L1">
      %fifo:1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      ktdf_lowering.execute_on %u_lu_cl0, %u_lu_cl1 {
        ktdf.data_transfer from %alloc[%c0] size [64]
                           to %fifo#0 size [64]
            : memref<128xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      }
    }
    return
  }
}
