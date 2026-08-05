// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// memref <-> FIFO transfers wider than the hardware vector width (64 f16 lanes
// on sample_device) are NOT split, unlike the memref-to-memref path. They lower
// to a single wide vector_load/send or receive/vector_store.
//
// This is deliberate, not an oversight: a transfer wider than the FIFO slot may
// be intentional streaming semantics (push N times through a narrow slot), which
// is the shape ReductionLoopExposure produces in reverse when it shrinks a FIFO
// and adds a loop. There is no device-verified target for splitting a
// memref<->FIFO transfer across AGEN time dimensions, so the current
// unsplit behaviour is pinned here rather than changed. Changing it should be a
// deliberate, visible diff to this file, not an incidental side effect of
// touching the memref-to-memref splitting logic in DataTransferLowering.cpp.
//
// See also data-transfer-exceeds-vector-width.mlir, data-transfer-split-innermost.mlir
// and data-transfer-three-time-dims.mlir for the memref-to-memref path that IS
// split, and data-transfer-vector-width-error.mlir for the shapes that path
// rejects.
//
// Both cases live in one module (no -split-input-file) so that
// generate-test-checks.py can produce checks over a single output stream.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>


// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @load_and_send_exceeds_vector_width() attributes {grid = [2]} {
// CHECK-NEXT:     %[[VAL_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     %[[VAL_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     %[[VAL_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:     %[[VAL_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_4:.*]] -> (%[[VAL_0]], %[[VAL_1]]) : {
// CHECK-NEXT:       %[[VAL_5:.*]] = memref.alloc() : memref<128x64xf16, "L1">
// CHECK-NEXT:       %[[VAL_6:.*]] = agen.vector_load %[[VAL_5]][0, 0] {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_1]]} : memref<128x64xf16, "L1">, vector<8192xf16>
// CHECK-NEXT:       %[[VAL_7:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_0]] -> %[[VAL_2]]], {{\[}}%[[VAL_1]] -> %[[VAL_3]]]):index
// CHECK-NEXT:       %[[VAL_8:.*]] = uniform.query_map(map:%[[VAL_7]], key:%[[VAL_4]]) : index
// CHECK-NEXT:       dataflow.send %[[VAL_8]], %[[VAL_6]] : vector<8192xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_9:.*]] -> (%[[VAL_2]], %[[VAL_3]]) : {
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @receive_and_store_exceeds_vector_width() attributes {grid = [2]} {
// CHECK-NEXT:     %[[VAL_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     %[[VAL_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     %[[VAL_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:     %[[VAL_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_4:.*]] -> (%[[VAL_2]], %[[VAL_3]]) : {
// CHECK-NEXT:       %[[VAL_5:.*]] = memref.alloc() : memref<128x64xf16, "L1">
// CHECK-NEXT:       %[[VAL_6:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_2]] -> %[[VAL_0]]], {{\[}}%[[VAL_3]] -> %[[VAL_1]]]):index
// CHECK-NEXT:       %[[VAL_7:.*]] = uniform.query_map(map:%[[VAL_6]], key:%[[VAL_4]]) : index
// CHECK-NEXT:       %[[VAL_8:.*]] = dataflow.receive %[[VAL_7]] : vector<8192xf16>
// CHECK-NEXT:       agen.vector_store %[[VAL_8]], %[[VAL_5]][0, 0] {store_order = #[[$ATTR_0]], store_set = #[[$ATTR_1]]} : memref<128x64xf16, "L1">, vector<8192xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_9:.*]] -> (%[[VAL_0]], %[[VAL_1]]) : {
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }


module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // memref -> FIFO (kLoadAndSend): 128*64 = 8192 elements, emitted un-split.
  func.func @load_and_send_exceeds_vector_width() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %2 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %3 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %u_l1lu = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu = uniform.def_immutable_mapping([%c0 -> %2], [%c1 -> %3]):index
    %u_sfu = uniform.query_map(map:%map_sfu, key:%tile_id) : index
    ktdf_lowering.execute_on %u_l1lu, %u_sfu {
      %alloc = memref.alloc() : memref<128x64xf16, "L1">
      %fifo:1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>
      ktdf_lowering.execute_on %u_l1lu {
        ktdf.data_transfer from %alloc[0, 0] size [128, 64]
                           to %fifo#0 size [8192]
          : memref<128x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>
      }
    }
    return
  }

  // FIFO -> memref (kReceiveAndStore): 8192 elements, emitted un-split.
  func.func @receive_and_store_exceeds_vector_width() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %2 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
    %3 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
    %tile_id = ktdp.get_compute_tile_id : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %u_l1lu = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu = uniform.def_immutable_mapping([%c0 -> %2], [%c1 -> %3]):index
    %u_sfu = uniform.query_map(map:%map_sfu, key:%tile_id) : index
    ktdf_lowering.execute_on %u_l1lu, %u_sfu {
      %alloc = memref.alloc() : memref<128x64xf16, "L1">
      %fifo:1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>
      ktdf_lowering.execute_on %u_sfu {
        ktdf.data_transfer from %fifo#0 size [8192]
                           to %alloc[0, 0] size [128, 64]
          : !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>, memref<128x64xf16, "L1">
      }
    }
    return
  }
}
