// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// An AGEN composite transfer moves one hardware vector per time step, so a
// transfer wider than that walks the remaining elements over AGEN time
// dimensions instead of widening load_iv. This file exercises which shapes
// get split that way (kLoadAndStore, memref-to-memref) and which do not
// (kLoadAndSend / kReceiveAndStore, memref-to-FIFO) -- both live in one
// module (no -split-input-file) so generate-test-checks.py sees a single
// output stream.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (0, d0, 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0) -> (0, 0, d0, 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0) -> (0, 0, d0 * 64)>
// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2, 0)>
// CHECK: #[[$ATTR_7:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_8:.+]] = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_9:.+]] = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 == 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$ATTR_10:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 255 >= 0)>
// CHECK: #[[$ATTR_11:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>
// CHECK: #[[$ATTR_12:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 7 >= 0)>
// CHECK: #[[$ATTR_13:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 127 >= 0, d1 >= 0, -d1 + 63 >= 0)>

// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // A kLoadAndStore transfer of 1*256*64 = 16384 elements: the single non-unit
  // outer dim (256) becomes a one-dim time walk; the innermost dim (64) is
  // exactly one vector wide and contributes no time dim of its own.
  // CHECK-LABEL:   func.func @one_walked_dim() attributes {grid = [2]} {
  // CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
  // CHECK:           %[[VAL_1:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
  // CHECK:           %[[VAL_2:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
  // CHECK:           dataflow.program_unit iter_arg : %[[VAL_3:.*]] -> (%[[VAL_1]], %[[VAL_2]]) : {
  // CHECK:             %[[VAL_4:.*]] = memref.alloc() : memref<2x256x64xf16, "DDR">
  // CHECK:             %[[VAL_5:.*]] = memref.alloc() : memref<2x1x256x64xf16, "L1">
  // CHECK:             agen.composite_load_and_store src:%[[VAL_4]]{{\[}}%[[VAL_0]], 0, 0] dst:%[[VAL_5]]{{\[}}%[[VAL_0]], 0, 0, 0]
  // CHECK:              time_symbols(), load_iv(%[[VAL_6:.*]]:vector<64xf16>)
  // CHECK:              {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_8]], load_time_addr_map = #[[$ATTR_1]], store_order = #[[$ATTR_2]], store_set = #[[$ATTR_9]], store_time_addr_map = #[[$ATTR_3]], time_order = #[[$ATTR_4]], time_set = #[[$ATTR_10]]}
  // CHECK:             {
  // CHECK:               agen.yield
  // CHECK:             } : memref<2x256x64xf16, "DDR">, memref<2x1x256x64xf16, "L1">
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
  func.func @one_walked_dim() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index

    // Source: rank-3 DDR buffer holding 2 sticks x 256 rows x 64 lanes.
    %ddr_buf = memref.alloc() : memref<2x256x64xf16, "DDR">
    // Destination: rank-4 L1 buffer (stage-coarsened, leading stick dim).
    %l1_buf = memref.alloc() : memref<2x1x256x64xf16, "L1">

    ktdf_lowering.execute_on %unit {
      ktdf_lowering.execute_on %unit {
        // 1*256*64 = 16384 elements = 256 vectors of 64 lanes.
        ktdf.data_transfer from %ddr_buf[%c0, 0, 0] size [1, 256, 64]
                           to %l1_buf[%c0, 0, 0, 0] size [1, 1, 256, 64]
          : memref<2x256x64xf16, "DDR">, memref<2x1x256x64xf16, "L1">
      }
    }
    return
  }

  // A transfer whose only non-unit dim is the innermost one still splits:
  // since the innermost size (128) is more than one vector, the innermost
  // dim itself becomes the (sole) time dimension, with coefficient equal to
  // the vector width rather than 1.
  // CHECK-LABEL:   func.func @split_innermost() attributes {grid = [2]} {
  // CHECK:           %[[VAL_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
  // CHECK:           %[[VAL_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
  // CHECK:           dataflow.program_unit iter_arg : %[[VAL_2:.*]] -> (%[[VAL_0]], %[[VAL_1]]) : {
  // CHECK:             %[[VAL_3:.*]] = memref.alloc() : memref<1x1x128xf16, "DDR">
  // CHECK:             %[[VAL_4:.*]] = memref.alloc() : memref<1x1x128xf16, "L1">
  // CHECK:             agen.composite_load_and_store src:%[[VAL_3]][0, 0, 0] dst:%[[VAL_4]][0, 0, 0]
  // CHECK:              time_symbols(), load_iv(%[[VAL_5:.*]]:vector<64xf16>)
  // CHECK:              {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_8]], load_time_addr_map = #[[$ATTR_5]], store_order = #[[$ATTR_0]], store_set = #[[$ATTR_8]], store_time_addr_map = #[[$ATTR_5]], time_order = #[[$ATTR_4]], time_set = #[[$ATTR_11]]}
  // CHECK:             {
  // CHECK:               agen.yield
  // CHECK:             } : memref<1x1x128xf16, "DDR">, memref<1x1x128xf16, "L1">
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
  func.func @split_innermost() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    %ddr_buf = memref.alloc() : memref<1x1x128xf16, "DDR">
    %l1_buf = memref.alloc() : memref<1x1x128xf16, "L1">
    ktdf_lowering.execute_on %unit {
      ktdf.data_transfer from %ddr_buf[0, 0, 0] size [1, 1, 128]
                         to %l1_buf[0, 0, 0] size [1, 1, 128]
        : memref<1x1x128xf16, "DDR">, memref<1x1x128xf16, "L1">
    }
    return
  }

  // There is no cap on the number of AGEN time dimensions a split transfer
  // can use: a transfer walking three non-unit outer dims (2, 4 and 8)
  // succeeds with a 3-dim time_set, same as the 1-dim and 2-dim cases
  // exercised elsewhere in this file. The innermost dim (64) is exactly one
  // vector, so it contributes no extra time dim.
  // CHECK-LABEL:   func.func @three_walked_dims() attributes {grid = [2]} {
  // CHECK:           %[[VAL_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
  // CHECK:           %[[VAL_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
  // CHECK:           dataflow.program_unit iter_arg : %[[VAL_2:.*]] -> (%[[VAL_0]], %[[VAL_1]]) : {
  // CHECK:             %[[VAL_3:.*]] = memref.alloc() : memref<2x4x8x64xf16, "DDR">
  // CHECK:             %[[VAL_4:.*]] = memref.alloc() : memref<2x4x8x64xf16, "L1">
  // CHECK:             agen.composite_load_and_store src:%[[VAL_3]][0, 0, 0, 0] dst:%[[VAL_4]][0, 0, 0, 0]
  // CHECK:              time_symbols(), load_iv(%[[VAL_5:.*]]:vector<64xf16>)
  // CHECK:              {load_order = #[[$ATTR_2]], load_set = #[[$ATTR_9]], load_time_addr_map = #[[$ATTR_6]], store_order = #[[$ATTR_2]], store_set = #[[$ATTR_9]], store_time_addr_map = #[[$ATTR_6]], time_order = #[[$ATTR_0]], time_set = #[[$ATTR_12]]}
  // CHECK:             {
  // CHECK:               agen.yield
  // CHECK:             } : memref<2x4x8x64xf16, "DDR">, memref<2x4x8x64xf16, "L1">
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
  func.func @three_walked_dims() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    %ddr_buf = memref.alloc() : memref<2x4x8x64xf16, "DDR">
    %l1_buf = memref.alloc() : memref<2x4x8x64xf16, "L1">
    ktdf_lowering.execute_on %unit {
      ktdf.data_transfer from %ddr_buf[0, 0, 0, 0] size [2, 4, 8, 64]
                         to %l1_buf[0, 0, 0, 0] size [2, 4, 8, 64]
        : memref<2x4x8x64xf16, "DDR">, memref<2x4x8x64xf16, "L1">
    }
    return
  }

  // memref -> FIFO (kLoadAndSend) transfers wider than the hardware vector
  // width are NOT split, unlike the memref-to-memref path above: they lower
  // to a single wide vector_load/send. This is deliberate: a transfer wider
  // than the FIFO slot may denote repeated pushes through a narrow slot -- a
  // stream length -- rather than one wide spatial access, and the two
  // readings are not distinguishable from the op alone. Current behaviour is
  // pinned by this test rather than changed; changing it should be a
  // deliberate, visible diff to this file, not an incidental side effect of
  // touching the memref-to-memref splitting logic in DataTransferLowering.cpp.
  // CHECK-LABEL:   func.func @load_and_send_unsplit() attributes {grid = [2]} {
  // CHECK:           %[[VAL_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
  // CHECK:           %[[VAL_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
  // CHECK:           %[[VAL_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  // CHECK:           %[[VAL_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
  // CHECK:           dataflow.program_unit iter_arg : %[[VAL_4:.*]] -> (%[[VAL_0]], %[[VAL_1]]) : {
  // CHECK:             %[[VAL_5:.*]] = memref.alloc() : memref<128x64xf16, "L1">
  // CHECK:             %[[VAL_6:.*]] = agen.vector_load %[[VAL_5]][0, 0] {load_order = #[[$ATTR_7]], load_set = #[[$ATTR_13]]} : memref<128x64xf16, "L1">, vector<8192xf16>
  // CHECK:             %[[VAL_7:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_0]] -> %[[VAL_2]]], {{\[}}%[[VAL_1]] -> %[[VAL_3]]]):index
  // CHECK:             %[[VAL_8:.*]] = uniform.query_map(map:%[[VAL_7]], key:%[[VAL_4]]) : index
  // CHECK:             dataflow.send %[[VAL_8]], %[[VAL_6]] : vector<8192xf16>
  // CHECK:           }
  // CHECK:           dataflow.program_unit iter_arg : %[[VAL_9:.*]] -> (%[[VAL_2]], %[[VAL_3]]) : {
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
  func.func @load_and_send_unsplit() attributes {grid = [2]} {
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

  // FIFO -> memref (kReceiveAndStore): the same unsplit behaviour in the
  // opposite direction, lowering to a single receive/vector_store.
  // CHECK-LABEL:   func.func @receive_and_store_unsplit() attributes {grid = [2]} {
  // CHECK:           %[[VAL_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
  // CHECK:           %[[VAL_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
  // CHECK:           %[[VAL_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
  // CHECK:           %[[VAL_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
  // CHECK:           dataflow.program_unit iter_arg : %[[VAL_4:.*]] -> (%[[VAL_2]], %[[VAL_3]]) : {
  // CHECK:             %[[VAL_5:.*]] = memref.alloc() : memref<128x64xf16, "L1">
  // CHECK:             %[[VAL_6:.*]] = uniform.def_immutable_mapping({{\[}}%[[VAL_2]] -> %[[VAL_0]]], {{\[}}%[[VAL_3]] -> %[[VAL_1]]]):index
  // CHECK:             %[[VAL_7:.*]] = uniform.query_map(map:%[[VAL_6]], key:%[[VAL_4]]) : index
  // CHECK:             %[[VAL_8:.*]] = dataflow.receive %[[VAL_7]] : vector<8192xf16>
  // CHECK:             agen.vector_store %[[VAL_8]], %[[VAL_5]][0, 0] {store_order = #[[$ATTR_7]], store_set = #[[$ATTR_13]]} : memref<128x64xf16, "L1">, vector<8192xf16>
  // CHECK:           }
  // CHECK:           dataflow.program_unit iter_arg : %[[VAL_9:.*]] -> (%[[VAL_0]], %[[VAL_1]]) : {
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
  func.func @receive_and_store_unsplit() attributes {grid = [2]} {
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
