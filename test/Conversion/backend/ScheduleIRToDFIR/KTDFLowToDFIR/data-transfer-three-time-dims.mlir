// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// There is no cap on the number of AGEN time dimensions a split transfer can
// use: a transfer walking three non-unit outer dims (2, 4 and 8) succeeds with
// a 3-dim time_set, same as the 1-dim and 2-dim cases exercised elsewhere.
// The hardware vector width here is 64 f16 lanes (sample_device), and the
// innermost dim (64) is exactly one vector, so it contributes no extra time
// dim -- only the three outer dims become time dims.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2, 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_3:.+]] = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 == 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 7 >= 0)>

// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @three_time_dims() attributes {grid = [2]} {
// CHECK-NEXT:     %[[VAL_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[VAL_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_2:.*]] -> (%[[VAL_0]], %[[VAL_1]]) : {
// CHECK-NEXT:       %[[VAL_3:.*]] = memref.alloc() : memref<2x4x8x64xf16, "DDR">
// CHECK-NEXT:       %[[VAL_4:.*]] = memref.alloc() : memref<2x4x8x64xf16, "L1">
// CHECK-NEXT:       agen.composite_load_and_store src:%[[VAL_3]][0, 0, 0, 0] dst:%[[VAL_4]][0, 0, 0, 0]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_5:.*]]:vector<64xf16>)
// CHECK-NEXT:        {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_3]], load_time_addr_map = #[[$ATTR_1]], store_order = #[[$ATTR_0]], store_set = #[[$ATTR_3]], store_time_addr_map = #[[$ATTR_1]], time_order = #[[$ATTR_2]], time_set = #[[$ATTR_4]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<2x4x8x64xf16, "DDR">, memref<2x4x8x64xf16, "L1">
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @three_time_dims() attributes {grid = [2]} {
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
}
