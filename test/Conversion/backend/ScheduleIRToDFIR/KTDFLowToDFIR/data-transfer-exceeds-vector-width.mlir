// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// A kLoadAndStore ktdf.data_transfer moving 1*256*64 = 16384 elements DDR -> L1.
// The hardware vector width is 64 f16 lanes (sample_device declares
// lanes = <f16 = 64>), so this request is 256x wider than one vector.
//
// An AGEN composite transfer moves one hardware vector per time step, so the
// count is moved onto the AGEN time axis: the single non-unit outer dim (256)
// becomes a time dim bounded by its extent, contributing an offset at that
// dim's position in load_time_addr_map / store_time_addr_map on BOTH sides.
// Those maps are added to the base address, so the base indices and the
// access maps are untouched. The load_iv narrows to the vector width and the
// load/store sets cover exactly one vector. Everything stays a single AGEN
// instruction -- no enclosing loop.
//
// The innermost dim (64) is exactly one vector width, so it contributes no
// extra time dim; only the outer dim becomes a time dim. Shapes whose
// innermost dim is not a multiple of the vector width are rejected with a
// diagnostic (see data-transfer-vector-width-error.mlir); shapes needing more
// time dims are not capped (see data-transfer-three-time-dims.mlir).

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (0, d0, 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0) -> (0, 0, d0, 0)>
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_6:.+]] = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 == 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$ATTR_7:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 255 >= 0)>

// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @bulk_transfer_exceeds_vector_width() attributes {grid = [2]} {
// CHECK-NEXT:     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_1:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     %[[VAL_2:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_3:.*]] -> (%[[VAL_1]], %[[VAL_2]]) : {
// CHECK-NEXT:       %[[VAL_4:.*]] = memref.alloc() : memref<2x256x64xf16, "DDR">
// CHECK-NEXT:       %[[VAL_5:.*]] = memref.alloc() : memref<2x1x256x64xf16, "L1">
// CHECK-NEXT:       agen.composite_load_and_store src:%[[VAL_4]]{{\[}}%[[VAL_0]], 0, 0] dst:%[[VAL_5]]{{\[}}%[[VAL_0]], 0, 0, 0]
// CHECK-NEXT:        time_symbols(), load_iv(%[[VAL_6:.*]]:vector<64xf16>)
// CHECK-NEXT:        {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_5]], load_time_addr_map = #[[$ATTR_1]], store_order = #[[$ATTR_2]], store_set = #[[$ATTR_6]], store_time_addr_map = #[[$ATTR_3]], time_order = #[[$ATTR_4]], time_set = #[[$ATTR_7]]}
// CHECK-NEXT:       {
// CHECK-NEXT:         agen.yield
// CHECK-NEXT:       } : memref<2x256x64xf16, "DDR">, memref<2x1x256x64xf16, "L1">
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @bulk_transfer_exceeds_vector_width() attributes {grid = [2]} {
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
        // 1*256*64 = 16384 elements = 256 vectors of 64 f16.
        ktdf.data_transfer from %ddr_buf[%c0, 0, 0] size [1, 256, 64]
                           to %l1_buf[%c0, 0, 0, 0] size [1, 1, 256, 64]
          : memref<2x256x64xf16, "DDR">, memref<2x1x256x64xf16, "L1">
      }
    }
    return
  }
}
