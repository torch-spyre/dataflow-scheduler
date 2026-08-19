// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// A time dimension of an AGEN composite transfer is one step count shared by
// both sides, so it can only describe a walk in which both sides move the same
// distance per step. Matching extents are not enough: the sides may reach the
// same walk through different shapes and layouts, and one count cannot stand
// for two different strides.
//
// Such a dimension is instead walked by an enclosing loop, whose induction
// variable enters the memref subscripts on both sides, where each side applies
// its own layout. What is left on the time axis is a single pinned step with
// all-zero offsets.

// CHECK: #[[$LOAD_ORDER:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$PINNED_LOAD_ADDR:.+]] = affine_map<(d0) -> (0, 0, 0)>
// CHECK: #[[$STORE_ORDER:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$PINNED_STORE_ADDR:.+]] = affine_map<(d0) -> (0, 0, 0, 0)>
// CHECK: #[[$TIME_ORDER:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$WALKED_LOAD_ADDR:.+]] = affine_map<(d0) -> (0, d0, 0)>
// CHECK: #[[$WALKED_STORE_ADDR:.+]] = affine_map<(d0) -> (0, 0, d0, 0)>
// CHECK: #[[$LOAD_SET:.+]] = affine_set<(d0, d1, d2) : (d0 == 0, d1 == 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$STORE_SET:.+]] = affine_set<(d0, d1, d2, d3) : (d0 == 0, d1 == 0, d2 == 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$ONE_STEP:.+]] = affine_set<(d0) : (d0 == 0)>
// CHECK: #[[$TIME_256:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 255 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  // Stepping the source's leading dimension advances 256*64 = 16384 elements;
  // stepping the destination dimension of matching extent advances 1*64 = 64.
  // The two distances differ, so the dimension leaves the time axis for a
  // 2-iteration loop and the descriptor is pinned to a single step with
  // all-zero offsets on both sides.
  // CHECK-LABEL:   func.func @divergent_stride_becomes_a_loop() attributes {grid = [2]} {
  // CHECK:           %[[C1:.*]] = arith.constant 1 : index
  // CHECK:           %[[C2:.*]] = arith.constant 2 : index
  // CHECK:           %[[C0:.*]] = arith.constant 0 : index
  // CHECK:           dataflow.program_unit iter_arg : %{{.*}} -> (%{{.*}}, %{{.*}}) : {
  // CHECK:             %[[DDR:.*]] = memref.alloc() : memref<2x256x64xf16, "DDR">
  // CHECK:             %[[L1:.*]] = memref.alloc() : memref<256x2x1x64xf16, "L1">
  // CHECK:             scf.for %[[STICK:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
  // CHECK:               agen.composite_load_and_store src:%[[DDR]]{{\[}}%[[STICK]], %[[C0]], 0] dst:%[[L1]]{{\[}}%[[C0]], %[[STICK]], 0, 0]
  // CHECK:                time_symbols(), load_iv(%{{.*}}:vector<64xf16>)
  // CHECK:                {load_order = #[[$LOAD_ORDER]], load_set = #[[$LOAD_SET]], load_time_addr_map = #[[$PINNED_LOAD_ADDR]], store_order = #[[$STORE_ORDER]], store_set = #[[$STORE_SET]], store_time_addr_map = #[[$PINNED_STORE_ADDR]], time_order = #[[$TIME_ORDER]], time_set = #[[$ONE_STEP]]}
  // CHECK:             }
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
  func.func @divergent_stride_becomes_a_loop() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index

    // Source: 2 sticks of 256 rows x 64 lanes. One stick is 16384 elements.
    %ddr_buf = memref.alloc() : memref<2x256x64xf16, "DDR">
    // Destination: one row of both sticks, packed adjacently. One stick is
    // 64 elements.
    %l1_buf = memref.alloc() : memref<256x2x1x64xf16, "L1">

    ktdf_lowering.execute_on %unit {
      ktdf_lowering.execute_on %unit {
        // 2*1*64 = 128 elements = 2 vectors of 64 lanes.
        ktdf.data_transfer from %ddr_buf[0, %c0, 0] size [2, 1, 64]
                           to %l1_buf[%c0, 0, 0, 0] size [1, 2, 1, 64]
          : memref<2x256x64xf16, "DDR">, memref<256x2x1x64xf16, "L1">
      }
    }
    return
  }

  // The same 256-step walk on both sides: the source steps dimension 1 of a
  // 2x256x64 buffer and the destination dimension 2 of a 2x1x256x64 buffer,
  // both advancing 64 elements per step. One count does stand for both
  // strides, so the walk stays on the time axis and no loop appears.
  // CHECK-LABEL:   func.func @matching_stride_stays_on_the_time_axis() attributes {grid = [2]} {
  // CHECK:           %[[C0:.*]] = arith.constant 0 : index
  // CHECK:           dataflow.program_unit iter_arg : %{{.*}} -> (%{{.*}}, %{{.*}}) : {
  // CHECK:             %[[DDR:.*]] = memref.alloc() : memref<2x256x64xf16, "DDR">
  // CHECK:             %[[L1:.*]] = memref.alloc() : memref<2x1x256x64xf16, "L1">
  // CHECK-NOT:         scf.for
  // CHECK:             agen.composite_load_and_store src:%[[DDR]]{{\[}}%[[C0]], 0, 0] dst:%[[L1]]{{\[}}%[[C0]], 0, 0, 0]
  // CHECK:              time_symbols(), load_iv(%{{.*}}:vector<64xf16>)
  // CHECK:              {load_order = #[[$LOAD_ORDER]], load_set = #[[$LOAD_SET]], load_time_addr_map = #[[$WALKED_LOAD_ADDR]], store_order = #[[$STORE_ORDER]], store_set = #[[$STORE_SET]], store_time_addr_map = #[[$WALKED_STORE_ADDR]], time_order = #[[$TIME_ORDER]], time_set = #[[$TIME_256]]}
  // CHECK:           }
  // CHECK:           return
  // CHECK:         }
  func.func @matching_stride_stays_on_the_time_axis() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index

    %ddr_buf = memref.alloc() : memref<2x256x64xf16, "DDR">
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
}
