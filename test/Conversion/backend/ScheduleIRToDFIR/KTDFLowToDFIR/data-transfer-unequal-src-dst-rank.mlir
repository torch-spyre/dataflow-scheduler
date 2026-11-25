// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Regression test: kLoadAndStore data_transfer where source and destination
// memrefs have different ranks (7 vs 5). Previously lowerAsLoadAndStore used
// a single num_dims derived from the source rank for both load_order and
// store_order, causing an assertion in CompositeLoadAndStoreOp::build because
// store_set.getNumDims() != store_order.getNumDims().

// Affine maps/sets are printed as top-level aliases; verify the aliases carry
// the right dimensionality, then verify the op references them correctly.

// load_order and load_time_addr_map must use the source rank (7).
// CHECK-DAG: #[[$LOAD_ORDER:.+]] = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
// CHECK-DAG: #[[$LOAD_TIME_ADDR:.+]] = affine_map<(d0) -> (0, 0, 0, 0, 0, 0, 0)>
// load_set: 7-dim set where the last dim ranges [0,63] and first 6 are ==0.
// CHECK-DAG: #[[$LOAD_SET:.+]] = affine_set<(d0, d1, d2, d3, d4, d5, d6) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0, d4 == 0, d5 == 0

// store_order and store_time_addr_map must use the destination rank (5).
// CHECK-DAG: #[[$STORE_ORDER:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-DAG: #[[$STORE_TIME_ADDR:.+]] = affine_map<(d0) -> (0, 0, 0, 0, 0)>
// store_set: 5-dim set where the last dim ranges [0,63] and first 4 are ==0.
// CHECK-DAG: #[[$STORE_SET:.+]] = affine_set<(d0, d1, d2, d3, d4) : (d0 == 0, d1 == 0, d2 == 0, d3 == 0

// The emitted composite_load_and_store op must reference the right aliases.
// Attributes are printed on the line following the op signature.
// CHECK: agen.composite_load_and_store
// CHECK-NEXT: time_symbols
// CHECK-NEXT: {load_order = #[[$LOAD_ORDER]], load_set = #[[$LOAD_SET]], load_time_addr_map = #[[$LOAD_TIME_ADDR]], store_order = #[[$STORE_ORDER]], store_set = #[[$STORE_SET]], store_time_addr_map = #[[$STORE_TIME_ADDR]]

#set = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @test(%arg0: index, %arg1: index) attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c113216 = arith.constant 113216 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNISU", type = "MNISU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNISU", type = "MNISU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    // Source: rank-7 L1 buffer (stage-coarsened, two dynamic leading dims).
    %l1_buf = memref.alloc(%arg0, %arg1) : memref<?x?x1x1x1x1x64xf16, "L1">
    // Destination: rank-5 DDR buffer.
    %ddr_view = ktdp.construct_memory_view %c113216, sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {
      coordinate_set = #set,
      memory_space = #ktdp.memory_space<global>
    } : memref<12x1x1x64x64xf16>
    %memspacecast = memref.memory_space_cast %ddr_view : memref<12x1x1x64x64xf16> to memref<12x1x1x64x64xf16, "DDR">
    %reinterpret = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] : memref<12x1x1x64x64xf16, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR">
    %cast = memref.cast %reinterpret : memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
    scf.for %i = %c0 to %arg0 step %c1 {
      scf.for %j = %c0 to %arg1 step %c1 {
        ktdf_lowering.execute_on %unit {
          ktdf_lowering.execute_on %unit {
            ktdf.data_transfer from %l1_buf[%i, %j, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64]
                               to %cast[%i, 0, 0, %j, 0] size [1, 1, 1, 1, 64]
              {dataflow_scheduler.access_granularity = 64 : i64}
              : memref<?x?x1x1x1x1x64xf16, "L1">,
                memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
          }
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
