// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// Two stores into one memory view, inside a parallel construct distributed over two
// corelets.
//
// The offset that sends the second corelet to its half of the iteration space belongs
// to the view, not to the operation, so a view is moved once however many operations
// read or write it. Moving it per operation put the second corelet's stores a whole
// half beyond where the transfer out of that buffer reads, and nothing wrote the half
// it does read -- the first corelet's results came out right and the second's came out
// as whatever the buffer held.
//
// So the start address takes exactly one addi over the query_map, and both stores go
// to the same view.

// CHECK-LABEL:   func.func @two_stores_one_view()
// CHECK:           dataflow.program_unit
// CHECK:             %[[MAP:.*]] = uniform.def_immutable_mapping({{.*}}%c128{{.*}})
// CHECK-NEXT:        %[[OFFSET:.*]] = uniform.query_map(map:%[[MAP]]
// The view follows the one addi directly: a second would sit here.
// CHECK-NEXT:        %[[START:.*]] = arith.addi %[[OFFSET]], %{{.*}} : index
// CHECK-NEXT:        %[[VIEW:.*]] = dataflow.get_logical_memory_view %{{.*}}, %[[START]]
// CHECK:             agen.vector_store %{{.*}}, %[[VIEW]]
// CHECK-NEXT:        agen.vector_store %{{.*}}, %[[VIEW]]

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @two_stores_one_view() attributes {grid = [2]} {
    %0 = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-SFU-CL0", type = "SFU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-SFU-CL0", type = "SFU"} : index
    %2 = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-SFU-CL1", type = "SFU"} : index
    %3 = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-SFU-CL1", type = "SFU"} : index
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
    %c4 = arith.constant 4 : index
    %base = arith.constant 1024 : index
    ktdf.parallel (%arg0, %arg1) = (%c0_2) to (%c4) step (%c1_3) distribute(num_instances = 2) {
      ktdf_lowering.execute_on %6, %8 {
        %view = dataflow.get_logical_memory_view %6, %base {layout_map = affine_map<(d0, d1) -> (d0 * 64 + d1)>} : index, index, memref<4x64xf16>
        %v0 = vectorchain.constant_bitstream {value = [0x3c00]} : vector<1xf16>
        %v1 = vectorchain.shuffle input(%v0) {indices = [0 : i32], repetition = 64 : i32} : vector<1xf16>, vector<64xf16>
        agen.vector_store %v1, %view[%arg0, %c0_2] {store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<4x64xf16>, vector<64xf16>
        agen.vector_store %v1, %view[%arg0, %c0_2] {store_order = affine_map<(d0, d1) -> (d0, d1)>, store_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>} : memref<4x64xf16>, vector<64xf16>
      }
      ktdf.parallel_yield
    }
    return
  }
}
