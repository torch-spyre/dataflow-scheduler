// RUN: dataflow-scheduler-opt --reduction-dim-chunking %s | FileCheck %s
// RUN: dataflow-scheduler-opt --reduction-dim-chunking="num-chunks=2" %s \
// RUN:   -verify-diagnostics

// A reduction on a SINGLE-LEVEL pipeline: one ktdf.pipeline holding the three
// stages, with no enclosing scf.for.  That is what a reduction whose outermost
// parallel dim has extent 1 lowers to -- ConstructThreeStagePipeline emits a
// loop for that dim, the loop is single-iteration, and the canonicalizer
// deletes it, so StageCoarsening has no nest to invert and builds no nested
// pipeline.
//
// The reduction extent here (256 f16) is far below chunk-size-threshold, so the
// analysis infers one chunk and the pass must leave the IR alone.  It used to
// enter the rewrite anyway and abort on cast<scf::ForOp>, reading the batch loop
// that a single-level pipeline does not have.
//
// The second RUN forces num-chunks=2, which is the only way to reach the
// rewrite here.  That must report the missing batch loop rather than assert.

// The generic and its stage survive untouched, still directly under the one
// pipeline: no chunk loop, no partial FIFO slot, no scf.if.
// CHECK-LABEL: func.func @local_schedule_0
// CHECK:         ktdf.pipeline {
// CHECK-NOT:       scf.for
// CHECK:           linalg.generic
// CHECK-SAME:        iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-NOT:       scf.for

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  module {
    func.func @sum_2core() attributes {grid = [2]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c16384 = arith.constant 16384 : index
      %c64 = arith.constant 64 : index
      %c8589934592 = arith.constant 8589934592 : index
      %0 = ktdp.get_compute_tile_id : index
      %1 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %2 = ktdp.construct_memory_view %c8589934592, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
      %3 = arith.muli %0, %c16384 : index
      %memspacecast = memref.memory_space_cast %1 : memref<2x256x64xf16> to memref<2x256x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [%3], sizes: [1, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, #ktdp.memory_space<global>> to memref<1x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>
      %4 = arith.muli %0, %c64 : index
      %memspacecast_0 = memref.memory_space_cast %2 : memref<2x64xf16> to memref<2x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [%4], sizes: [1, 64], strides: [64, 1] : memref<2x64xf16, #ktdp.memory_space<global>> to memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
      // expected-error @below {{reduction-dim-chunking: reduction needs chunking, but its pipeline is not nested in a batch scf.for}}
      ktdf.pipeline {
        %5:8 = ktdf.private -> (memref<1x256x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<1x256x64xf16, #ktdp.memory_space<ct_local>>
          %6 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
          %7 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
          %alloc_2 = memref.alloc() : memref<1x64xf16, #ktdp.memory_space<ct_local>>
          %8 = ktdf.create_token : !ktdf.token
          %9 = ktdf.create_token : !ktdf.token
          %10 = ktdf.create_token : !ktdf.token
          %11 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %6, %7, %alloc_2, %8, %9, %10, %11 : memref<1x256x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%5#4) {
          ktdf.data_transfer from %reinterpret_cast[%c0, 0, %c0 * 64] size [1, 256, 64] to %5#0[0, 0, 0] size [1, 256, 64] : memref<1x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<1x256x64xf16, #ktdp.memory_space<ct_local>>
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%5#4) depends_out(%5#5) {
          ktdf.data_transfer from %5#0[0, 0, 0] size [1, 256, 64] to %5#1 size [16384] : memref<1x256x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%5#5) depends_out(%5#6) {
          %6 = ktdf.read_from_fifo %5#1 : <"L1LU" -> "SFU", 16384xf16> -> tensor<1x256x64xf16>
          %7 = tensor.empty() : tensor<1x64xf16>
          %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%6 : tensor<1x256x64xf16>) outs(%7 : tensor<1x64xf16>) {
          ^bb0(%in: f16, %out: f16):
            %9 = arith.addf %in, %out : f16
            linalg.yield %9 : f16
          } -> tensor<1x64xf16>
          ktdf.write_to_fifo %8, %5#2 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%5#6) depends_out(%5#7) {
          ktdf.data_transfer from %5#2 size [64] to %5#3[0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, #ktdp.memory_space<ct_local>>
        } {applicable_units = ["L1SU"]}
        ktdf.stage depends_in(%5#7) depends_out(none) {
          ktdf.data_transfer from %5#3[0, 0] size [1, 64] to %reinterpret_cast_1[%c0, %c0 * 64] size [1, 64] : memref<1x64xf16, #ktdp.memory_space<ct_local>>, memref<1x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
