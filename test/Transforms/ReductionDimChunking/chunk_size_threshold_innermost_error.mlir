// RUN: dataflow-scheduler-opt --reduction-dim-chunking="chunk-size-threshold=100" %s -verify-diagnostics

// Test: threshold path emits an error when no chunking fits the threshold.
//
// Input linalg.generic over tensor<4x3x64xf16>:
//   iterator_types = ["reduction", "reduction", "parallel"]
//   total_input_bytes = 4 * 3 * 64 * 2 = 1536 bytes
//
// threshold=100 bytes.  The innermost overall loop dim is index 2 (parallel),
// so both reduction dims are chunkable: d0=4 (index 0) and d1=3 (index 1).
// Chunking both down to 1 still leaves one row of the innermost dim,
// 64 * 2 = 128 bytes, which exceeds the threshold, and the pass signals
// failure.

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 2 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
module {
  module {
    func.func @sum_1core() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @spyre_single_corelet import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [4, 3, 64], strides: [192, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<4x3x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<4x3x64xf16> to memref<4x3x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [4, 3, 64], strides: [192, 64, 1] : memref<4x3x64xf16, "DDR"> to memref<4x3x64xf16, strided<[192, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<4x3x64xf16, strided<[192, 64, 1]>, "DDR"> to memref<4x3x64xf16, strided<[192, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<64xf16> to memref<64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [64], strides: [1] : memref<64xf16, "DDR"> to memref<64xf16, strided<[1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<64xf16, strided<[1]>, "DDR"> to memref<64xf16, strided<[1], offset: ?>, "DDR">
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<1x4x3x64xf16, "L1">, memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<1x4x3x64xf16, "L1">
          %alloc_3 = memref.alloc() : memref<1x64xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<1x4x3x64xf16, "L1">, memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c1 step %c1 {
            ktdf.data_transfer from %cast[%c0, %c0, %c0] size [4, 3, 64] to %2#0[%arg0, 0, 0, 0] size [1, 4, 3, 64] : memref<4x3x64xf16, strided<[192, 64, 1], offset: ?>, "DDR">, memref<1x4x3x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c1 step %c1 {
            // expected-error @below {{reduction-dim-chunking: no chunking of the reduction fits the threshold of 100 bytes}}
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 768xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 768xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 768xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                ktdf.data_transfer from %2#0[%arg0, 0, 0, 0] size [1, 4, 3, 64] to %3#0 size [768] : memref<1x4x3x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 768xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 768xf16> -> tensor<4x3x64xf16>
                %5 = tensor.empty() : tensor<64xf16>
                %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction", "parallel"]} ins(%4 : tensor<4x3x64xf16>) outs(%5 : tensor<64xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %7 = arith.addf %in, %out : f16
                  linalg.yield %7 : f16
                } -> tensor<64xf16>
                ktdf.write_to_fifo %6, %3#1 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                ktdf.data_transfer from %3#1 size [64] to %2#1[%arg0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c1 step %c1 {
            ktdf.data_transfer from %2#1[%arg0, 0] size [1, 64] to %cast_2[%c0] size [64] : memref<1x64xf16, "L1">, memref<64xf16, strided<[1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
