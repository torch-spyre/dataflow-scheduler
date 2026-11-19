// RUN: dataflow-scheduler-opt --reduction-dim-chunking="chunk-size-threshold=32768" %s | FileCheck %s

// Test: auto-infer num_chunks from chunk-size-threshold, N=1 no-op case.
//
// Input: tensor<1x256x64xf16>  (reduction dim 1, size 256)
//   total_input_bytes = 1 * 256 * 64 * 2 = 32768
//
//   threshold=32768 → smallest N with 32768/N ≤ 32768 is N=1 → nothing to do
//
// N=1 means every per-dim chunk count is 1, so the all-ones guard fires and
// the pass leaves the IR completely unchanged.

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$SET0:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$SET1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   module @local_schedule_0 {
// CHECK-NEXT:     func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK-NEXT:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CADDR:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[MV0:.*]] = ktdp.construct_memory_view %[[C0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$SET0]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
// CHECK-NEXT:       %[[MV1:.*]] = ktdp.construct_memory_view %[[CADDR]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$SET1]], memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
// CHECK-NEXT:       %[[MSC0:.*]] = memref.memory_space_cast %[[MV0]] : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
// CHECK-NEXT:       %[[RC0:.*]] = memref.reinterpret_cast %[[MSC0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST0:.*]] = memref.cast %[[RC0]] : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[MSC1:.*]] = memref.memory_space_cast %[[MV1]] : memref<2x64xf16> to memref<2x64xf16, "DDR">
// CHECK-NEXT:       %[[RC1:.*]] = memref.reinterpret_cast %[[MSC1]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "DDR"> to memref<2x64xf16, strided<[64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST1:.*]] = memref.cast %[[RC1]] : memref<2x64xf16, strided<[64, 1]>, "DDR"> to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[P:.*]]:4 = ktdf.private -> (memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[A0:.*]] = memref.alloc() : memref<2x1x256x64xf16, "L1">
// CHECK-NEXT:           %[[A1:.*]] = memref.alloc() : memref<2x1x64xf16, "L1">
// CHECK-NEXT:           %[[T0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[T1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[A0]], %[[A1]], %[[T0]], %[[T1]] : memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[P]]#2) {
// CHECK-NEXT:           scf.for %[[IV0:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:             %[[S0:.*]] = arith.subi %[[IV0]], %[[C0]] : index
// CHECK-NEXT:             %[[D0:.*]] = arith.divsi %[[S0]], %[[C1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[CAST0]]{{\[}}%[[IV0]], 0, %[[C0]] * 64] size [1, 256, 64] to %[[P]]#0{{\[}}%[[D0]], 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x1x256x64xf16, "L1">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNILU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[P]]#2) depends_out(%[[P]]#3) {
// CHECK-NEXT:           scf.for %[[IV1:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:             ktdf.pipeline {
// CHECK-NEXT:               %[[Q:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                 %[[F0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
// CHECK-NEXT:                 %[[F1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                 %[[T2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 %[[T3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                 ktdf.private_yield %[[F0]], %[[F1]], %[[T2]], %[[T3]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:               }
// CHECK-NEXT:               ktdf.stage depends_in(none) depends_out(%[[Q]]#2) {
// CHECK-NEXT:                 %[[S1:.*]] = arith.subi %[[IV1]], %[[C0]] : index
// CHECK-NEXT:                 %[[D1:.*]] = arith.divsi %[[S1]], %[[C1]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[P]]#0{{\[}}%[[D1]], 0, 0, 0] size [1, 1, 256, 64] to %[[Q]]#0 size [16384] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
// CHECK-NEXT:               } {applicable_units = ["L1LU"]}
// CHECK-NEXT:               ktdf.stage depends_in(%[[Q]]#2) depends_out(%[[Q]]#3) {
// CHECK-NEXT:                 %[[RFF:.*]] = ktdf.read_from_fifo %[[Q]]#0 : <"L1LU" -> "SFU", 16384xf16> -> tensor<1x256x64xf16>
// CHECK-NEXT:                 %[[EMPTY:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:                 %[[GEN:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[RFF]] : tensor<1x256x64xf16>) outs(%[[EMPTY]] : tensor<1x64xf16>) {
// CHECK-NEXT:                 ^bb0(%[[IN:.*]]: f16, %[[OUT:.*]]: f16):
// CHECK-NEXT:                   %[[ADD:.*]] = arith.addf %[[IN]], %[[OUT]] : f16
// CHECK-NEXT:                   linalg.yield %[[ADD]] : f16
// CHECK-NEXT:                 } -> tensor<1x64xf16>
// CHECK-NEXT:                 ktdf.write_to_fifo %[[GEN]], %[[Q]]#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:               } {applicable_units = ["SFU"]}
// CHECK-NEXT:               ktdf.stage depends_in(%[[Q]]#3) depends_out(none) {
// CHECK-NEXT:                 %[[S2:.*]] = arith.subi %[[IV1]], %[[C0]] : index
// CHECK-NEXT:                 %[[D2:.*]] = arith.divsi %[[S2]], %[[C1]] : index
// CHECK-NEXT:                 ktdf.data_transfer from %[[Q]]#1 size [64] to %[[P]]#1{{\[}}%[[D2]], 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
// CHECK-NEXT:               } {applicable_units = ["L1SU"]}
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[P]]#3) depends_out(none) {
// CHECK-NEXT:           scf.for %[[IV2:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT:             %[[S3:.*]] = arith.subi %[[IV2]], %[[C0]] : index
// CHECK-NEXT:             %[[D3:.*]] = arith.divsi %[[S3]], %[[C1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[P]]#1{{\[}}%[[D3]], 0, 0] size [1, 1, 64] to %[[CAST1]]{{\[}}%[[IV2]], %[[C0]] * 64] size [1, 64] : memref<2x1x64xf16, "L1">, memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNISU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  module {
    func.func @sum_1core() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %c2 = arith.constant 2 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<2x64xf16> to memref<2x64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, "DDR"> to memref<2x64xf16, strided<[64, 1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<2x64xf16, strided<[64, 1]>, "DDR"> to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<2x1x256x64xf16, "L1">
          %alloc_3 = memref.alloc() : memref<2x1x64xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<2x1x256x64xf16, "L1">, memref<2x1x64xf16, "L1">, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %cast[%arg0, 0, %c0 * 64] size [1, 256, 64] to %2#0[%4, 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, memref<2x1x256x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                ktdf.data_transfer from %2#0[%5, 0, 0, 0] size [1, 1, 256, 64] to %3#0 size [16384] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 16384xf16> -> tensor<1x256x64xf16>
                %5 = tensor.empty() : tensor<1x64xf16>
                %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "parallel"]} ins(%4 : tensor<1x256x64xf16>) outs(%5 : tensor<1x64xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %7 = arith.addf %in, %out : f16
                  linalg.yield %7 : f16
                } -> tensor<1x64xf16>
                ktdf.write_to_fifo %6, %3#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                ktdf.data_transfer from %3#1 size [64] to %2#1[%5, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %2#1[%4, 0, 0] size [1, 1, 64] to %cast_2[%arg0, %c0 * 64] size [1, 64] : memref<2x1x64xf16, "L1">, memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
