// RUN: dataflow-scheduler-opt --reduction-dim-chunking %s | FileCheck %s

// Test: threshold path chunks two reduction dims when the outermost one does
// not fit the threshold alone.
//
// Input: tensor<2x4x3136x64xf16>
//   iterator_types = ["reduction", "reduction", "reduction", "parallel"]
//   total_input_bytes = 2 * 4 * 3136 * 64 * 2 = 3,211,264 bytes (~3.06 MiB)
//
// Default threshold = 1 MiB = 1,048,576 bytes.
//
// The reduction dims shrink from the outermost in:
//   d0: no divisor of 2 fits, 1 saves the most → 1,605,632 bytes
//   d1: 2 is the largest divisor of 4 that fits → 802,816 bytes  (stop)
//   d2=3136 (innermost reduction) is never touched
//
// Result: chunk_sizes = [1, 2, 3136], per_dim_num_chunks = [2, 2, 1]
// Two nested scf.for loops: outer bound=2 (d0), inner bound=2 (d1).
// Chunk tensor: tensor<1x2x3136x64xf16>  (401408 elements = 784 KB ≤ 1 MiB ✓)
// is_first = (iv_d0 == 0) AND (iv_d1 == 0)

// CHECK: #[[$MAP0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$MAP1:.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK: #[[$MAP2:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$SET0:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 3135 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$SET1:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 63 >= 0)>
// CHECK-LABEL:   module @local_schedule_0 {
// CHECK-NEXT:     func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK-NEXT:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CADDR:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:       %[[MV0:.*]] = ktdp.construct_memory_view %[[C0]], sizes: [2, 4, 3136, 64], strides: [802816, 200704, 64, 1] {coordinate_set = #[[$SET0]], memory_space = #ktdp.memory_space<global>} : memref<2x4x3136x64xf16>
// CHECK-NEXT:       %[[MV1:.*]] = ktdp.construct_memory_view %[[CADDR]], sizes: [64], strides: [1] {coordinate_set = #[[$SET1]], memory_space = #ktdp.memory_space<global>} : memref<64xf16>
// CHECK-NEXT:       %[[MSC0:.*]] = memref.memory_space_cast %[[MV0]] : memref<2x4x3136x64xf16> to memref<2x4x3136x64xf16, "DDR">
// CHECK-NEXT:       %[[RC0:.*]] = memref.reinterpret_cast %[[MSC0]] to offset: [0], sizes: [2, 4, 3136, 64], strides: [802816, 200704, 64, 1] : memref<2x4x3136x64xf16, "DDR"> to memref<2x4x3136x64xf16, strided<[802816, 200704, 64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST0:.*]] = memref.cast %[[RC0]] : memref<2x4x3136x64xf16, strided<[802816, 200704, 64, 1]>, "DDR"> to memref<2x4x3136x64xf16, strided<[802816, 200704, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[MSC1:.*]] = memref.memory_space_cast %[[MV1]] : memref<64xf16> to memref<64xf16, "DDR">
// CHECK-NEXT:       %[[RC1:.*]] = memref.reinterpret_cast %[[MSC1]] to offset: [0], sizes: [64], strides: [1] : memref<64xf16, "DDR"> to memref<64xf16, strided<[1]>, "DDR">
// CHECK-NEXT:       %[[CAST1:.*]] = memref.cast %[[RC1]] : memref<64xf16, strided<[1]>, "DDR"> to memref<64xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[P:.*]]:4 = ktdf.private -> (memref<1x2x4x3136x64xf16, "L1">, memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[A0:.*]] = memref.alloc() : memref<1x2x4x3136x64xf16, "L1">
// CHECK-NEXT:           %[[A1:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:           %[[T0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[T1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[A0]], %[[A1]], %[[T0]], %[[T1]] : memref<1x2x4x3136x64xf16, "L1">, memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[P]]#2) {
// CHECK-NEXT:           scf.for %[[BIV:.*]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK-NEXT:             ktdf.data_transfer from %[[CAST0]]{{\[}}0, 0, 0, 0] size [2, 4, 3136, 64] to %[[P]]#0{{\[}}%[[BIV]], 0, 0, 0, 0] size [1, 2, 4, 3136, 64] : memref<2x4x3136x64xf16, strided<[802816, 200704, 64, 1], offset: ?>, "DDR">, memref<1x2x4x3136x64xf16, "L1">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNILU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[P]]#2) depends_out(%[[P]]#3) {
// CHECK-NEXT:           scf.for %[[BIV2:.*]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK-NEXT:             %[[CL0:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[CL1B:.*]] = arith.constant 1 : index
// CHECK-NEXT:             %[[CD0:.*]] = arith.constant 2 : index
// CHECK-NEXT:             scf.for %[[IV0:.*]] = %[[CL0]] to %[[CD0]] step %[[CL1B]] {
// CHECK-NEXT:               %[[EQ0:.*]] = arith.cmpi eq, %[[IV0]], %[[CL0]] : index
// CHECK-NEXT:               %[[CD1:.*]] = arith.constant 2 : index
// CHECK-NEXT:               scf.for %[[IV1:.*]] = %[[CL0]] to %[[CD1]] step %[[CL1B]] {
// CHECK-NEXT:                 %[[EQ1:.*]] = arith.cmpi eq, %[[IV1]], %[[CL0]] : index
// CHECK-NEXT:                 %[[ISFIRST:.*]] = arith.andi %[[EQ0]], %[[EQ1]] : i1
// CHECK-NEXT:                 ktdf.pipeline {
// CHECK-NEXT:                   %[[Q:.*]]:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 401408xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                     %[[F0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 401408xf16>
// CHECK-NEXT:                     %[[F1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                     %[[F2:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                     %[[T2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     %[[T3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     ktdf.private_yield %[[F0]], %[[F1]], %[[F2]], %[[T2]], %[[T3]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 401408xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(none) depends_out(%[[Q]]#3) {
// CHECK-NEXT:                     ktdf.data_transfer from %[[P]]#0{{\[}}%[[BIV2]], %[[IV0]], %[[IV1]] * 2, 0, 0] size [1, 1, 2, 3136, 64] to %[[Q]]#0 size [401408] : memref<1x2x4x3136x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 401408xf16>
// CHECK-NEXT:                     scf.if %[[ISFIRST]] {
// CHECK-NEXT:                     } else {
// CHECK-NEXT:                       ktdf.data_transfer from %[[P]]#1{{\[}}%[[BIV2]], 0] size [1, 64] to %[[Q]]#1 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                     }
// CHECK-NEXT:                   } {applicable_units = ["L1LU"]}
// CHECK-NEXT:                   ktdf.stage depends_in(%[[Q]]#3) depends_out(%[[Q]]#4) {
// CHECK-NEXT:                     %[[RFF:.*]] = ktdf.read_from_fifo %[[Q]]#0 : <"L1LU" -> "SFU", 401408xf16> -> tensor<1x2x3136x64xf16>
// CHECK-NEXT:                     %[[EMPTY:.*]] = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:                     %[[GEN0:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "reduction", "reduction", "parallel"]} ins(%[[RFF]] : tensor<1x2x3136x64xf16>) outs(%[[EMPTY]] : tensor<64xf16>) {
// CHECK-NEXT:                     ^bb0(%[[IN:.*]]: f16, %[[OUT:.*]]: f16):
// CHECK-NEXT:                       %[[ADD0:.*]] = arith.addf %[[IN]], %[[OUT]] : f16
// CHECK-NEXT:                       linalg.yield %[[ADD0]] : f16
// CHECK-NEXT:                     } -> tensor<64xf16>
// CHECK-NEXT:                     %[[IF:.*]] = scf.if %[[ISFIRST]] -> (tensor<64xf16>) {
// CHECK-NEXT:                       scf.yield %[[GEN0]] : tensor<64xf16>
// CHECK-NEXT:                     } else {
// CHECK-NEXT:                       %[[RFF2:.*]] = ktdf.read_from_fifo %[[Q]]#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
// CHECK-NEXT:                       %[[GEN1:.*]] = linalg.generic {indexing_maps = [#[[$MAP2]], #[[$MAP2]]], iterator_types = ["parallel"]} ins(%[[RFF2]] : tensor<64xf16>) outs(%[[GEN0]] : tensor<64xf16>) {
// CHECK-NEXT:                       ^bb0(%[[IN2:.*]]: f16, %[[OUT2:.*]]: f16):
// CHECK-NEXT:                         %[[ADD1:.*]] = arith.addf %[[IN2]], %[[OUT2]] : f16
// CHECK-NEXT:                         linalg.yield %[[ADD1]] : f16
// CHECK-NEXT:                       } -> tensor<64xf16>
// CHECK-NEXT:                       scf.yield %[[GEN1]] : tensor<64xf16>
// CHECK-NEXT:                     }
// CHECK-NEXT:                     ktdf.write_to_fifo %[[IF]], %[[Q]]#2 : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                   } {applicable_units = ["SFU"]}
// CHECK-NEXT:                   ktdf.stage depends_in(%[[Q]]#4) depends_out(none) {
// CHECK-NEXT:                     ktdf.data_transfer from %[[Q]]#2 size [64] to %[[P]]#1{{\[}}%[[BIV2]], 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
// CHECK-NEXT:                   } {applicable_units = ["L1SU"]}
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[P]]#3) depends_out(none) {
// CHECK-NEXT:           scf.for %[[BIV3:.*]] = %[[C0]] to %[[C1]] step %[[C1]] {
// CHECK-NEXT:             ktdf.data_transfer from %[[P]]#1{{\[}}%[[BIV3]], 0] size [1, 64] to %[[CAST1]]{{\[}}0] size [64] : memref<1x64xf16, "L1">, memref<64xf16, strided<[1], offset: ?>, "DDR">
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["MNISU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3)>
#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 3 >= 0, d2 >= 0, -d2 + 3135 >= 0, d3 >= 0, -d3 + 63 >= 0)>
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
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 4, 3136, 64], strides: [802816, 200704, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x4x3136x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [64], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x4x3136x64xf16> to memref<2x4x3136x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 4, 3136, 64], strides: [802816, 200704, 64, 1] : memref<2x4x3136x64xf16, "DDR"> to memref<2x4x3136x64xf16, strided<[802816, 200704, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<2x4x3136x64xf16, strided<[802816, 200704, 64, 1]>, "DDR"> to memref<2x4x3136x64xf16, strided<[802816, 200704, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<64xf16> to memref<64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [64], strides: [1] : memref<64xf16, "DDR"> to memref<64xf16, strided<[1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<64xf16, strided<[1]>, "DDR"> to memref<64xf16, strided<[1], offset: ?>, "DDR">
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<1x2x4x3136x64xf16, "L1">, memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<1x2x4x3136x64xf16, "L1">
          %alloc_3 = memref.alloc() : memref<1x64xf16, "L1">
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<1x2x4x3136x64xf16, "L1">, memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c1 step %c1 {
            ktdf.data_transfer from %cast[%c0, %c0, %c0, %c0] size [2, 4, 3136, 64] to %2#0[%arg0, 0, 0, 0, 0] size [1, 2, 4, 3136, 64] : memref<2x4x3136x64xf16, strided<[802816, 200704, 64, 1], offset: ?>, "DDR">, memref<1x2x4x3136x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c1 step %c1 {
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 1605632xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 1605632xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 1605632xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                ktdf.data_transfer from %2#0[%arg0, 0, 0, 0, 0] size [1, 2, 4, 3136, 64] to %3#0 size [1605632] : memref<1x2x4x3136x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 1605632xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 1605632xf16> -> tensor<2x4x3136x64xf16>
                %5 = tensor.empty() : tensor<64xf16>
                %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "reduction", "reduction", "parallel"]} ins(%4 : tensor<2x4x3136x64xf16>) outs(%5 : tensor<64xf16>) {
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
