// RUN: dataflow-scheduler-opt --reduction-loop-exposure %s | FileCheck %s

// A SIMD lane need not be the last input axis alone.  For f16 the default
// compute unit has 64 lanes, and the trailing [32, 2] axes together hold one
// SIMD lane.  The input is what split-reduction-inner-outer-dim produces:
// Generic 1 reduces d0 across SIMD lanes, and Generic 2 reduces the SIMD lane
// itself.  Only Generic 1's d0 becomes an scf.for, and each iteration loads a
// [1, 8, 32, 2] slice.  Generic 2's reductions over both SIMD-lane axes stay
// inside it; a loop over its 32-wide axis would leave two elements per load.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @simd_lane_spans_two_dims(
// CHECK-SAME:      %[[ARG0:.*]]: memref<4x8x32x2xf16, "L1">,
// CHECK-SAME:      %[[ARG1:.*]]: memref<8xf16, "L1">) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 3 : index
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[PRIVATE_0:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 512xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 8xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:         %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 512xf16>
// CHECK-NEXT:         %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 8xf16>
// CHECK-NEXT:         %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 512xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 8xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-NEXT:         %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK-NEXT:         scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:           ktdf.data_transfer from %[[ARG0]]{{\[}}%[[VAL_1]], 0, 0, 0] size [1, 8, 32, 2] to %[[VAL_0]]#0 size [512] : memref<4x8x32x2xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 512xf16>
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:       } {applicable_units = ["L1LU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK-NEXT:         %[[EMPTY_0:.*]] = tensor.empty() : tensor<8x32x2xf16>
// CHECK-NEXT:         %[[CONSTANT_4:.*]] = arith.constant 4 : index
// CHECK-NEXT:         %[[FOR_0:.*]] = scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_4]] step %[[CONSTANT_1]] iter_args(%[[VAL_4:.*]] = %[[EMPTY_0]]) -> (tensor<8x32x2xf16>) {
// CHECK-NEXT:           %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_2]]#0 : <"L1LU" -> "SFU", 512xf16> -> tensor<1x8x32x2xf16>
// CHECK-NEXT:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<1x8x32x2xf16>) outs(%[[VAL_4]] : tensor<8x32x2xf16>) {
// CHECK-NEXT:           ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK-NEXT:             %[[ADDF_0:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f16
// CHECK-NEXT:             linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:           } -> tensor<8x32x2xf16>
// CHECK-NEXT:           scf.yield %[[GENERIC_0]] : tensor<8x32x2xf16>
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:         %[[EMPTY_1:.*]] = tensor.empty() : tensor<8xf16>
// CHECK-NEXT:         %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_3]]], iterator_types = ["parallel", "reduction", "reduction"]} ins(%[[FOR_0]] : tensor<8x32x2xf16>) outs(%[[EMPTY_1]] : tensor<8xf16>) {
// CHECK-NEXT:         ^bb0(%[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16):
// CHECK-NEXT:           %[[ADDF_1:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f16
// CHECK-NEXT:           linalg.yield %[[ADDF_1]] : f16
// CHECK-NEXT:         } -> tensor<8xf16>
// CHECK-NEXT:         ktdf.write_to_fifo %[[GENERIC_1]], %[[VAL_2]]#1 : tensor<8xf16>, <"SFU" -> "L1SU", 8xf16>
// CHECK-NEXT:       } {applicable_units = ["SFU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_9:.*]]#3) depends_out(none) {
// CHECK-NEXT:         %[[CONSTANT_5:.*]] = arith.constant 4 : index
// CHECK-NEXT:         scf.for %[[VAL_10:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_5]] step %[[CONSTANT_1]] {
// CHECK-NEXT:           %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_10]], %[[CONSTANT_2]] : index
// CHECK-NEXT:           scf.if %[[CMPI_0]] {
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_9]]#1 size [8] to %[[ARG1]][0] size [8] : !ktdf.fifo.slot<"SFU" -> "L1SU", 8xf16>, memref<8xf16, "L1">
// CHECK-NEXT:           }
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:       } {applicable_units = ["L1SU"]}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @simd_lane_spans_two_dims(%src: memref<4x8x32x2xf16, "L1">, %dst: memref<8xf16, "L1">) {
    ktdf.pipeline {
      %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 2048xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 8xf16>, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 2048xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 8xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 2048xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 8xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#2) {
        ktdf.data_transfer from %src[0, 0, 0, 0] size [4, 8, 32, 2] to %p#0 size [2048] : memref<4x8x32x2xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 2048xf16>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%p#2) depends_out(%p#3) {
        %in = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 2048xf16> -> tensor<4x8x32x2xf16>
        %init0 = tensor.empty() : tensor<8x32x2xf16>
        %partial = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel", "parallel", "parallel"]} ins(%in : tensor<4x8x32x2xf16>) outs(%init0 : tensor<8x32x2xf16>) {
        ^bb0(%a: f16, %b: f16):
          %s = arith.addf %a, %b : f16
          linalg.yield %s : f16
        } -> tensor<8x32x2xf16>
        %init1 = tensor.empty() : tensor<8xf16>
        %r = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction", "reduction"]} ins(%partial : tensor<8x32x2xf16>) outs(%init1 : tensor<8xf16>) {
        ^bb0(%a: f16, %b: f16):
          %s = arith.addf %a, %b : f16
          linalg.yield %s : f16
        } -> tensor<8xf16>
        ktdf.write_to_fifo %r, %p#1 : tensor<8xf16>, <"SFU" -> "L1SU", 8xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#3) depends_out(none) {
        ktdf.data_transfer from %p#1 size [8] to %dst[0] size [8] : !ktdf.fifo.slot<"SFU" -> "L1SU", 8xf16>, memref<8xf16, "L1">
      } {applicable_units = ["L1SU"]}
    }
    return
  }
}
