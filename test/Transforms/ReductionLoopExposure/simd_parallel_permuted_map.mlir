// RUN: dataflow-scheduler-opt --reduction-loop-exposure %s | FileCheck %s

// The SIMD dims come from the input's trailing axes, not from the loop order.
// The loops are d0 = 8, d1 = 32, d2 = 2 and d3 = 64, with d3 the reduction,
// but the input is tensor<8x64x32x2xf16> indexed by (d0, d3, d1, d2).  On the
// sample device a SIMD vector is 64 f16 elements, so it is the 32x2 of the two
// innermost input axes, held by the parallel d1 and d2.  The SIMD dims are not
// reductions, so findReductionSimdDims returns no dims and d3 becomes an
// scf.for over input axis 1.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @simd_parallel_permuted_map(
// CHECK-SAME:      %[[ARG0:.*]]: memref<8x64x32x2xf16, "L1">,
// CHECK-SAME:      %[[ARG1:.*]]: memref<8x32x2xf16, "L1">) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 63 : index
// CHECK-NEXT:     ktdf.pipeline {
// CHECK-NEXT:       %[[PRIVATE_0:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 512xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 512xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:         %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 512xf16>
// CHECK-NEXT:         %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 512xf16>
// CHECK-NEXT:         %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:         ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 512xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 512xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:       }
// CHECK-NEXT:       ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-NEXT:         %[[CONSTANT_3:.*]] = arith.constant 64 : index
// CHECK-NEXT:         scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:           ktdf.data_transfer from %[[ARG0]][0, %[[VAL_1]], 0, 0] size [8, 1, 32, 2] to %[[VAL_0]]#0 size [512] : memref<8x64x32x2xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 512xf16>
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:       } {applicable_units = ["L1LU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK-NEXT:         %[[EMPTY_0:.*]] = tensor.empty() : tensor<8x32x2xf16>
// CHECK-NEXT:         %[[CONSTANT_4:.*]] = arith.constant 64 : index
// CHECK-NEXT:         %[[FOR_0:.*]] = scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_4]] step %[[CONSTANT_1]] iter_args(%[[VAL_4:.*]] = %[[EMPTY_0]]) -> (tensor<8x32x2xf16>) {
// CHECK-NEXT:           %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_2]]#0 : <"L1LU" -> "SFU", 512xf16> -> tensor<8x1x32x2xf16>
// CHECK-NEXT:           %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[READ_FROM_FIFO_0]] : tensor<8x1x32x2xf16>) outs(%[[VAL_4]] : tensor<8x32x2xf16>) {
// CHECK-NEXT:           ^bb0(%[[VAL_5:.*]]: f16, %[[VAL_6:.*]]: f16):
// CHECK-NEXT:             %[[ADDF_0:.*]] = arith.addf %[[VAL_5]], %[[VAL_6]] : f16
// CHECK-NEXT:             linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:           } -> tensor<8x32x2xf16>
// CHECK-NEXT:           %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_3]], %[[CONSTANT_2]] : index
// CHECK-NEXT:           scf.if %[[CMPI_0]] {
// CHECK-NEXT:             ktdf.write_to_fifo %[[GENERIC_0]], %[[VAL_2]]#1 : tensor<8x32x2xf16>, <"SFU" -> "L1SU", 512xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:           scf.yield %[[GENERIC_0]] : tensor<8x32x2xf16>
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:       } {applicable_units = ["SFU"]}
// CHECK-NEXT:       ktdf.stage depends_in(%[[VAL_7:.*]]#3) depends_out(none) {
// CHECK-NEXT:         %[[CONSTANT_5:.*]] = arith.constant 64 : index
// CHECK-NEXT:         scf.for %[[VAL_8:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_5]] step %[[CONSTANT_1]] {
// CHECK-NEXT:           %[[CMPI_1:.*]] = arith.cmpi eq, %[[VAL_8]], %[[CONSTANT_2]] : index
// CHECK-NEXT:           scf.if %[[CMPI_1]] {
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_7]]#1 size [512] to %[[ARG1]][0, 0, 0] size [8, 32, 2] : !ktdf.fifo.slot<"SFU" -> "L1SU", 512xf16>, memref<8x32x2xf16, "L1">
// CHECK-NEXT:           }
// CHECK-NEXT:         } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:       } {applicable_units = ["L1SU"]}
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#in  = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#out = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @simd_parallel_permuted_map(%src: memref<8x64x32x2xf16, "L1">, %dst: memref<8x32x2xf16, "L1">) {
    ktdf.pipeline {
      %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 512xf16>, !ktdf.token, !ktdf.token) {
        %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>
        %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 512xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %f0, %f1, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 512xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%p#2) {
        ktdf.data_transfer from %src[0, 0, 0, 0] size [8, 64, 32, 2] to %p#0 size [32768] : memref<8x64x32x2xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%p#2) depends_out(%p#3) {
        %in = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 32768xf16> -> tensor<8x64x32x2xf16>
        %init = tensor.empty() : tensor<8x32x2xf16>
        %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%in : tensor<8x64x32x2xf16>) outs(%init : tensor<8x32x2xf16>) {
        ^bb0(%a: f16, %acc: f16):
          %s = arith.addf %a, %acc : f16
          linalg.yield %s : f16
        } -> tensor<8x32x2xf16>
        ktdf.write_to_fifo %r, %p#1 : tensor<8x32x2xf16>, <"SFU" -> "L1SU", 512xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%p#3) depends_out(none) {
        ktdf.data_transfer from %p#1 size [512] to %dst[0, 0, 0] size [8, 32, 2] : !ktdf.fifo.slot<"SFU" -> "L1SU", 512xf16>, memref<8x32x2xf16, "L1">
      } {applicable_units = ["L1SU"]}
    }
    return
  }
}
