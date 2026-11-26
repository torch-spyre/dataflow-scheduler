// RUN: dataflow-scheduler-opt --reduction-dim-chunking="chunk-size-threshold=128" %s | FileCheck %s

// The SIMD dims can be more than one input dim. On the sample device a SIMD vector is
// 64 f16 elements, so for tensor<1x4x32x2xf16> it is the 32x2 of the two
// innermost dims, and both of their reduction loop dims (d2, d3) stay out of
// chunking, not only the innermost one.
//
// total_input_bytes = 4 * 32 * 2 * 2 = 512; threshold=128 -> N=4, which only
// the 4-wide d1 is left to take: 4 chunks of one SIMD vector each.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3) -> (d0)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @two_dim_simd(
// CHECK-SAME:      %[[ARG0:.*]]: memref<2x1x4x32x2xf16, "L1">,
// CHECK-SAME:      %[[ARG1:.*]]: memref<2x1xf16, "L1">) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_1]] {
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_3]] step %[[CONSTANT_5]] {
// CHECK-NEXT:         %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_1]], %[[CONSTANT_4]] : index
// CHECK-NEXT:         ktdf.pipeline modulo(size : 1) {
// CHECK-NEXT:           %[[PRIVATE_0:.*]]:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 1xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:             %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:             %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 1xf16>
// CHECK-NEXT:             %[[FIFO_2:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>
// CHECK-NEXT:             %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[FIFO_2]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 1xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[VAL_2:.*]]#3) {
// CHECK-NEXT:             %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[CONSTANT_7:.*]] = arith.constant 1 : index
// CHECK-NEXT:             %[[SUBI_0:.*]] = arith.subi %[[VAL_0]], %[[CONSTANT_6]] : index
// CHECK-NEXT:             %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_7]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[ARG0]]{{\[}}%[[DIVSI_0]], %[[CONSTANT_6]], %[[VAL_1]], %[[CONSTANT_6]], %[[CONSTANT_6]]] size [1, 1, 1, 32, 2] to %[[VAL_2]]#0 size [64] : memref<2x1x4x32x2xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:             scf.if %[[CMPI_0]] {
// CHECK-NEXT:             } else {
// CHECK-NEXT:               ktdf.data_transfer from %[[ARG1]]{{\[}}%[[DIVSI_0]], %[[CONSTANT_6]]] size [1, 1] to %[[VAL_2]]#1 size [1] : memref<2x1xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 1xf16>
// CHECK-NEXT:             }
// CHECK-NEXT:           } {applicable_units = ["L1LU"]}
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_3:.*]]#3) depends_out(%[[VAL_3]]#4) {
// CHECK-NEXT:             %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_3]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x32x2xf16>
// CHECK-NEXT:             %[[EMPTY_0:.*]] = tensor.empty() : tensor<1xf16>
// CHECK-NEXT:             %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "reduction", "reduction", "reduction"]} ins(%[[READ_FROM_FIFO_0]] : tensor<1x1x32x2xf16>) outs(%[[EMPTY_0]] : tensor<1xf16>) {
// CHECK-NEXT:             ^bb0(%[[VAL_4:.*]]: f16, %[[VAL_5:.*]]: f16):
// CHECK-NEXT:               %[[ADDF_0:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f16
// CHECK-NEXT:               linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:             } -> tensor<1xf16>
// CHECK-NEXT:             %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (tensor<1xf16>) {
// CHECK-NEXT:               scf.yield %[[GENERIC_0]] : tensor<1xf16>
// CHECK-NEXT:             } else {
// CHECK-NEXT:               %[[READ_FROM_FIFO_1:.*]] = ktdf.read_from_fifo %[[VAL_3]]#1 : <"L1LU" -> "SFU", 1xf16> -> tensor<1xf16>
// CHECK-NEXT:               %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel"]} ins(%[[READ_FROM_FIFO_1]] : tensor<1xf16>) outs(%[[GENERIC_0]] : tensor<1xf16>) {
// CHECK-NEXT:               ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK-NEXT:                 %[[ADDF_1:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f16
// CHECK-NEXT:                 linalg.yield %[[ADDF_1]] : f16
// CHECK-NEXT:               } -> tensor<1xf16>
// CHECK-NEXT:               scf.yield %[[GENERIC_1]] : tensor<1xf16>
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf.write_to_fifo %[[IF_0]], %[[VAL_3]]#2 : tensor<1xf16>, <"SFU" -> "L1SU", 1xf16>
// CHECK-NEXT:           } {applicable_units = ["SFU"]}
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_8:.*]]#4) depends_out(none) {
// CHECK-NEXT:             %[[CONSTANT_8:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[CONSTANT_9:.*]] = arith.constant 1 : index
// CHECK-NEXT:             %[[SUBI_1:.*]] = arith.subi %[[VAL_0]], %[[CONSTANT_8]] : index
// CHECK-NEXT:             %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_9]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_8]]#2 size [1] to %[[ARG1]]{{\[}}%[[DIVSI_1]], %[[CONSTANT_8]]] size [1, 1] : !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, memref<2x1xf16, "L1">
// CHECK-NEXT:           } {applicable_units = ["L1SU"]}
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// The SIMD dims can be more than one input dim. On the sample device a SIMD vector is
// 64 f16 elements, so for tensor<1x4x32x2xf16> it is the 32x2 of the two
// innermost dims, and both of their reduction loop dims (d2, d3) stay out of
// chunking, not only the innermost one.
//
// total_input_bytes = 4 * 32 * 2 * 2 = 512; threshold=128 -> N=4, which only
// the 4-wide d1 is left to take: 4 chunks of one SIMD vector each.

#in  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#out = affine_map<(d0, d1, d2, d3) -> (d0)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @two_dim_simd(%src: memref<2x1x4x32x2xf16, "L1">, %dst: memref<2x1xf16, "L1">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %b = %c0 to %c2 step %c1 {
      ktdf.pipeline {
        %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, !ktdf.token, !ktdf.token) {
          %i = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
          %o = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %i, %o, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#2) {
          ktdf.data_transfer from %src[%b, 0, 0, 0, 0] size [1, 1, 4, 32, 2] to %p#0 size [256] : memref<2x1x4x32x2xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%p#2) depends_out(%p#3) {
          %x = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 256xf16> -> tensor<1x4x32x2xf16>
          %e = tensor.empty() : tensor<1xf16>
          %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction", "reduction", "reduction"]} ins(%x : tensor<1x4x32x2xf16>) outs(%e : tensor<1xf16>) {
          ^bb0(%a: f16, %acc: f16):
            %s = arith.addf %a, %acc : f16
            linalg.yield %s : f16
          } -> tensor<1xf16>
          ktdf.write_to_fifo %r, %p#1 : tensor<1xf16>, <"SFU" -> "L1SU", 1xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%p#3) depends_out(none) {
          ktdf.data_transfer from %p#1 size [1] to %dst[%b, 0] size [1, 1] : !ktdf.fifo.slot<"SFU" -> "L1SU", 1xf16>, memref<2x1xf16, "L1">
        } {applicable_units = ["L1SU"]}
      }
    }
    return
  }
}
