// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-LABEL:   func.func @multi_producer(
// CHECK-SAME:        %[[SRC1:[a-zA-Z0-9_]+]]: memref<64xf16, "DDR">, %[[SRC2:[a-zA-Z0-9_]+]]: memref<64xf16, "DDR">, %[[DST:[a-zA-Z0-9_]+]]: memref<64xf16, "DDR">) {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[C8:.+]] = arith.constant 8 : index
// CHECK-NEXT:      scf.for %[[I:.+]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK-NEXT:        ktdf.pipeline {
// CHECK-NEXT:          %[[TKS:.+]]:4 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:            %[[ALLOC:.+]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:            %[[T1:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            %[[T2:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            %[[T3:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            ktdf.private_yield %[[ALLOC]], %[[T1]], %[[T2]], %[[T3]] : memref<64xf16, "L1">, !ktdf.token, !ktdf.token, !ktdf.token
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(none) depends_out(%[[TKS]]#1) {
// CHECK-NEXT:            ktdf.data_transfer from %[[SRC1]][%[[C0]]] size [64] to %[[TKS]]#0[%[[C0]]] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(none) depends_out(%[[TKS]]#2) {
// CHECK-NEXT:            ktdf.data_transfer from %[[SRC2]][%[[C0]]] size [64] to %[[TKS]]#0[%[[C0]]] size [64] : memref<64xf16, "DDR">, memref<64xf16, "L1">
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(%[[TKS]]#1, %[[TKS]]#2) depends_out(%[[TKS]]#3) {
// CHECK-NEXT:            ktdf.data_transfer from %[[TKS]]#0[%[[C0]]] size [64] to %[[DST]][%[[C0]]] size [64] : memref<64xf16, "L1">, memref<64xf16, "DDR">
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
func.func @multi_producer(%src1: memref<64xf16, "DDR">,
                          %src2: memref<64xf16, "DDR">,
                          %dst: memref<64xf16, "DDR">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %l1, %t1, %t2, %t3 = ktdf.private -> (
          memref<64xf16, "L1">,
          !ktdf.token, !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, "L1">
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        %tk3 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2, %tk3
          : memref<64xf16, "L1">, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1) {
        ktdf.data_transfer from %src1[%c0] size [64] to %l1[%c0] size [64]
          : memref<64xf16, "DDR">,
            memref<64xf16, "L1">
      }
      ktdf.stage depends_in(none) depends_out(%t2) {
        ktdf.data_transfer from %src2[%c0] size [64] to %l1[%c0] size [64]
          : memref<64xf16, "DDR">,
            memref<64xf16, "L1">
      }
      ktdf.stage depends_in(%t1, %t2) depends_out(%t3) {
        ktdf.data_transfer from %l1[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, "L1">,
            memref<64xf16, "DDR">
      }
    }
  }
  return
}
