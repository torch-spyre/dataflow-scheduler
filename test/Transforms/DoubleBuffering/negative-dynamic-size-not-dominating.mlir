// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-LABEL:   func.func @dynamic_size_not_dominating(
// CHECK-SAME:        %[[SRC:[a-zA-Z0-9_]+]]: memref<?xf16, "DDR">, %[[DST:[a-zA-Z0-9_]+]]: memref<?xf16, "DDR">, %[[M:[a-zA-Z0-9_]+]]: index) {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[C8:.+]] = arith.constant 8 : index
// CHECK-NEXT:      scf.for %[[I:.+]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK-NEXT:        %[[SIZE:.+]] = arith.addi %[[I]], %[[M]] : index
// CHECK-NEXT:        ktdf.pipeline {
// CHECK-NEXT:          %[[TKS:.+]]:3 = ktdf.private -> (memref<?xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:            %[[ALLOC:.+]] = memref.alloc(%[[SIZE]]) : memref<?xf16, "L1">
// CHECK-NEXT:            %[[T1:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            %[[T2:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            ktdf.private_yield %[[ALLOC]], %[[T1]], %[[T2]] : memref<?xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(none) depends_out(%[[TKS]]#1) {
// CHECK-NEXT:            ktdf.data_transfer from %[[SRC]][%[[C0]]] size [%[[SIZE]]] to %[[TKS]]#0[%[[C0]]] size [%[[SIZE]]] : memref<?xf16, "DDR">, memref<?xf16, "L1">
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(%[[TKS]]#1) depends_out(%[[TKS]]#2) {
// CHECK-NEXT:            ktdf.data_transfer from %[[TKS]]#0[%[[C0]]] size [%[[SIZE]]] to %[[DST]][%[[C0]]] size [%[[SIZE]]] : memref<?xf16, "L1">, memref<?xf16, "DDR">
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
func.func @dynamic_size_not_dominating(%src: memref<?xf16, "DDR">,
                                        %dst: memref<?xf16, "DDR">,
                                        %M: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    // Compute size INSIDE the loop — does not dominate the hoist target.
    %size = arith.addi %i, %M : index
    ktdf.pipeline {
      %l1, %t1, %t2 = ktdf.private -> (
          memref<?xf16, "L1">,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc(%size) : memref<?xf16, "L1">
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2
          : memref<?xf16, "L1">, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1) {
        ktdf.data_transfer from %src[%c0] size [%size] to %l1[%c0] size [%size]
          : memref<?xf16, "DDR">,
            memref<?xf16, "L1">
      }
      ktdf.stage depends_in(%t1) depends_out(%t2) {
        ktdf.data_transfer from %l1[%c0] size [%size] to %dst[%c0] size [%size]
          : memref<?xf16, "L1">,
            memref<?xf16, "DDR">
      }
    }
  }
  return
}
