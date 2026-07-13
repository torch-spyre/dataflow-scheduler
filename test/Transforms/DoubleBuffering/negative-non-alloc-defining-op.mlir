// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -double-buffering | FileCheck %s

// CHECK-LABEL:   func.func @non_alloc_defining_op(
// CHECK-SAME:        %[[SRC:[a-zA-Z0-9_]+]]: memref<64xf16, "DDR">, %[[DST:[a-zA-Z0-9_]+]]: memref<64xf16, "DDR">) {
// CHECK-NEXT:      %[[C0:.+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:.+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[C8:.+]] = arith.constant 8 : index
// CHECK-NEXT:      scf.for %[[I:.+]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK-NEXT:        ktdf.pipeline {
// CHECK-NEXT:          %[[TKS:.+]]:3 = ktdf.private -> (memref<*xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:            %[[ALLOC:.+]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:            %[[CAST:.+]] = memref.cast %[[ALLOC]] : memref<64xf16, "L1"> to memref<*xf16, "L1">
// CHECK-NEXT:            %[[T1:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            %[[T2:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:            ktdf.private_yield %[[CAST]], %[[T1]], %[[T2]] : memref<*xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(none) depends_out(%[[TKS]]#1) {
// CHECK-NEXT:            "test.write"(%[[TKS]]#0) : (memref<*xf16, "L1">) -> ()
// CHECK-NEXT:          }
// CHECK-NEXT:          ktdf.stage depends_in(%[[TKS]]#1) depends_out(%[[TKS]]#2) {
// CHECK-NEXT:            "test.read"(%[[TKS]]#0) : (memref<*xf16, "L1">) -> ()
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }

ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
func.func @non_alloc_defining_op(%src: memref<64xf16, "DDR">,
                                  %dst: memref<64xf16, "DDR">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %l1, %t1, %t2 = ktdf.private -> (
          memref<*xf16, "L1">,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, "L1">
        %a_cast = memref.cast %a : memref<64xf16, "L1"> to memref<*xf16, "L1">
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a_cast, %tk1, %tk2
          : memref<*xf16, "L1">, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1) {
        "test.write"(%l1) : (memref<*xf16, "L1">) -> ()
      }
      ktdf.stage depends_in(%t1) depends_out(%t2) {
        "test.read"(%l1) : (memref<*xf16, "L1">) -> ()
      }
    }
  }
  return
}
