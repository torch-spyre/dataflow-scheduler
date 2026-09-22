// RUN: dataflow-scheduler-opt --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// The indirect-address-buffer fill is a plain data_transfer into a memref and
// is invariant in %n, so it looks like a perfectly good hoist candidate. It is
// not: the IAB is local state of the unit running the stage, so the fill must
// stay in the same stage as the ind_data_transfer that dereferences it.
// IR after the pass must be byte-equivalent to the input.

// CHECK-LABEL:   func.func @no_hoist_iab_fill
// CHECK-SAME:        %[[M:[^,]+]]: index, %[[N:[^,]+]]: index,
// CHECK-SAME:        %[[ADDRS:[^,]+]]: memref<32xindex, "DDR">,
// CHECK-SAME:        %[[IAB:[^,]+]]: memref<32xindex, "IAB">,
// CHECK-SAME:        %[[DATA:[^,]+]]: memref<64x64xf16, "DDR">) {
// CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     scf.for %[[IM:.*]] = %[[C0]] to %[[M]] step %[[C1]] {
// CHECK-NEXT:       scf.for %[[IN:.*]] = %[[C0]] to %[[N]] step %[[C1]] {
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[PRIV:.*]]:2 = ktdf.private -> (memref<1x64xf16, "L1">, !ktdf.token) {
// CHECK-NEXT:             %[[ALLOC:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:             %[[TOK:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[ALLOC]], %[[TOK]] : memref<1x64xf16, "L1">, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[PRIV]]#1) {
// CHECK-NEXT:             ktdf.data_transfer from %[[ADDRS]]{{\[}}%[[C0]]] size [32] to %[[IAB]]{{\[}}%[[C0]]] size [32] : memref<32xindex, "DDR">, memref<32xindex, "IAB">
// CHECK-NEXT:             ktdf.ind_data_transfer
// CHECK-NEXT:               ind_src = %[[IAB]]{{\[}}%[[IM]]]
// CHECK-NEXT:               dir_src = %[[DATA]]{{\[}}%[[C0]], %[[C0]]] size [1, 64]
// CHECK-NEXT:               ind_dst = none
// CHECK-NEXT:               dir_dst = %[[PRIV]]#0{{\[}}%[[C0]], %[[C0]]] size [1, 64]
// CHECK-NEXT:               : memref<32xindex, "IAB">, memref<64x64xf16, "DDR">, none, memref<1x64xf16, "L1">
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

module {
  func.func @no_hoist_iab_fill(%M: index, %N: index,
                               %addrs: memref<32xindex, "DDR">,
                               %iab: memref<32xindex, "IAB">,
                               %data: memref<64x64xf16, "DDR">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %m = %c0 to %M step %c1 {
      scf.for %n = %c0 to %N step %c1 {
        ktdf.pipeline {
          %l1, %t = ktdf.private -> (memref<1x64xf16, "L1">, !ktdf.token) {
            %a = memref.alloc() : memref<1x64xf16, "L1">
            %k = ktdf.create_token : !ktdf.token
            ktdf.private_yield %a, %k : memref<1x64xf16, "L1">, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%t) {
            // Invariant in %n and destined for a memref, but it is the IAB
            // fill for the transfer below -> never hoisted.
            ktdf.data_transfer from %addrs[%c0] size [32] to %iab[%c0] size [32]
              : memref<32xindex, "DDR">, memref<32xindex, "IAB">
            ktdf.ind_data_transfer
                ind_src = %iab[%m]
                dir_src = %data[%c0, %c0] size [1, 64]
                ind_dst = none
                dir_dst = %l1[%c0, %c0]   size [1, 64]
                : memref<32xindex, "IAB">, memref<64x64xf16, "DDR">, none, memref<1x64xf16, "L1">
          }
        }
      }
    }
    return
  }
}
