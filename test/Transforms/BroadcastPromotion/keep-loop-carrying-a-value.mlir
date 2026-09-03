// RUN: dataflow-scheduler-opt --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// A transfer under a loop that carries a value out. The loop stays in the clone,
// carrying what it did, and what its yield hands back is kept with it.
//
// Neither is optional. A loop whose iterations are not interchangeable cannot be
// collapsed into one run of its body, and a region that stays has to go on
// answering what its terminator answered -- prune the accumulator away and the
// clone comes out holding a yield with a null operand.

// CHECK-LABEL:   func.func @keeps_loop_carrying_a_value
// The hoisted clone, in front of the loop it was lifted out of.
// CHECK:           ktdf.pipeline {
// CHECK-NEXT:        ktdf.stage
// CHECK-NEXT:          scf.for %[[I:.*]] = %[[C0:.*]] to %{{.*}} step %[[C1:.*]] {
// CHECK-NEXT:            %{{.*}} = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ACC:.*]] = %[[C0]]) -> (index) {
// CHECK-NEXT:              %[[FROM:.*]] = arith.addi %[[I]], %[[C0]]
// CHECK-NEXT:              %[[INTO:.*]] = arith.addi %[[I]], %[[C1]]
// CHECK-NEXT:              ktdf.data_transfer from %{{.*}}{{\[}}%[[FROM]]] size [64] to %{{.*}}{{\[}}%[[INTO]]] size [64]
// CHECK-NEXT:              %[[NEXT:.*]] = arith.addi %[[ACC]], %[[C1]]
// CHECK-NEXT:              scf.yield %[[NEXT]] : index

module {
  func.func @keeps_loop_carrying_a_value(%A: memref<?xf16, "DDR">,
                                         %M: index, %N: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %n = %c0 to %N step %c1 {
      ktdf.pipeline {
        %r:2 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.token) {
          %a = memref.alloc() : memref<64xf16, "L1">
          %k = ktdf.create_token : !ktdf.token
          ktdf.private_yield %a, %k : memref<64xf16, "L1">, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%r#1) {
          scf.for %i = %c0 to %M step %c1 {
            %count = scf.for %j = %c0 to %M step %c1
                iter_args(%acc = %c0) -> (index) {
              %from = arith.addi %i, %c0 : index
              %into = arith.addi %i, %c1 : index
              ktdf.data_transfer from %A[%from] size [64] to %r#0[%into] size [64]
                : memref<?xf16, "DDR">, memref<64xf16, "L1">
              %next = arith.addi %acc, %c1 : index
              scf.yield %next : index
            }
            "test.use"(%count) : (index) -> ()
          }
        }
        ktdf.stage depends_in(%r#1) depends_out(none) {
          "test.use"(%r#0) : (memref<64xf16, "L1">) -> ()
        }
      }
    }
    return
  }

}
