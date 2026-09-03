// RUN: dataflow-scheduler-opt --broadcast-promotion %s -allow-unregistered-dialect | FileCheck %s

// A loop and an scf.if that stay in the clone, each reading something computed
// inside the region being cloned: the loop its trip count, the scf.if its
// condition. Both come with the transfer, or the clone has a bound and a
// condition it cannot name.
//
// Neither op is read by the transfer, so neither is required for what it
// produces. They are kept for holding it, and what a kept op reads is required
// on that account alone.

// CHECK-LABEL:   func.func @keeps_what_a_kept_loop_reads
// CHECK:           ktdf.pipeline {
// CHECK-NEXT:        ktdf.stage
// CHECK-NEXT:          %[[TRIP:.*]] = "test.trip_count"()
// CHECK-NEXT:          %[[COND:.*]] = "test.gate"()
// CHECK-NEXT:          scf.if %[[COND]] {
// CHECK-NEXT:            scf.for %[[I:.*]] = %{{.*}} to %[[TRIP]] step %{{.*}} {
// CHECK-NEXT:              %[[FROM:.*]] = arith.addi %[[I]], %{{.*}}
// CHECK-NEXT:              ktdf.data_transfer from %{{.*}}{{\[}}%[[FROM]]]

module {
  func.func @keeps_what_a_kept_loop_reads(%A: memref<?xf16, "DDR">, %N: index) {
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
          %trip = "test.trip_count"() : () -> index
          %cond = "test.gate"() : () -> i1
          scf.if %cond {
            scf.for %i = %c0 to %trip step %c1 {
              %from = arith.addi %i, %c0 : index
              ktdf.data_transfer from %A[%from] size [64] to %r#0[%c0] size [64]
                : memref<?xf16, "DDR">, memref<64xf16, "L1">
            }
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
