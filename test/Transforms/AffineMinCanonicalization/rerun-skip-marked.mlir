// RUN: dataflow-scheduler-opt --affine-min-canonicalization -allow-unregistered-dialect %s | FileCheck %s

// A re-run: @old is a program an earlier run scheduled and marked, @new is
// one the run schedules, and the unnamed module is the orchestration module.
//
// The pass leaves @old as it is, its addi/minsi/apply chain and its unused
// constant included, since the greedy driver would otherwise erase dead ops
// there too. It rewrites the chain in the orchestration module and in @new.

// CHECK-LABEL: func.func @orch
// CHECK:         affine.min
// CHECK-NOT:     affine.apply

// CHECK-LABEL: module @old attributes {dataflow_scheduler.scheduled}
// CHECK:         func.func @old_body
// CHECK:           arith.constant 42 : index
// CHECK:           arith.addi
// CHECK:           arith.minsi
// CHECK:           affine.apply
// CHECK-NOT:       affine.min

// CHECK-LABEL: module @new
// CHECK:         func.func @new_body
// CHECK:           affine.min
// CHECK-NOT:       affine.apply

module {
  module {
    func.func @orch(%iv: index) {
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %add = arith.addi %iv, %c2 : index
      %min = arith.minsi %add, %c8 : index
      %r = affine.apply affine_map<()[s0, s1] -> (-s0 + s1)>()[%iv, %min]
      "test.use"(%r) : (index) -> ()
      return
    }
  }
  module @old attributes {dataflow_scheduler.scheduled} {
    func.func @old_body(%iv: index) {
      %unused = arith.constant 42 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %add = arith.addi %iv, %c2 : index
      %min = arith.minsi %add, %c8 : index
      %r = affine.apply affine_map<()[s0, s1] -> (-s0 + s1)>()[%iv, %min]
      "test.use"(%r) : (index) -> ()
      return
    }
  }
  module @new {
    func.func @new_body(%iv: index) {
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %add = arith.addi %iv, %c2 : index
      %min = arith.minsi %add, %c8 : index
      %r = affine.apply affine_map<()[s0, s1] -> (-s0 + s1)>()[%iv, %min]
      "test.use"(%r) : (index) -> ()
      return
    }
  }
}
