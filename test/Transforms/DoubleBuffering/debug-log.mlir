// Verify the pass emits debug output via LLVM_DEBUG when --debug-only is
// supplied.

// RUN: dataflow-scheduler-opt -allow-unregistered-dialect --debug-only=double-buffering %s -double-buffering 2>&1 | FileCheck %s

// CHECK: [double-buffering {{.*}}] starting

func.func @debug_log() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        ^bb0:
      }
    }
  }
  return
}
