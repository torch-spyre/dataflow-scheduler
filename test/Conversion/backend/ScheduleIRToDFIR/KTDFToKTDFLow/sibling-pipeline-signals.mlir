// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdf-to-ktdflowering)" -allow-unregistered-dialect %s | FileCheck %s

// The writer and reader run in separate pipelines on different units. Their
// synchronization must be between the pipelines: inside the producer pipeline,
// the reader unit would never execute its half of the signal.
// CHECK-LABEL: func.func @sibling_signal_in_loop(
// CHECK: %[[WRITER:.*]] = uniform.query_map
// CHECK: %[[READER:.*]] = uniform.query_map
// CHECK: scf.for
// CHECK-NEXT: ktdf_lowering.execute_on %[[WRITER]] {
// CHECK-NEXT:   ktdf_lowering.execute_on %[[WRITER]] {
// CHECK-NEXT:     ktdf.data_transfer
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: ktdf_lowering.signal %[[WRITER]], %[[READER]]{{$}}
// CHECK-NEXT: ktdf_lowering.execute_on %[[READER]] {
// CHECK-NEXT:   ktdf_lowering.execute_on %[[READER]] {
// CHECK-NEXT:     "test.use"
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: return

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @sibling_signal_in_loop(%input: memref<64xf16, "DDR">,
                                    %scratch: memref<64xf16, "L1">,
                                    %count: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %count step %c1 {
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          ktdf.data_transfer from %input[%c0] size [64] to %scratch[%c0] size [64]
            : memref<64xf16, "DDR">, memref<64xf16, "L1">
        } {applicable_units = ["MNILU"]}
      }
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          "test.use"(%scratch) : (memref<64xf16, "L1">) -> ()
        } {applicable_units = ["L1LU"]}
      }
    }
    return
  }

  // A common outer stage must not pull the signal out of the inner loop. The
  // signal still belongs between the two inner pipelines on every iteration.
  // CHECK-LABEL: func.func @nested_sibling_signal_in_loop(
  // CHECK: %[[NESTED_WRITER:.*]] = uniform.query_map
  // CHECK: %[[NESTED_READER:.*]] = uniform.query_map
  // CHECK: ktdf_lowering.execute_on %[[NESTED_WRITER]], %[[NESTED_READER]] {
  // CHECK-NEXT: ktdf_lowering.execute_on %[[NESTED_WRITER]], %[[NESTED_READER]] {
  // CHECK-NEXT: scf.for
  // CHECK-NEXT: ktdf_lowering.execute_on %[[NESTED_WRITER]] {
  // CHECK-NEXT:   ktdf_lowering.execute_on %[[NESTED_WRITER]] {
  // CHECK-NEXT:     ktdf.data_transfer
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: ktdf_lowering.signal %[[NESTED_WRITER]], %[[NESTED_READER]]{{$}}
  // CHECK-NEXT: ktdf_lowering.execute_on %[[NESTED_READER]] {
  // CHECK-NEXT:   ktdf_lowering.execute_on %[[NESTED_READER]] {
  // CHECK-NEXT:     "test.use"
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  func.func @nested_sibling_signal_in_loop(%input: memref<64xf16, "DDR">,
                                           %scratch: memref<64xf16, "L1">,
                                           %count: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        scf.for %i = %c0 to %count step %c1 {
          ktdf.pipeline {
            ktdf.stage depends_in(none) depends_out(none) {
              ktdf.data_transfer from %input[%c0] size [64] to %scratch[%c0] size [64]
                : memref<64xf16, "DDR">, memref<64xf16, "L1">
            } {applicable_units = ["MNILU"]}
          }
          ktdf.pipeline {
            ktdf.stage depends_in(none) depends_out(none) {
              "test.use"(%scratch) : (memref<64xf16, "L1">) -> ()
            } {applicable_units = ["L1LU"]}
          }
        }
      } {applicable_units = ["MNILU", "L1LU"]}
    }
    return
  }
}
