// RUN: dataflow-scheduler-opt --pass-pipeline="builtin.module(builtin.module(func.func(ktir-pipeline)))" %s | FileCheck %s

// The declaration module calls the programs, in a loop here. The loop holds a
// call, not a pipeline, so ktir-pipeline leaves the declaration module alone
// rather than building a pipeline from it.

// CHECK-LABEL: func.func @kernel()
// CHECK:         scf.for
// CHECK-NEXT:      func.call @prog() : () -> ()
// CHECK-NEXT:    }
// CHECK-NOT:     ktdf.pipeline
// CHECK:         return

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @kernel() {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        func.call @prog() : () -> ()
      }
      return
    }
    func.func private @prog()
  }
}
