// XFAIL:*
// RUN: mkdir -p %t
// RUN: dataflow-scheduler-opt -allow-unregistered-dialect --emit-split-dfir='output-dir=%t' %s
// RUN: FileCheck %s --input-file=%t/global.mlir --check-prefix=GLOBAL
// RUN: FileCheck %s --input-file=%t/local-schedule-0.mlir --check-prefix=IMPL

// GLOBAL-LABEL: module {
// GLOBAL: func.func @Add_155
// GLOBAL: call @"local-schedule-0"
// GLOBAL: func.func private @"local-schedule-0"()
// GLOBAL-NOT: dataflow.program_unit

// IMPL-LABEL: module {
// IMPL: func.func private @"local-schedule-0"
// IMPL: dataflow.program_unit
// IMPL-NOT: func.func @Add_155

#map = affine_map<(d0) -> (d0)>
module {
  module {
    func.func @Add_155() attributes {grid = [1]} {
      call @"local-schedule-0"() : () -> ()
      return
    }
    func.func private @"local-schedule-0"()
  }
  module {
    func.func private @"local-schedule-0"() attributes {grid = [1]} {
      %0 = dataflow.get_unit {name = "DDR", type = "DDR"} : index
      dataflow.program_unit iter_arg : %arg0 -> (%0) : {
        "test.op"() : () -> ()
      }
      return
    }
  }
}
