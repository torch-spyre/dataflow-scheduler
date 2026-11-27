// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -verify-diagnostics %s

// One kernel: which function's arguments are the run's inputs could otherwise
// only be guessed.

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index) attributes {grid = [2]} {
      return
    }
    // expected-error @+1 {{more than one function with a body in the orchestration module; @test is already its kernel}}
    func.func @other(%B: index) attributes {grid = [2]} {
      return
    }
  }
}
