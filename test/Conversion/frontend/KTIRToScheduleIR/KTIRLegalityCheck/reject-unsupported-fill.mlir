// RUN: dataflow-scheduler-opt --ktir-legality-check --verify-diagnostics --split-input-file %s

// A linalg.fill is only legal when it can become a splat arith.constant: a
// constant into a statically shaped tensor.

func.func @fill_runtime_value(%v: f16) -> tensor<4xf16> {
  %e = tensor.empty() : tensor<4xf16>
  // expected-error @+1 {{V1 only supports linalg.fill of a constant into a statically shaped tensor}}
  %r = linalg.fill ins(%v : f16) outs(%e : tensor<4xf16>) -> tensor<4xf16>
  return %r : tensor<4xf16>
}

// -----

func.func @fill_dynamic_shape(%n: index) -> tensor<?xf16> {
  %c = arith.constant 1.0 : f16
  %e = tensor.empty(%n) : tensor<?xf16>
  // expected-error @+1 {{V1 only supports linalg.fill of a constant into a statically shaped tensor}}
  %r = linalg.fill ins(%c : f16) outs(%e : tensor<?xf16>) -> tensor<?xf16>
  return %r : tensor<?xf16>
}
