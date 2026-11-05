// RUN: dataflow-scheduler-opt --ktir-legality-check --split-input-file --verify-diagnostics %s
#map = affine_map<(d0) -> (d0)>
// Only f16 <-> f32 has a device instruction: a conversion between any other
// widths is still rejected.
func.func @bf16_to_f32(%a: tensor<64xbf16>, %o: tensor<64xf32>) -> tensor<64xf32> {
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%a : tensor<64xbf16>) outs(%o : tensor<64xf32>) {
  ^bb0(%in: bf16, %out: f32):
    // expected-error @+1 {{V1 only supports add/mul/sub compute ops; found unsupported compute op}}
    %e = arith.extf %in : bf16 to f32
    linalg.yield %e : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

// The device converts whole vectors, so a conversion is the only operation of
// its linalg.generic: one fused with other compute is rejected.
#map = affine_map<(d0) -> (d0)>
func.func @fused_extf(%a: tensor<64xf16>, %o: tensor<64xf32>) -> tensor<64xf32> {
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%a : tensor<64xf16>) outs(%o : tensor<64xf32>) {
  ^bb0(%in: f16, %out: f32):
    // expected-error @+1 {{V1 only supports a conversion between f16 and f32 as the only operation of its linalg.generic, converting its one input}}
    %e = arith.extf %in : f16 to f32
    %d = arith.addf %e, %e : f32
    linalg.yield %d : f32
  } -> tensor<64xf32>
  return %r : tensor<64xf32>
}

// -----

// The f32 side of a conversion moves as two vectors, which only memory takes
// apart: a dl16tofp32 whose result is computed on is rejected.
#map = affine_map<(d0) -> (d0)>
func.func @extf_then_compute(%a: tensor<64xf16>, %o: tensor<64xf32>) -> tensor<64xf32> {
  // expected-error @+1 {{V1 only supports a dl16tofp32 conversion whose result is stored, not computed on}}
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%a : tensor<64xf16>) outs(%o : tensor<64xf32>) {
  ^bb0(%in: f16, %out: f32):
    %e = arith.extf %in : f16 to f32
    linalg.yield %e : f32
  } -> tensor<64xf32>
  %s = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%r : tensor<64xf32>) outs(%o : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %d = arith.addf %in, %in : f32
    linalg.yield %d : f32
  } -> tensor<64xf32>
  return %s : tensor<64xf32>
}

// -----

// Likewise a fp32todl16 whose input is computed.
#map = affine_map<(d0) -> (d0)>
func.func @compute_then_truncf(%a: tensor<64xf32>, %o: tensor<64xf32>, %h: tensor<64xf16>) -> tensor<64xf16> {
  %s = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%a : tensor<64xf32>) outs(%o : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %d = arith.addf %in, %in : f32
    linalg.yield %d : f32
  } -> tensor<64xf32>
  // expected-error @+1 {{V1 only supports a fp32todl16 conversion whose input is loaded, not computed}}
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%s : tensor<64xf32>) outs(%h : tensor<64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %r : tensor<64xf16>
}

// -----

// A standalone narrowing consumer does not make another compute use legal.
#map = affine_map<(d0) -> (d0)>
func.func @extf_to_truncf_and_compute(%a: tensor<64xf16>, %f: tensor<64xf32>, %h: tensor<64xf16>) -> (tensor<64xf16>, tensor<64xf32>) {
  // expected-error @+1 {{V1 only supports a dl16tofp32 conversion whose result is stored, not computed on}}
  %wide = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%a : tensor<64xf16>) outs(%f : tensor<64xf32>) {
  ^bb0(%in: f16, %out: f32):
    %e = arith.extf %in : f16 to f32
    linalg.yield %e : f32
  } -> tensor<64xf32>
  %sum = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%wide : tensor<64xf32>) outs(%f : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %d = arith.addf %in, %in : f32
    linalg.yield %d : f32
  } -> tensor<64xf32>
  %narrow = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%wide : tensor<64xf32>) outs(%h : tensor<64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %narrow, %sum : tensor<64xf16>, tensor<64xf32>
}

// -----

// A consumer containing both arithmetic and a cast is not standalone.
#map = affine_map<(d0) -> (d0)>
func.func @extf_to_fused_truncf(%a: tensor<64xf16>, %f: tensor<64xf32>, %h: tensor<64xf16>) -> tensor<64xf16> {
  // expected-error @+1 {{V1 only supports a dl16tofp32 conversion whose result is stored, not computed on}}
  %wide = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%a : tensor<64xf16>) outs(%f : tensor<64xf32>) {
  ^bb0(%in: f16, %out: f32):
    %e = arith.extf %in : f16 to f32
    linalg.yield %e : f32
  } -> tensor<64xf32>
  %narrow = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
      ins(%wide : tensor<64xf32>) outs(%h : tensor<64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %d = arith.addf %in, %in : f32
    %t = arith.truncf %d : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %narrow : tensor<64xf16>
}
