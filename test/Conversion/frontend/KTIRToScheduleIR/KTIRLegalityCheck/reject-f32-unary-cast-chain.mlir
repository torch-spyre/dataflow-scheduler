//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

// RUN: dataflow-scheduler-opt --ktir-legality-check --split-input-file --verify-diagnostics %s

// The unary result cannot also feed ordinary FP32 computation.
#id = affine_map<(d0) -> (d0)>
func.func @unary_fanout(%a: tensor<64xf16>, %f: tensor<64xf32>, %h: tensor<64xf16>) -> (tensor<64xf16>, tensor<64xf32>) {
  // expected-error @+1 {{V1 only supports a dl16tofp32 conversion whose result is stored, not computed on}}
  %wide = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%a : tensor<64xf16>) outs(%f : tensor<64xf32>) {
  ^bb0(%in: f16, %out: f32):
    %e = arith.extf %in : f16 to f32
    linalg.yield %e : f32
  } -> tensor<64xf32>
  %exp = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%wide : tensor<64xf32>) outs(%f : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %e = spyreop.exp %in : f32
    linalg.yield %e : f32
  } -> tensor<64xf32>
  %sum = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%exp : tensor<64xf32>) outs(%f : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %in : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  %narrow = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%exp : tensor<64xf32>) outs(%h : tensor<64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %narrow, %sum : tensor<64xf16>, tensor<64xf32>
}

// -----

// A transpose changes the logical indices even when both maps are permutations.
#id = affine_map<(d0, d1) -> (d0, d1)>
#swap = affine_map<(d0, d1) -> (d1, d0)>
func.func @unary_transpose(%a: tensor<8x8xf16>, %f: tensor<8x8xf32>, %h: tensor<8x8xf16>) -> tensor<8x8xf16> {
  // expected-error @+1 {{V1 only supports a dl16tofp32 conversion whose result is stored, not computed on}}
  %wide = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<8x8xf16>) outs(%f : tensor<8x8xf32>) {
  ^bb0(%in: f16, %out: f32):
    %e = arith.extf %in : f16 to f32
    linalg.yield %e : f32
  } -> tensor<8x8xf32>
  %exp = linalg.generic {indexing_maps = [#id, #swap], iterator_types = ["parallel", "parallel"]}
      ins(%wide : tensor<8x8xf32>) outs(%f : tensor<8x8xf32>) {
  ^bb0(%in: f32, %out: f32):
    %e = spyreop.exp %in : f32
    linalg.yield %e : f32
  } -> tensor<8x8xf32>
  %narrow = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%exp : tensor<8x8xf32>) outs(%h : tensor<8x8xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<8x8xf16>
  return %narrow : tensor<8x8xf16>
}

// -----

// The exception requires the leading standalone widening conversion.
#id = affine_map<(d0) -> (d0)>
func.func @unary_without_widening(%a: tensor<64xf32>, %f: tensor<64xf32>, %h: tensor<64xf16>) -> tensor<64xf16> {
  %exp = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%a : tensor<64xf32>) outs(%f : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %e = spyreop.exp %in : f32
    linalg.yield %e : f32
  } -> tensor<64xf32>
  // expected-error @+1 {{V1 only supports a fp32todl16 conversion whose input is loaded, not computed}}
  %narrow = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%exp : tensor<64xf32>) outs(%h : tensor<64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %narrow : tensor<64xf16>
}

// -----

// One tensor input does not make a two-operand scalar operation unary.
#id = affine_map<(d0) -> (d0)>
func.func @binary_between_casts(%a: tensor<64xf16>, %f: tensor<64xf32>, %h: tensor<64xf16>) -> tensor<64xf16> {
  // expected-error @+1 {{V1 only supports a dl16tofp32 conversion whose result is stored, not computed on}}
  %wide = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%a : tensor<64xf16>) outs(%f : tensor<64xf32>) {
  ^bb0(%in: f16, %out: f32):
    %e = arith.extf %in : f16 to f32
    linalg.yield %e : f32
  } -> tensor<64xf32>
  %sum = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%wide : tensor<64xf32>) outs(%f : tensor<64xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %in : f32
    linalg.yield %s : f32
  } -> tensor<64xf32>
  %narrow = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%sum : tensor<64xf32>) outs(%h : tensor<64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %narrow : tensor<64xf16>
}
