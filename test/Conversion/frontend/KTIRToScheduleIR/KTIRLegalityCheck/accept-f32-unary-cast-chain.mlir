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

// RUN: dataflow-scheduler-opt --ktir-legality-check %s | FileCheck %s

// Exercise legality without canonicalization or downstream lowering.
// CHECK-DAG: #[[SWAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
#id = affine_map<(d0) -> (d0)>
#id2 = affine_map<(d0, d1) -> (d0, d1)>
#swap = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func.func @extf_exp_truncf
// CHECK:         %[[WIDE:.*]] = linalg.generic
// CHECK:           arith.extf
// CHECK:         %[[EXP:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[WIDE]] : tensor<64xf32>)
// CHECK:           spyreop.exp
// CHECK:         %[[NARROW:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[EXP]] : tensor<64xf32>)
// CHECK:           arith.truncf
// CHECK:         return %[[NARROW]] : tensor<64xf16>
func.func @extf_exp_truncf(%a: tensor<64xf16>, %f: tensor<64xf32>, %h: tensor<64xf16>) -> tensor<64xf16> {
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
  %narrow = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
      ins(%exp : tensor<64xf32>) outs(%h : tensor<64xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<64xf16>
  return %narrow : tensor<64xf16>
}

// Equal nonidentity maps change iteration order without changing indices.
// CHECK-LABEL: func.func @extf_rsqrt_truncf_same_permutation
// CHECK:         %[[WIDE2:.*]] = linalg.generic
// CHECK:           arith.extf
// CHECK:         %[[RSQRT:.*]] = linalg.generic
// CHECK-SAME:      indexing_maps = [#[[SWAP]], #[[SWAP]]]
// CHECK-SAME:      ins(%[[WIDE2]] : tensor<2x32xf32>)
// CHECK:           spyreop.rsqrt
// CHECK:         %[[NARROW2:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[RSQRT]] : tensor<2x32xf32>)
// CHECK:           arith.truncf
// CHECK:         return %[[NARROW2]] : tensor<2x32xf16>
func.func @extf_rsqrt_truncf_same_permutation(%a: tensor<2x32xf16>, %f: tensor<2x32xf32>, %h: tensor<2x32xf16>) -> tensor<2x32xf16> {
  %wide = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<2x32xf16>) outs(%f : tensor<2x32xf32>) {
  ^bb0(%in: f16, %out: f32):
    %e = arith.extf %in : f16 to f32
    linalg.yield %e : f32
  } -> tensor<2x32xf32>
  %rsqrt = linalg.generic {indexing_maps = [#swap, #swap], iterator_types = ["parallel", "parallel"]}
      ins(%wide : tensor<2x32xf32>) outs(%f : tensor<2x32xf32>) {
  ^bb0(%in: f32, %out: f32):
    %r = spyreop.rsqrt %in : f32
    linalg.yield %r : f32
  } -> tensor<2x32xf32>
  %narrow = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
      ins(%rsqrt : tensor<2x32xf32>) outs(%h : tensor<2x32xf16>) {
  ^bb0(%in: f32, %out: f16):
    %t = arith.truncf %in : f32 to f16
    linalg.yield %t : f16
  } -> tensor<2x32xf16>
  return %narrow : tensor<2x32xf16>
}
