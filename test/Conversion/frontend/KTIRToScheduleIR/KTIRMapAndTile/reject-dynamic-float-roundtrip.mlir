// RUN: dataflow-scheduler-opt --ktir-map-and-tile --verify-diagnostics %s

// Copyright 2026 The Torch-Spyre Authors.
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

// The inner dimension fills a vector, but the outer dimension remains dynamic.
// Reject before consumer tiling can introduce tensor.dim users of the widening
// result that would survive replacing its tile and erasing the original cast.
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @dynamic_float_roundtrip(%input: tensor<?x64xf16>) -> tensor<?x64xf16> attributes {grid = [1 : index]} {
    %c0 = arith.constant 0 : index
    %rows = tensor.dim %input, %c0 : tensor<?x64xf16>
    %wide_init = tensor.empty(%rows) : tensor<?x64xf32>
    %wide = linalg.generic {
        indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%input : tensor<?x64xf16>) outs(%wide_init : tensor<?x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %cast = arith.extf %in : f16 to f32
      linalg.yield %cast : f32
    } -> tensor<?x64xf32>
    %narrow_init = tensor.empty(%rows) : tensor<?x64xf16>
    // expected-error @+1 {{tiling error: a direct float conversion round trip requires static tensor shapes}}
    %narrow = linalg.generic {
        indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%wide : tensor<?x64xf32>) outs(%narrow_init : tensor<?x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %cast = arith.truncf %in : f32 to f16
      linalg.yield %cast : f16
    } -> tensor<?x64xf16>
    return %narrow : tensor<?x64xf16>
  }
}
