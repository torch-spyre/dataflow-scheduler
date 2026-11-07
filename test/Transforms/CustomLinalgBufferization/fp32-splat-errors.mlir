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

// RUN: dataflow-scheduler-opt --custom-linalg-bufferization --split-input-file -verify-diagnostics %s

ktdf_arch.device @sample_device attributes {}
    import("../Inputs/fp32-splat-device.mlir")

#id = affine_map<(i) -> (i)>
func.func @two_stick_splat(%input: !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf32>,
                           %output: memref<64xf32, "SFU_REG">) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // expected-error @below {{splat register requires a static 32-element 'f32' stick}}
      %r = ktdf.read_from_fifo %input
          {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
          : <"L1LU" -> "SFU", 64xf32> -> memref<64xf32>
      linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel"]
      } ins(%r, %r : memref<64xf32>, memref<64xf32>)
          outs(%output : memref<64xf32, "SFU_REG">) {
      ^bb0(%x: f32, %y: f32, %o: f32):
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
    } {applicable_units = ["SFU"]}
  }
  return
}

// -----

// A compute unit without its local register file cannot materialize an
// arithmetic input.
ktdf_arch.device @no_register {
  group {kind = "core"} share() {
    %unit = exec_unit {kind = "SFU", ktdf_arch.features = {
      ktdf_arch.feature.compute,
      ktdf_arch.feature.simd = {lanes = #ktdf_arch.map<f32 = 32>}
    }}
  }
}

#id = affine_map<(i) -> (i)>
func.func @missing_register(%input: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
                            %output: memref<32xf32>) {
  ktdf.pipeline {
    // expected-error @below {{no unambiguous local memory found for exec_unit kind '"SFU"'}}
    ktdf.stage depends_in(none) depends_out(none) {
      %r = ktdf.read_from_fifo %input
          {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
          : <"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel"]
      } ins(%r, %r : memref<32xf32>, memref<32xf32>)
          outs(%output : memref<32xf32>) {
      ^bb0(%x: f32, %y: f32, %o: f32):
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
    } {applicable_units = ["SFU"]}
  }
  return
}
