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

// RUN: cat %S/../Inputs/fp32-splat-device.mlir %s > %t.valid.mlir
// RUN: dataflow-scheduler-opt --splat-legalization %t.valid.mlir | FileCheck %s --check-prefix=VALID
// RUN: sed 's/f32 = 4/f32 = 2/' %S/../Inputs/fp32-splat-device.mlir > %t.invalid.mlir
// RUN: cat %s >> %t.invalid.mlir
// RUN: not dataflow-scheduler-opt --splat-legalization %t.invalid.mlir 2>&1 | FileCheck %s --check-prefix=INVALID
// RUN: sed '/ktdf_arch.feature.load = {/,/^    },$/d' %S/../Inputs/fp32-splat-device.mlir > %t.noload.mlir
// RUN: cat %s >> %t.noload.mlir
// RUN: dataflow-scheduler-opt --splat-legalization %t.noload.mlir | FileCheck %s --check-prefix=NOLOAD

// A declared group of two cannot complete a repeated four-float load: every
// other group would copy the third input value instead of the scalar.
// INVALID: error: incompatible splat load width, FIFO width, or sub-SIMD group width
// A load unit that can splat but declares no load granularity splats the one
// element as asked: the transfer is left as it is and no shuffle is named.
// VALID-LABEL: func.func @declared_group(
// NOLOAD-LABEL: func.func @declared_group(
func.func @declared_group(
    %source: memref<32xf32, "L1">,
    %fifo: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
    %output: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // VALID: ktdf.data_transfer from %arg0[0] size [4] to %arg1 size [32]
      // NOLOAD: ktdf.data_transfer from %arg0[0] size [1] to %arg1 size [32]
      ktdf.data_transfer from %source[0] size [1] to %fifo size [32]
          {transfer_mode = #ktdf.transfer_mode<splat>}
          : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(none) depends_out(none) {
      // VALID: ktdf.read_from_fifo %arg1 {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
      // NOLOAD: ktdf.read_from_fifo %arg1 :
      // NOLOAD-NOT: splat =
      %read = ktdf.read_from_fifo %fifo
          : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      ktdf.write_to_fifo %read, %output
          : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}
