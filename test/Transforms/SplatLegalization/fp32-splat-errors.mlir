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

// RUN: dataflow-scheduler-opt --splat-legalization --split-input-file -verify-diagnostics %s

ktdf_arch.device @sample_device attributes {}
    import("../Inputs/fp32-splat-device.mlir")

func.func @overrun_at_end(
    %source: memref<32xf32, "L1">,
    %fifo: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // expected-error @below {{cannot prove widened splat load is within the source view}}
      ktdf.data_transfer from %source[29] size [1] to %fifo size [32]
          {transfer_mode = #ktdf.transfer_mode<splat>}
          : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
  }
  return
}

// -----

ktdf_arch.device @sample_device attributes {}
    import("../Inputs/fp32-splat-device.mlir")

func.func @narrow_source_view(
    %source: memref<1xf32, "L1">,
    %fifo: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // expected-error @below {{cannot prove widened splat load is within the source view}}
      ktdf.data_transfer from %source[0] size [1] to %fifo size [32]
          {transfer_mode = #ktdf.transfer_mode<splat>}
          : memref<1xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
  }
  return
}

// -----

ktdf_arch.device @sample_device attributes {}
    import("../Inputs/fp32-splat-device.mlir")

func.func @unknown_inner_index(
    %source: memref<32xf32, "L1">,
    %index: index,
    %fifo: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // expected-error @below {{cannot prove widened splat load is within the source view}}
      ktdf.data_transfer from %source[%index] size [1] to %fifo size [32]
          {transfer_mode = #ktdf.transfer_mode<splat>}
          : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
  }
  return
}

// -----

ktdf_arch.device @sample_device attributes {}
    import("../Inputs/fp32-splat-device.mlir")

func.func @strided_source(
    %source: memref<32xf32, strided<[2]>, "L1">,
    %fifo: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // expected-error @below {{widened splat requires a contiguous innermost source dimension}}
      ktdf.data_transfer from %source[0] size [1] to %fifo size [32]
          {transfer_mode = #ktdf.transfer_mode<splat>}
          : memref<32xf32, strided<[2]>, "L1">,
            !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
  }
  return
}
