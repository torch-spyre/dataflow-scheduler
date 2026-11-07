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

// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(access-granularity-legalization,scalar-broadcast-legalization,splat-legalization)" %s | FileCheck %s

ktdf_arch.device @sample_device attributes {}
    import("../Inputs/fp32-splat-device.mlir")

#id = affine_map<(i) -> (i)>
#broadcast = affine_map<(i) -> (0)>

// Run the three passes in pipeline order: load legalization must not shrink
// the 32-element FIFO to the four elements required by the physical load.
// CHECK: #[[ID:.*]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL: func.func @keep_stick_width(
// CHECK-SAME: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
func.func @keep_stick_width(
    %source: memref<32xf32, "L1">,
    %fifo: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
    %output: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: ktdf.data_transfer from %arg0[0] size [4] to %arg1 size [32]
      // CHECK-SAME: transfer_mode = #ktdf.transfer_mode<splat>
      ktdf.data_transfer from %source[0] size [1] to %fifo size [32]
          {transfer_mode = #ktdf.transfer_mode<splat>}
          : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: %[[READ:.*]] = ktdf.read_from_fifo %arg1
      // CHECK-SAME: {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
      // CHECK-SAME: -> tensor<32xf32>
      %read = ktdf.read_from_fifo %fifo
          : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      %empty = tensor.empty() : tensor<32xf32>
      // CHECK: linalg.generic {indexing_maps = [#[[ID]], #[[ID]]]
      // CHECK-SAME: ins(%[[READ]] : tensor<32xf32>)
      %result = linalg.generic {
        indexing_maps = [#broadcast, #id],
        iterator_types = ["parallel"]
      } ins(%read : tensor<32xf32>) outs(%empty : tensor<32xf32>) {
      ^bb0(%value: f32, %out: f32):
        linalg.yield %value : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %result, %output
          : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// CHECK-LABEL: func.func @nonzero_offset(
func.func @nonzero_offset(
    %source: memref<32xf32, "L1">,
    %fifo: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
    %output: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: ktdf.data_transfer from %arg0[1] size [4] to %arg1 size [32]
      ktdf.data_transfer from %source[1] size [1] to %fifo size [32]
          {transfer_mode = #ktdf.transfer_mode<splat>}
          : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: ktdf.read_from_fifo %arg1 {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
      %read = ktdf.read_from_fifo %fifo
          : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      ktdf.write_to_fifo %read, %output
          : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// CHECK-LABEL: func.func @last_full_granule(
func.func @last_full_granule(
    %source: memref<32xf32, "L1">,
    %fifo: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
    %output: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: ktdf.data_transfer from %arg0[28] size [4] to %arg1 size [32]
      ktdf.data_transfer from %source[28] size [1] to %fifo size [32]
          {transfer_mode = #ktdf.transfer_mode<splat>}
          : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: ktdf.read_from_fifo %arg1 {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
      %read = ktdf.read_from_fifo %fifo
          : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      ktdf.write_to_fifo %read, %output
          : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// Only the innermost dimension widens; the symbolic outer coordinate is kept.
// CHECK-LABEL: func.func @symbolic_outer(
func.func @symbolic_outer(
    %source: memref<?x32xf32, "L1">,
    %row: index,
    %fifo: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
    %output: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: ktdf.data_transfer from %arg0[%arg1, 0] size [1, 4] to %arg2 size [32]
      ktdf.data_transfer from %source[%row, 0] size [1, 1] to %fifo size [32]
          {transfer_mode = #ktdf.transfer_mode<splat>}
          : memref<?x32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: ktdf.read_from_fifo %arg2 {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
      // CHECK-SAME: -> tensor<1x32xf32>
      %read = ktdf.read_from_fifo %fifo
          : <"L1LU" -> "SFU", 32xf32> -> tensor<1x32xf32>
      ktdf.write_to_fifo %read, %output
          : tensor<1x32xf32>, <"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// ValueBounds proves 0 <= i <= 28, including the last legal four-element load.
// CHECK-LABEL: func.func @bounded_inner_index(
func.func @bounded_inner_index(
    %source: memref<32xf32, "L1">,
    %fifo: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
    %output: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c29 = arith.constant 29 : index
  scf.for %i = %c0 to %c29 step %c1 {
    ktdf.pipeline {
      ktdf.stage depends_in(none) depends_out(none) {
        // CHECK: ktdf.data_transfer from %arg0[%{{.*}}] size [4] to %arg1 size [32]
        ktdf.data_transfer from %source[%i] size [1] to %fifo size [32]
            {transfer_mode = #ktdf.transfer_mode<splat>}
            : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(none) depends_out(none) {
        // CHECK: ktdf.read_from_fifo %arg1 {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
        %read = ktdf.read_from_fifo %fifo
            : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
        ktdf.write_to_fifo %read, %output
            : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
      } {applicable_units = ["SFU"]}
    }
  }
  return
}

// The load unit declares no granularity for this memory, so it loads the one
// element as asked: nothing to widen and no shuffle to name.
// CHECK-LABEL: func.func @undeclared_source_memory(
func.func @undeclared_source_memory(
    %source: memref<32xf32, "UNKNOWN">,
    %fifo: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
    %output: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: ktdf.data_transfer from %arg0[0] size [1] to %arg1 size [32]
      ktdf.data_transfer from %source[0] size [1] to %fifo size [32]
          {transfer_mode = #ktdf.transfer_mode<splat>}
          : memref<32xf32, "UNKNOWN">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: ktdf.read_from_fifo %arg1 :
      // CHECK-NOT: splat =
      %read = ktdf.read_from_fifo %fifo
          : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      ktdf.write_to_fifo %read, %output
          : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}
