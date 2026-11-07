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

// RUN: dataflow-scheduler-opt --custom-linalg-bufferization %s | FileCheck %s

ktdf_arch.device @sample_device attributes {}
    import("../Inputs/fp32-splat-device.mlir")

#id = affine_map<(i) -> (i)>

// Only the splat gets a register; the ordinary read keeps its FIFO position and
// the arithmetic result keeps tensor semantics.
// CHECK-LABEL: func.func @one_splat(
func.func @one_splat(%vector: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
                     %scalar: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
                     %output: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: %[[V:.*]] = ktdf.read_from_fifo
      // CHECK-SAME: -> tensor<32xf32>
      // CHECK-NEXT: %[[REG:.*]] = memref.alloc() : memref<32xf32, "SFU_REG">
      // CHECK-NEXT: %[[READ:.*]] = ktdf.read_from_fifo
      // CHECK-SAME: {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
      // CHECK-SAME: -> memref<32xf32>
      // CHECK-NEXT: memref.copy %[[READ]], %[[REG]]
      // CHECK-NEXT: %[[S:.*]] = bufferization.to_tensor %[[REG]] restrict
      // CHECK-SAME: {dataflow_scheduler.splat_register_view}
      %v = ktdf.read_from_fifo %vector : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      %s = ktdf.read_from_fifo %scalar
          {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
          : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      %init = tensor.empty() : tensor<32xf32>
      // CHECK-NOT: memref.alloc
      // CHECK: %[[RESULT:.*]] = linalg.generic
      // CHECK-SAME: ins(%[[V]], %[[S]] : tensor<32xf32>, tensor<32xf32>)
      %r = linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel"]
      } ins(%v, %s : tensor<32xf32>, tensor<32xf32>)
          outs(%init : tensor<32xf32>) {
      ^bb0(%x: f32, %y: f32, %o: f32):
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      } -> tensor<32xf32>
      // CHECK: ktdf.write_to_fifo %[[RESULT]]
      ktdf.write_to_fifo %r, %output : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// Both slots share one hardware FIFO; their reads must not be reordered.
// CHECK-LABEL: func.func @two_splats(
func.func @two_splats(%first: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
                      %second: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
                      %output: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: %[[REG0:.*]] = memref.alloc() : memref<32xf32, "SFU_REG">
      // CHECK-NEXT: %[[R0:.*]] = ktdf.read_from_fifo %arg0
      // CHECK-NEXT: memref.copy %[[R0]], %[[REG0]]
      // CHECK-NEXT: %[[T0:.*]] = bufferization.to_tensor %[[REG0]] restrict
      // CHECK-SAME: {dataflow_scheduler.splat_register_view}
      // CHECK-NEXT: %[[REG1:.*]] = memref.alloc() : memref<32xf32, "SFU_REG">
      // CHECK-NEXT: %[[R1:.*]] = ktdf.read_from_fifo %arg1
      // CHECK-NEXT: memref.copy %[[R1]], %[[REG1]]
      // CHECK-NEXT: %[[T1:.*]] = bufferization.to_tensor %[[REG1]] restrict
      // CHECK-SAME: {dataflow_scheduler.splat_register_view}
      %a = ktdf.read_from_fifo %first
          {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
          : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      %b = ktdf.read_from_fifo %second
          {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
          : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      %init = tensor.empty() : tensor<32xf32>
      // CHECK-NOT: memref.alloc
      // CHECK: linalg.generic
      // CHECK-SAME: ins(%[[T0]], %[[T1]] : tensor<32xf32>, tensor<32xf32>)
      %r = linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<32xf32>, tensor<32xf32>)
          outs(%init : tensor<32xf32>) {
      ^bb0(%x: f32, %y: f32, %o: f32):
        %difference = arith.subf %x, %y : f32
        linalg.yield %difference : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %r, %output : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// Repeated uses share one materialized value, not additional FIFO receives.
// CHECK-LABEL: func.func @repeated_operand(
func.func @repeated_operand(%input: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
                            %output: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: %[[REG:.*]] = memref.alloc()
      // CHECK-NEXT: %[[R:.*]] = ktdf.read_from_fifo
      // CHECK-NEXT: memref.copy %[[R]], %[[REG]]
      // CHECK-NEXT: %[[T:.*]] = bufferization.to_tensor %[[REG]] restrict
      // CHECK-SAME: {dataflow_scheduler.splat_register_view}
      %s = ktdf.read_from_fifo %input
          {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
          : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      %init = tensor.empty() : tensor<32xf32>
      // CHECK-NOT: ktdf.read_from_fifo
      // CHECK-NOT: memref.alloc
      // CHECK: linalg.generic
      // CHECK-SAME: ins(%[[T]], %[[T]] : tensor<32xf32>, tensor<32xf32>)
      %r = linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel"]
      } ins(%s, %s : tensor<32xf32>, tensor<32xf32>)
          outs(%init : tensor<32xf32>) {
      ^bb0(%x: f32, %y: f32, %o: f32):
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      } -> tensor<32xf32>
      ktdf.write_to_fifo %r, %output : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// Pure broadcasts forward directly; no temporary is necessary.
// CHECK-LABEL: func.func @direct_write(
func.func @direct_write(%input: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
                        %output: !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK-NOT: memref.alloc
      // CHECK: %[[R:.*]] = ktdf.read_from_fifo
      // CHECK-SAME: -> tensor<32xf32>
      // CHECK-NEXT: ktdf.write_to_fifo %[[R]]
      %r = ktdf.read_from_fifo %input
          {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
          : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      ktdf.write_to_fifo %r, %output : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
    } {applicable_units = ["SFU"]}
  }
  return
}

// Existing device-pattern storage is already a legal shuffle destination.
// CHECK-LABEL: func.func @already_materialized(
func.func @already_materialized(%input: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
                                %register: memref<32xf32, "SFU_REG">) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK-NOT: memref.alloc
      // CHECK: %[[R:.*]] = ktdf.read_from_fifo
      // CHECK-NEXT: memref.copy %[[R]], %arg1
      %r = ktdf.read_from_fifo %input
          {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
          : <"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      memref.copy %r, %register : memref<32xf32> to memref<32xf32, "SFU_REG">
    } {applicable_units = ["SFU"]}
  }
  return
}

// A pre-existing buffer-semantics consumer needs the same materialization.
// CHECK-LABEL: func.func @buffer_consumer(
func.func @buffer_consumer(%input: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>,
                           %output: memref<32xf32, "SFU_REG">) {
  ktdf.pipeline {
    ktdf.stage depends_in(none) depends_out(none) {
      // CHECK: %[[REG:.*]] = memref.alloc()
      // CHECK-NEXT: %[[R:.*]] = ktdf.read_from_fifo
      // CHECK-NEXT: memref.copy %[[R]], %[[REG]]
      // CHECK-NOT: bufferization.to_tensor
      // CHECK: linalg.generic
      // CHECK-SAME: ins(%[[REG]], %[[REG]] : memref<32xf32, "SFU_REG">, memref<32xf32, "SFU_REG">)
      %r = ktdf.read_from_fifo %input
          {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
          : <"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      linalg.generic {
        indexing_maps = [#id, #id, #id],
        iterator_types = ["parallel"]
      } ins(%r, %r : memref<32xf32>, memref<32xf32>)
          outs(%output : memref<32xf32, "SFU_REG">) {
      ^bb0(%x: f32, %y: f32, %o: f32):
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      }
    } {applicable_units = ["SFU"]}
  }
  return
}
