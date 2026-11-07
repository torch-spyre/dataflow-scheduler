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

// RUN: dataflow-scheduler-opt %s \
// RUN:   -pass-pipeline='builtin.module(access-granularity-legalization,scalar-broadcast-legalization,splat-legalization,map-reduction-partials,custom-linalg-bufferization)' \
// RUN:   | FileCheck %s

// Each row produces its own sum in lane 0 of a padded FP32 stick. The next
// stage must reload that lane, not the neighbouring reduction partials, and
// broadcast it before adding the original row. Check the shared memory, row
// index, and store-to-load dependency as well as the consumer's splat.
// This stops before device-specific SIMD reduction lowering; it does not
// validate a reduction instruction recipe or its numerical execution.

// CHECK: #[[ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @sum_broadcast_add(
// CHECK-SAME: %[[SOURCE:[^:]+]]: memref<2x32xf32, "L1">,
// CHECK-SAME: %[[STATISTICS:[^:]+]]: memref<2x32xf32, "L1">,
// CHECK-SAME: %[[DESTINATION:[^:]+]]: memref<2x32xf32, "L1">
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[ROWS:.*]] = arith.constant 2 : index
// CHECK: scf.for %[[ROW:.*]] = %[[ZERO]] to %[[ROWS]] step %[[ONE]]
// CHECK: %[[STATE:.*]]:10 = ktdf.private
// CHECK: ktdf.data_transfer from %[[SOURCE]][%[[ROW]], 0] size [1, 32] to %[[STATE]]#0 size [32]
// CHECK: ktdf.stage depends_in(%[[STATE]]#5) depends_out(%[[STATE]]#6)
// CHECK: %[[ACC:.*]] = memref.alloc() : memref<1x32xf32, "SFU_REG">
// CHECK: %[[INPUT:.*]] = ktdf.read_from_fifo %[[STATE]]#0
// CHECK-SAME: -> memref<1x32xf32>
// CHECK-NEXT: memref.copy %[[INPUT]], %[[ACC]]
// CHECK-NEXT: %[[HEAD:.*]] = memref.subview %[[ACC]][0, 0] [1, 1] [1, 1]
// CHECK-NEXT: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[ACC]] : memref<1x32xf32, "SFU_REG">)
// CHECK-SAME: outs(%[[HEAD]] : memref<1x1xf32,
// CHECK-NEXT: ^bb0(%[[VALUE:.*]]: f32, %[[PARTIAL:.*]]: f32):
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[VALUE]], %[[PARTIAL]] : f32
// CHECK-NEXT: linalg.yield %[[SUM]] : f32
// CHECK: ktdf.write_to_fifo %[[ACC]], %[[STATE]]#1
// CHECK-SAME: <"SFU" -> "L1SU", 32xf32>
// CHECK: ktdf.stage depends_in(%[[STATE]]#6) depends_out(%[[STATE]]#7)
// CHECK: ktdf.data_transfer from %[[STATE]]#1 size [1, 32] to %[[STATISTICS]][%[[ROW]], 0] size [1, 32]
// CHECK: ktdf.stage depends_in(%[[STATE]]#7) depends_out(%[[STATE]]#8)
// CHECK: ktdf.data_transfer from %[[STATISTICS]][%[[ROW]], 0] size [1, 4] to %[[STATE]]#2 size [32]
// CHECK-SAME: transfer_mode = #ktdf.transfer_mode<splat>
// CHECK-NEXT: ktdf.data_transfer from %[[SOURCE]][%[[ROW]], 0] size [1, 32] to %[[STATE]]#3 size [32]
// CHECK: ktdf.stage depends_in(%[[STATE]]#8) depends_out(%[[STATE]]#9)
// CHECK: %[[REGISTER:.*]] = memref.alloc() : memref<1x32xf32, "SFU_REG">
// CHECK-NEXT: %[[SPLAT:.*]] = ktdf.read_from_fifo %[[STATE]]#2
// CHECK-SAME: {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
// CHECK-SAME: -> memref<1x32xf32>
// CHECK-NEXT: memref.copy %[[SPLAT]], %[[REGISTER]]
// CHECK-NEXT: %[[SCALAR:.*]] = bufferization.to_tensor %[[REGISTER]] restrict
// CHECK-SAME: {dataflow_scheduler.splat_register_view}
// CHECK: %[[VECTOR:.*]] = ktdf.read_from_fifo %[[STATE]]#3
// CHECK: %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[ID]], #[[ID]], #[[ID]]]
// CHECK-SAME: ins(%[[SCALAR]], %[[VECTOR]] : tensor<1x32xf32>, tensor<1x32xf32>)
// CHECK-NEXT: ^bb0(%[[SCALAR_ELEMENT:.*]]: f32, %[[VECTOR_ELEMENT:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[ADDED:.*]] = arith.addf %[[SCALAR_ELEMENT]], %[[VECTOR_ELEMENT]] : f32
// CHECK-NEXT: linalg.yield %[[ADDED]] : f32
// CHECK: ktdf.write_to_fifo %[[RESULT]], %[[STATE]]#4
// CHECK: ktdf.stage depends_in(%[[STATE]]#9) depends_out(none)
// CHECK: ktdf.data_transfer from %[[STATE]]#4 size [32] to %[[DESTINATION]][%[[ROW]], 0] size [1, 32]

!load_fifo = !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
!store_fifo = !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
#id = affine_map<(row, lane) -> (row, lane)>
#broadcast = affine_map<(row, lane) -> (row, 0)>
#reduce_in = affine_map<(row, lane, out_lane) -> (row, lane)>
#reduce_out = affine_map<(row, lane, out_lane) -> (row, out_lane)>

ktdf_arch.device @sample_device import("../Inputs/fp32-splat-device.mlir")

func.func @sum_broadcast_add(
    %source: memref<2x32xf32, "L1">,
    %statistics: memref<2x32xf32, "L1">,
    %destination: memref<2x32xf32, "L1">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %row = %c0 to %c2 step %c1 {
    ktdf.pipeline {
      %state:10 = ktdf.private -> (
          !load_fifo, !store_fifo, !load_fifo, !load_fifo, !store_fifo,
          !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
        %input = ktdf.fifo.allocate() -> !load_fifo
        %reduced = ktdf.fifo.allocate() -> !store_fifo
        %scalar = ktdf.fifo.allocate() -> !load_fifo
        %vector = ktdf.fifo.allocate() -> !load_fifo
        %output = ktdf.fifo.allocate() -> !store_fifo
        %loaded = ktdf.create_token : !ktdf.token
        %computed = ktdf.create_token : !ktdf.token
        %stored = ktdf.create_token : !ktdf.token
        %reloaded = ktdf.create_token : !ktdf.token
        %added = ktdf.create_token : !ktdf.token
        ktdf.private_yield %input, %reduced, %scalar, %vector, %output,
            %loaded, %computed, %stored, %reloaded, %added
            : !load_fifo, !store_fifo, !load_fifo, !load_fifo, !store_fifo,
              !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%state#5) {
        ktdf.data_transfer from %source[%row, 0] size [1, 32]
            to %state#0 size [32] : memref<2x32xf32, "L1">, !load_fifo
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%state#5) depends_out(%state#6) {
        %values = ktdf.read_from_fifo %state#0
            : !load_fifo -> tensor<1x32xf32>
        %empty = tensor.empty() : tensor<1x32xf32>
        %reduced = linalg.generic {
          indexing_maps = [#reduce_in, #reduce_out],
          iterator_types = ["parallel", "reduction", "parallel"]
        } ins(%values : tensor<1x32xf32>) outs(%empty : tensor<1x32xf32>) {
        ^bb0(%value: f32, %acc: f32):
          %sum = arith.addf %value, %acc : f32
          linalg.yield %sum : f32
        } -> tensor<1x32xf32>
        ktdf.write_to_fifo %reduced, %state#1
            : tensor<1x32xf32>, !store_fifo
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%state#6) depends_out(%state#7) {
        ktdf.data_transfer from %state#1 size [32]
            to %statistics[%row, 0] size [1, 32]
            : !store_fifo, memref<2x32xf32, "L1">
      } {applicable_units = ["L1SU"]}
      ktdf.stage depends_in(%state#7) depends_out(%state#8) {
        ktdf.data_transfer from %statistics[%row, 0] size [1, 1]
            to %state#2 size [32] {transfer_mode = #ktdf.transfer_mode<splat>}
            : memref<2x32xf32, "L1">, !load_fifo
        ktdf.data_transfer from %source[%row, 0] size [1, 32]
            to %state#3 size [32] : memref<2x32xf32, "L1">, !load_fifo
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%state#8) depends_out(%state#9) {
        %scalar_value = ktdf.read_from_fifo %state#2
            : !load_fifo -> tensor<1x32xf32>
        %vector_value = ktdf.read_from_fifo %state#3
            : !load_fifo -> tensor<1x32xf32>
        %empty = tensor.empty() : tensor<1x32xf32>
        %result = linalg.generic {
          indexing_maps = [#broadcast, #id, #id],
          iterator_types = ["parallel", "parallel"]
        } ins(%scalar_value, %vector_value : tensor<1x32xf32>, tensor<1x32xf32>)
            outs(%empty : tensor<1x32xf32>) {
        ^bb0(%scalar: f32, %vector: f32, %out: f32):
          %sum = arith.addf %scalar, %vector : f32
          linalg.yield %sum : f32
        } -> tensor<1x32xf32>
        ktdf.write_to_fifo %result, %state#4
            : tensor<1x32xf32>, !store_fifo
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%state#9) depends_out(none) {
        ktdf.data_transfer from %state#4 size [32]
            to %destination[%row, 0] size [1, 32]
            : !store_fifo, memref<2x32xf32, "L1">
      } {applicable_units = ["L1SU"]}
    }
  }
  return
}
