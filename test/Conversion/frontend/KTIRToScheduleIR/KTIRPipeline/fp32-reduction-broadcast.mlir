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

// RUN: dataflow-scheduler-opt %s -ktir-to-ktdflow | FileCheck %s \
// RUN:   --implicit-check-not=ktdf.pipeline --implicit-check-not=ktdf.stage \
// RUN:   --implicit-check-not=ktdp.load --implicit-check-not=ktdp.store \
// RUN:   --implicit-check-not=ktdp_lowering.load \
// RUN:   --implicit-check-not=ktdp_lowering.store \
// RUN:   --implicit-check-not=ktdf.private \
// RUN:   --implicit-check-not=memref.alloc \
// RUN:   --implicit-check-not=dataflow.program_unit \
// RUN:   --implicit-check-not=ktdf.opaque

// Start from KTIR, with no scheduled FIFOs or stage dependencies. The two
// compute groups communicate through the same local-memory address: each row's
// sum occupies the first lane of a padded stick, which the add must broadcast.
// The single-iteration loops keep the two compute groups separate.
// KTDFLow retains the buffer-semantics reduction; device-specific reduction
// instructions and numerical execution are outside this test's scope.

// CHECK: #[[ID:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @sum_broadcast_add()
// CHECK-NEXT: call @local_schedule_0() : () -> ()
// CHECK-NEXT: call @local_schedule_1() : () -> ()
// CHECK-NEXT: return

// CHECK-LABEL: module @local_schedule_0 {
// CHECK: %[[R_ROWS:.*]] = arith.constant 2 : index
// CHECK: %[[R_ONE:.*]] = arith.constant 1 : index
// CHECK: %[[R_ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[R_STAT_ADDR:.*]] = arith.constant 1024 : index
// CHECK-DAG: %[[R_SOURCE_ADDR:.*]] = arith.constant 2048 : index
// CHECK-DAG: %[[R_SOURCE:.*]] = ktdp.construct_memory_view %[[R_SOURCE_ADDR]], sizes: [2, 32], strides: [32, 1]
// CHECK-DAG: %[[R_STATS:.*]] = ktdp.construct_memory_view %[[R_STAT_ADDR]], sizes: [2, 32], strides: [32, 1]
// CHECK-DAG: %[[R_SOURCE_MEM:.*]] = memref.memory_space_cast %[[R_SOURCE]] : memref<2x32xf32> to memref<2x32xf32, "L1">
// CHECK-DAG: %[[R_STATS_MEM:.*]] = memref.memory_space_cast %[[R_STATS]] : memref<2x32xf32> to memref<2x32xf32, "L1">
// CHECK: scf.for %[[R_ROW:.*]] = %[[R_ZERO]] to %[[R_ROWS]] step %[[R_ONE]]
// CHECK: %[[R_INPUT:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
// CHECK: %[[R_OUTPUT:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
// CHECK: ktdf_lowering.execute_on
// CHECK: ktdf.data_transfer from %[[R_SOURCE_MEM]][%[[R_ROW]], 0] size [1, 32] to %[[R_INPUT]] size [1, 32]
// CHECK: ktdf_lowering.execute_on
// CHECK: %[[ACC:.*]] = builtin.unrealized_conversion_cast %{{.*}} : index to memref<1x32xf32, "SFU_REG">
// CHECK: %[[VALUES:.*]] = ktdf.read_from_fifo %[[R_INPUT]]
// CHECK-NEXT: memref.copy %[[VALUES]], %[[ACC]]
// CHECK-NEXT: %[[HEAD:.*]] = memref.subview %[[ACC]][0, 0] [1, 1] [1, 1]
// CHECK-NEXT: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "reduction", "parallel"]
// CHECK-SAME: ins(%[[ACC]] : memref<1x32xf32, "SFU_REG">)
// CHECK-SAME: outs(%[[HEAD]] : memref<1x1xf32,
// CHECK-NEXT: ^bb0(%[[VALUE:.*]]: f32, %[[PARTIAL:.*]]: f32):
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[VALUE]], %[[PARTIAL]] : f32
// CHECK-NEXT: linalg.yield %[[SUM]] : f32
// CHECK: ktdf.write_to_fifo %[[ACC]], %[[R_OUTPUT]]
// CHECK: ktdf_lowering.execute_on
// CHECK: ktdf.data_transfer from %[[R_OUTPUT]] size [1, 32] to %[[R_STATS_MEM]][%[[R_ROW]], %[[R_ZERO]] * 32] size [1, 32]

// CHECK-LABEL: module @local_schedule_1 {
// CHECK: %[[C_ROWS:.*]] = arith.constant 2 : index
// CHECK: %[[C_ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C_DEST_ADDR:.*]] = arith.constant 3072 : index
// CHECK-DAG: %[[C_SOURCE_ADDR:.*]] = arith.constant 2048 : index
// CHECK-DAG: %[[C_ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C_STAT_ADDR:.*]] = arith.constant 1024 : index
// CHECK-DAG: %[[C_STATS:.*]] = ktdp.construct_memory_view %[[C_STAT_ADDR]], sizes: [2, 32], strides: [32, 1]
// CHECK-DAG: %[[C_SOURCE:.*]] = ktdp.construct_memory_view %[[C_SOURCE_ADDR]], sizes: [2, 32], strides: [32, 1]
// CHECK-DAG: %[[C_DEST:.*]] = ktdp.construct_memory_view %[[C_DEST_ADDR]], sizes: [2, 32], strides: [32, 1]
// CHECK-DAG: %[[C_SOURCE_MEM:.*]] = memref.memory_space_cast %[[C_SOURCE]] : memref<2x32xf32> to memref<2x32xf32, "L1">
// CHECK-DAG: %[[C_STATS_MEM:.*]] = memref.memory_space_cast %[[C_STATS]] : memref<2x32xf32> to memref<2x32xf32, "L1">
// CHECK: %[[C_DEST_MEM:.*]] = memref.memory_space_cast %[[C_DEST]] : memref<2x32xf32> to memref<2x32xf32, "L1">
// CHECK: scf.for %[[C_ROW:.*]] = %[[C_ZERO]] to %[[C_ROWS]] step %[[C_ONE]]
// CHECK: %[[C_INPUT:.*]]:2 = ktdf.fifo.allocate()
// CHECK-SAME: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
// CHECK: %[[C_OUTPUT:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
// CHECK: ktdf_lowering.execute_on
// CHECK: ktdf.data_transfer from %[[C_STATS_MEM]][%[[C_ROW]], 0] size [1, 4] to %[[C_INPUT]]#1 size [32, 1]
// CHECK-SAME: transfer_mode = #ktdf.transfer_mode<splat>
// CHECK-NEXT: ktdf.data_transfer from %[[C_SOURCE_MEM]][%[[C_ROW]], %[[C_ZERO]] * 32] size [1, 32] to %[[C_INPUT]]#0 size [1, 32]
// CHECK: ktdf_lowering.execute_on
// CHECK: %[[SPLAT:.*]] = ktdf.read_from_fifo %[[C_INPUT]]#1
// CHECK-SAME: {splat = #ktdf.splat<first_subsimd_lane_to_each_subsimd>}
// CHECK-SAME: -> memref<1x32xf32>
// CHECK-NEXT: %[[VECTOR:.*]] = ktdf.read_from_fifo %[[C_INPUT]]#0
// CHECK: %[[REGISTER:.*]] = builtin.unrealized_conversion_cast %{{.*}} : index to memref<1x32xf32, "SFU_REG">
// CHECK-NEXT: memref.copy %[[SPLAT]], %[[REGISTER]]
// CHECK-NEXT: %[[SCALAR:.*]] = bufferization.to_tensor %[[REGISTER]] restrict
// CHECK-SAME: {dataflow_scheduler.splat_register_view}
// CHECK-NEXT: %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[ID]], #[[ID]], #[[ID]]]
// CHECK-SAME: ins(%[[SCALAR]], %[[VECTOR]] : tensor<1x32xf32>, tensor<1x32xf32>)
// CHECK-NEXT: ^bb0(%[[SCALAR_ELEMENT:.*]]: f32, %[[VECTOR_ELEMENT:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: %[[ADDED:.*]] = arith.addf %[[SCALAR_ELEMENT]], %[[VECTOR_ELEMENT]] : f32
// CHECK-NEXT: linalg.yield %[[ADDED]] : f32
// CHECK: ktdf.write_to_fifo %[[RESULT]], %[[C_OUTPUT]]
// CHECK: ktdf_lowering.execute_on
// CHECK: ktdf.data_transfer from %[[C_OUTPUT]] size [1, 32] to %[[C_DEST_MEM]][%[[C_ROW]], %[[C_ZERO]] * 32] size [1, 32]

#reduce_in = affine_map<(row, lane, out_lane) -> (row, lane)>
#reduce_out = affine_map<(row, lane, out_lane) -> (row, out_lane)>
#id = affine_map<(row, lane) -> (row, lane)>
#broadcast = affine_map<(row, lane) -> (row, 0)>
#stick = affine_set<(row, lane) : (row >= 0, -row + 1 >= 0, lane >= 0, -lane + 31 >= 0)>
#scalar = affine_set<(row, lane) : (row >= 0, -row + 1 >= 0, lane == 0)>
module {
  ktdf_arch.device @sample_device attributes {
    mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<ct_local> = "L1">
  } import("../../../../Transforms/Inputs/fp32-splat-device.mlir")
  func.func @sum_broadcast_add() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %statistics_address = arith.constant 1024 : index
    %source_address = arith.constant 2048 : index
    %destination_address = arith.constant 3072 : index
    %statistics = ktdp.construct_memory_view %statistics_address,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #stick, memory_space = #ktdp.memory_space<ct_local>}
        : memref<2x32xf32>
    %source = ktdp.construct_memory_view %source_address,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #stick, memory_space = #ktdp.memory_space<ct_local>}
        : memref<2x32xf32>
    %destination = ktdp.construct_memory_view %destination_address,
        sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #stick, memory_space = #ktdp.memory_space<ct_local>}
        : memref<2x32xf32>
    %scalar_tile = ktdp.construct_access_tile %statistics[%c0, %c0]
        {access_tile_set = #scalar, access_tile_order = #id}
        : memref<2x32xf32> -> !ktdp.access_tile<2x1xindex>
    %source_tile = ktdp.construct_access_tile %source[%c0, %c0]
        {access_tile_set = #stick, access_tile_order = #id}
        : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
    %destination_tile = ktdp.construct_access_tile %destination[%c0, %c0]
        {access_tile_set = #stick, access_tile_order = #id}
        : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
    %reduction_tile = ktdp.construct_access_tile %statistics[%c0, %c0]
        {access_tile_set = #stick, access_tile_order = #id}
        : memref<2x32xf32> -> !ktdp.access_tile<2x32xindex>
    scf.for %reduction = %c0 to %c1 step %c1 {
      %input = ktdp.load %source_tile : <2x32xindex> -> tensor<2x32xf32>
      %empty = tensor.empty() : tensor<2x32xf32>
      %reduced = linalg.generic {
        indexing_maps = [#reduce_in, #reduce_out],
        iterator_types = ["parallel", "reduction", "parallel"]
      } ins(%input : tensor<2x32xf32>) outs(%empty : tensor<2x32xf32>) {
      ^bb0(%value: f32, %acc: f32):
        %sum = arith.addf %value, %acc : f32
        linalg.yield %sum : f32
      } -> tensor<2x32xf32>
      ktdp.store %reduced, %reduction_tile : tensor<2x32xf32>, <2x32xindex>
    }
    scf.for %consumer = %c0 to %c1 step %c1 {
      %scalar_value = ktdp.load %scalar_tile : <2x1xindex> -> tensor<2x1xf32>
      %vector_value = ktdp.load %source_tile : <2x32xindex> -> tensor<2x32xf32>
      %empty = tensor.empty() : tensor<2x32xf32>
      %result = linalg.generic {
        indexing_maps = [#broadcast, #id, #id],
        iterator_types = ["parallel", "parallel"]
      } ins(%scalar_value, %vector_value : tensor<2x1xf32>, tensor<2x32xf32>)
          outs(%empty : tensor<2x32xf32>) {
      ^bb0(%scalar: f32, %vector: f32, %out: f32):
        %sum = arith.addf %scalar, %vector : f32
        linalg.yield %sum : f32
      } -> tensor<2x32xf32>
      ktdp.store %result, %destination_tile : tensor<2x32xf32>, <2x32xindex>
    }
    return
  }
}
