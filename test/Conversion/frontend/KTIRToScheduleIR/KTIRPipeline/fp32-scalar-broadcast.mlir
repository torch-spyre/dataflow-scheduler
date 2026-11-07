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

// RUN: dataflow-scheduler-opt %s -split-input-file -ktir-to-dfir | \
// RUN:   FileCheck %s --implicit-check-not=ktdf.read_from_fifo \
// RUN:   --implicit-check-not=ktdf.write_to_fifo \
// RUN:   --implicit-check-not=ktdf.data_transfer \
// RUN:   --implicit-check-not=bufferization.to_tensor \
// RUN:   --implicit-check-not=memref.copy
// RUN: dataflow-scheduler-opt %s -split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(ktir-map-and-tile,ktir-bufferize,ktir-pipeline),path-expansion)' \
// RUN:   > %t.once
// RUN: dataflow-scheduler-opt %s -split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(ktir-map-and-tile,ktir-bufferize,ktir-pipeline),path-expansion,path-expansion)' \
// RUN:   > %t.twice
// RUN: diff %t.once %t.twice

// Begin with a genuine broadcast, so a compute stage is selected before the
// identity generic disappears. The backing allocation spans a complete stick;
// the logical scalar access alone does not authorize a wider physical read.
// CHECK-LABEL: module @local_schedule_0 {
// CHECK: func.func @local_schedule_0_body()
// CHECK: dataflow.program_unit
// CHECK: %[[LOAD:.*]] = agen.vector_load
// CHECK-SAME: vector<4xf32>
// CHECK-NEXT: %[[PACKET:.*]] = vectorchain.shuffle input(%[[LOAD]])
// CHECK-SAME: indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32], repetition = 8 : i32
// CHECK: dataflow.send %{{.*}}, %[[PACKET]] : vector<32xf32>
// CHECK: dataflow.program_unit
// CHECK: %[[RECV:.*]] = dataflow.receive %{{.*}} : vector<32xf32>
// CHECK-NEXT: %[[SPLAT:.*]] = vectorchain.shuffle input(%[[RECV]])
// CHECK-SAME: indices = [0 : i32, 0 : i32, 0 : i32, 0 : i32], repetition = 8 : i32
// CHECK-NOT: agen.vector_store
// CHECK: dataflow.send %{{.*}}, %[[SPLAT]] : vector<32xf32>
// CHECK: dataflow.program_unit
// CHECK: %[[OUTPUT:.*]] = dataflow.receive %{{.*}} : vector<32xf32>
// CHECK-NEXT: agen.vector_store %[[OUTPUT]],
// CHECK-SAME: vector<32xf32>

#id = affine_map<(i) -> (i)>
#broadcast = affine_map<(i) -> (0)>
#stick = affine_set<(i) : (i >= 0, -i + 31 >= 0)>
#scalar = affine_set<(i) : (i == 0)>
module {
  ktdf_arch.device @sample_device attributes {
    mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<ct_local> = "L1">
  } import("../../../../Transforms/Inputs/fp32-splat-device.mlir")
  func.func @broadcast_only() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %source_address = arith.constant 1024 : index
    %destination_address = arith.constant 2048 : index
    %source = ktdp.construct_memory_view %source_address,
        sizes: [32], strides: [1]
        {coordinate_set = #stick, memory_space = #ktdp.memory_space<ct_local>}
        : memref<32xf32>
    %destination = ktdp.construct_memory_view %destination_address,
        sizes: [32], strides: [1]
        {coordinate_set = #stick, memory_space = #ktdp.memory_space<ct_local>}
        : memref<32xf32>
    %source_tile = ktdp.construct_access_tile %source[%c0]
        {access_tile_set = #scalar, access_tile_order = #id}
        : memref<32xf32> -> !ktdp.access_tile<1xindex>
    %destination_tile = ktdp.construct_access_tile %destination[%c0]
        {access_tile_set = #stick, access_tile_order = #id}
        : memref<32xf32> -> !ktdp.access_tile<32xindex>
    %value = ktdp.load %source_tile : <1xindex> -> tensor<1xf32>
    %empty = tensor.empty() : tensor<32xf32>
    %result = linalg.generic {
      indexing_maps = [#broadcast, #id], iterator_types = ["parallel"]
    } ins(%value : tensor<1xf32>) outs(%empty : tensor<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<32xf32>
    ktdp.store %result, %destination_tile : tensor<32xf32>, <32xindex>
    return
  }
}

// -----

// The arithmetic consumer needs a local register between the receive-side
// shuffle and the binary op. Its output still travels directly through a FIFO.
// CHECK-LABEL: module @local_schedule_0 {
// CHECK: func.func @local_schedule_0_body()
// CHECK: %[[REGISTER_UNIT:.*]] = dataflow.get_unit {name = "sfu_reg", type = "sfu_reg"}
// CHECK: dataflow.program_unit
// CHECK: %[[RECV:.*]] = dataflow.receive %{{.*}} : vector<32xf32>
// CHECK-NEXT: %[[SPLAT:.*]] = vectorchain.shuffle input(%[[RECV]])
// CHECK-SAME: indices = [0 : i32, 0 : i32, 0 : i32, 0 : i32], repetition = 8 : i32
// CHECK: %[[VECTOR:.*]] = dataflow.receive %{{.*}} : vector<32xf32>
// CHECK: %[[REGISTER:.*]] = dataflow.get_logical_memory_view %[[REGISTER_UNIT]],
// CHECK-NEXT: agen.vector_store %[[SPLAT]], %[[REGISTER]]
// CHECK: %[[SCALAR:.*]] = agen.vector_load %[[REGISTER]]
// CHECK: %[[SUM:.*]] = vectorchain.binary %[[SCALAR]], %[[VECTOR]]
// CHECK: dataflow.send %{{.*}}, %[[SUM]] : vector<32xf32>
// CHECK: agen.vector_store
// CHECK-SAME: vector<32xf32>

#id = affine_map<(i) -> (i)>
#broadcast = affine_map<(i) -> (0)>
#stick = affine_set<(i) : (i >= 0, -i + 31 >= 0)>
#scalar = affine_set<(i) : (i == 0)>
module {
  ktdf_arch.device @sample_device attributes {
    mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<ct_local> = "L1">
  } import("../../../../Transforms/Inputs/fp32-splat-device.mlir")
  func.func @broadcast_add() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %scalar_address = arith.constant 1024 : index
    %vector_address = arith.constant 2048 : index
    %destination_address = arith.constant 3072 : index
    %scalar_source = ktdp.construct_memory_view %scalar_address,
        sizes: [32], strides: [1]
        {coordinate_set = #stick, memory_space = #ktdp.memory_space<ct_local>}
        : memref<32xf32>
    %vector_source = ktdp.construct_memory_view %vector_address,
        sizes: [32], strides: [1]
        {coordinate_set = #stick, memory_space = #ktdp.memory_space<ct_local>}
        : memref<32xf32>
    %destination = ktdp.construct_memory_view %destination_address,
        sizes: [32], strides: [1]
        {coordinate_set = #stick, memory_space = #ktdp.memory_space<ct_local>}
        : memref<32xf32>
    %scalar_tile = ktdp.construct_access_tile %scalar_source[%c0]
        {access_tile_set = #scalar, access_tile_order = #id}
        : memref<32xf32> -> !ktdp.access_tile<1xindex>
    %vector_tile = ktdp.construct_access_tile %vector_source[%c0]
        {access_tile_set = #stick, access_tile_order = #id}
        : memref<32xf32> -> !ktdp.access_tile<32xindex>
    %destination_tile = ktdp.construct_access_tile %destination[%c0]
        {access_tile_set = #stick, access_tile_order = #id}
        : memref<32xf32> -> !ktdp.access_tile<32xindex>
    %scalar_value = ktdp.load %scalar_tile : <1xindex> -> tensor<1xf32>
    %vector_value = ktdp.load %vector_tile : <32xindex> -> tensor<32xf32>
    %empty = tensor.empty() : tensor<32xf32>
    %result = linalg.generic {
      indexing_maps = [#broadcast, #id, #id], iterator_types = ["parallel"]
    } ins(%scalar_value, %vector_value : tensor<1xf32>, tensor<32xf32>)
        outs(%empty : tensor<32xf32>) {
    ^bb0(%scalar: f32, %vector: f32, %out: f32):
      %sum = arith.addf %scalar, %vector : f32
      linalg.yield %sum : f32
    } -> tensor<32xf32>
    ktdp.store %result, %destination_tile : tensor<32xf32>, <32xindex>
    return
  }
}
