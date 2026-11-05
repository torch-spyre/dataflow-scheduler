// RUN: dataflow-scheduler-opt -ktir-to-dfir --split-input-file %s | FileCheck %s

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

// Keep the two casts in independent lane-wise generics, connected directly by
// their tensor SSA value. The full pipeline must preserve both conversions and
// carry the intermediate f32 values as two physical vectors per 64-element tile.
// No KTIR store/load or host permutation separates the widening and narrowing.

// CHECK-DAG: #[[GLOBAL_LAYOUT:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// CHECK-DAG: #[[MID_LAYOUT:.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[HALF:.+]] = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>
// CHECK-LABEL: func.func @dl16_fp32_dl16_64
// CHECK: func.func @local_schedule_0_body() attributes
// CHECK-DAG: %[[LOAD_UNIT:.+]] = dataflow.get_unit {{.*}}type = "l1lu"{{.*}} : index
// CHECK-DAG: %[[STORE_UNIT:.+]] = dataflow.get_unit {{.*}}type = "l1su"{{.*}} : index
// CHECK-DAG: %[[MEMORY_UNIT:.+]] = dataflow.get_unit {{.*}}type = "l1"{{.*}} : index
// The load unit first supplies the f16 input, then reloads the two f32 halves
// from one scratch buffer in the order in which widening stored them.
// CHECK: dataflow.program_unit
// CHECK: %[[GLOBAL_INPUT:.+]] = dataflow.get_logical_memory_view %{{.+}} {layout_map = #[[GLOBAL_LAYOUT]]} : index, index, memref<1x64xf16>
// CHECK: agen.composite_load_and_store src:%[[GLOBAL_INPUT]]
// CHECK-NEXT: time_symbols(), load_iv(%{{.+}}:vector<64xf16>)
// CHECK: dataflow.program_unit iter_arg : %[[LOAD_KEY:.+]] -> (%[[LOAD_UNIT]]) : {
// CHECK: %[[LOAD_MEMORY_MAP:.+]] = uniform.def_immutable_mapping([%[[LOAD_UNIT]] -> %[[MEMORY_UNIT]]]):index
// CHECK-NEXT: %[[LOAD_MEMORY:.+]] = uniform.query_map(map:%[[LOAD_MEMORY_MAP]], key:%[[LOAD_KEY]]) : index
// CHECK: %[[LOAD_BUFFER:.+]] = dataflow.get_logical_memory_view %[[LOAD_MEMORY]], %[[MID_BASE:[^ ]+]] {layout_map = #[[MID_LAYOUT]]} : index, index, memref<64xf32>
// CHECK: %[[INPUT:.+]] = agen.vector_load %{{.+}}[0, 0] {{.+}} : memref<1x64xf16>, vector<64xf16>
// CHECK-NEXT: dataflow.send %{{.+}}, %[[INPUT]] : vector<64xf16>
// Wait for the scratch writer, not the HBM load unit, before reading either half.
// CHECK-NEXT: %[[WAIT_MAP:.+]] = uniform.def_immutable_mapping([%[[LOAD_UNIT]] -> %[[STORE_UNIT]]]):index
// CHECK-NEXT: %[[WAIT_PEER:.+]] = uniform.query_map(map:%[[WAIT_MAP]], key:%[[LOAD_KEY]]) : index
// CHECK-NEXT: dataflow.sync_send %[[WAIT_PEER]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT: dataflow.sync_recv %[[WAIT_PEER]] : index
// CHECK: %[[L0:.+]] = agen.vector_load %[[LOAD_BUFFER]][0] {load_order = #{{.+}}, load_set = #[[HALF]]} : memref<64xf32>, vector<32xf32>
// CHECK-NEXT: dataflow.send %[[LOAD_DEST:.+]], %[[L0]] : vector<32xf32>
// CHECK-NEXT: %[[L1:.+]] = agen.vector_load %[[LOAD_BUFFER]][32] {load_order = #{{.+}}, load_set = #[[HALF]]} : memref<64xf32>, vector<32xf32>
// CHECK-NEXT: dataflow.send %[[LOAD_DEST]], %[[L1]] : vector<32xf32>
// CHECK-NOT: agen.vector_load
// CHECK: dataflow.program_unit
// Both casts must survive: two widening casts/shuffles feed the scratch store,
// and two narrowing casts consume the reloaded halves before packing f16.
// CHECK: %[[IN:.+]] = dataflow.receive %{{.+}} : vector<64xf16>
// CHECK: %[[W0:.+]] = vectorchain.cast %[[IN]] : vector<64xf16>, vector<64xf32>
// CHECK-NEXT: %[[LO:.+]] = vectorchain.shuffle input(%[[W0]]) {indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32], repetition = 8 : i32} : vector<64xf32>, vector<32xf32>
// CHECK-NEXT: dataflow.send %[[WIDE_DEST:.+]], %[[LO]] : vector<32xf32>
// CHECK-NEXT: %[[W1:.+]] = vectorchain.cast %[[IN]] : vector<64xf16>, vector<64xf32>
// CHECK-NEXT: %[[HI:.+]] = vectorchain.shuffle input(%[[W1]]) {indices = [4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf32>, vector<32xf32>
// CHECK-NEXT: dataflow.send %[[WIDE_DEST]], %[[HI]] : vector<32xf32>
// CHECK: %[[R0:.+]] = dataflow.receive %[[NARROW_SRC:.+]] : vector<32xf32>
// CHECK-NEXT: %[[R1:.+]] = dataflow.receive %[[NARROW_SRC]] : vector<32xf32>
// CHECK-NEXT: %[[N0:.+]] = vectorchain.cast %[[R0]] : vector<32xf32>, vector<32xf16>
// CHECK-NEXT: %[[N1:.+]] = vectorchain.cast %[[R1]] : vector<32xf32>, vector<32xf16>
// CHECK-NEXT: %[[PACKED:.+]] = vectorchain.pack %[[N0]], %[[N1]] {indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : index} : vector<32xf16>, vector<32xf16>, vector<64xf16>
// CHECK: dataflow.send %{{.+}}, %[[PACKED]] : vector<64xf16>
// CHECK-NOT: vectorchain.cast
// CHECK: dataflow.program_unit iter_arg : %[[STORE_KEY:.+]] -> (%[[STORE_UNIT]]) : {
// The store unit writes both halves to the very same scratch allocation and
// offsets that the load unit reads, then stores the final f16 vector whole.
// CHECK: %[[STORE_MEMORY_MAP:.+]] = uniform.def_immutable_mapping([%[[STORE_UNIT]] -> %[[MEMORY_UNIT]]]):index
// CHECK-NEXT: %[[STORE_MEMORY:.+]] = uniform.query_map(map:%[[STORE_MEMORY_MAP]], key:%[[STORE_KEY]]) : index
// CHECK: %[[STORE_BUFFER:.+]] = dataflow.get_logical_memory_view %[[STORE_MEMORY]], %[[MID_BASE]] {layout_map = #[[MID_LAYOUT]]} : index, index, memref<64xf32>
// CHECK: %[[S0:.+]] = dataflow.receive %[[STORE_SRC:.+]] : vector<32xf32>
// CHECK-NEXT: agen.vector_store %[[S0]], %[[STORE_BUFFER]][0] {store_order = #{{.+}}, store_set = #[[HALF]]} : memref<64xf32>, vector<32xf32>
// CHECK-NEXT: %[[S1:.+]] = dataflow.receive %[[STORE_SRC]] : vector<32xf32>
// CHECK-NEXT: agen.vector_store %[[S1]], %[[STORE_BUFFER]][32] {store_order = #{{.+}}, store_set = #[[HALF]]} : memref<64xf32>, vector<32xf32>
// Notify the scratch reader only after both halves have been stored.
// CHECK-NEXT: %[[READY_MAP:.+]] = uniform.def_immutable_mapping([%[[STORE_UNIT]] -> %[[LOAD_UNIT]]]):index
// CHECK-NEXT: %[[READY_PEER:.+]] = uniform.query_map(map:%[[READY_MAP]], key:%[[STORE_KEY]]) : index
// CHECK-NEXT: dataflow.sync_send %[[READY_PEER]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT: dataflow.sync_recv %[[READY_PEER]] : index
// CHECK: %[[OUTPUT:.+]] = dataflow.receive %{{.+}} : vector<64xf16>
// CHECK-NEXT: agen.vector_store %[[OUTPUT]], %{{.+}}[0, 0] {{.+}} : memref<1x64xf16>, vector<64xf16>
// CHECK-NOT: dataflow.receive
// CHECK: dataflow.program_unit
// CHECK: %[[GLOBAL_OUTPUT:.+]] = dataflow.get_logical_memory_view %{{.+}} {layout_map = #[[GLOBAL_LAYOUT]]} : index, index, memref<1x64xf16>
// CHECK: agen.composite_load_and_store src:%{{.+}} dst:%[[GLOBAL_OUTPUT]]
// CHECK-NEXT: time_symbols(), load_iv(%{{.+}}:vector<64xf16>)
// CHECK: return

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @dl16_fp32_dl16_64(%input: index, %output: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %input_view = ktdp.construct_memory_view %input, sizes: [1, 64], strides: [64, 1] {
      coordinate_set = #set, memory_space = #ktdp.memory_space<global>
    } : memref<1x64xf16>
    %input_tile = ktdp.construct_access_tile %input_view[%c0, %c0] {
      access_tile_order = #map, access_tile_set = #set
    } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
    %values = ktdp.load %input_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
    %wide_empty = tensor.empty() : tensor<1x64xf32>
    %wide = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%values : tensor<1x64xf16>) outs(%wide_empty : tensor<1x64xf32>) {
    ^bb0(%in: f16, %out: f32):
      %cast = arith.extf %in : f16 to f32
      linalg.yield %cast : f32
    } -> tensor<1x64xf32>
    %narrow_empty = tensor.empty() : tensor<1x64xf16>
    %result = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%wide : tensor<1x64xf32>) outs(%narrow_empty : tensor<1x64xf16>) {
    ^bb0(%in: f32, %out: f16):
      %cast = arith.truncf %in : f32 to f16
      linalg.yield %cast : f16
    } -> tensor<1x64xf16>
    %output_view = ktdp.construct_memory_view %output, sizes: [1, 64], strides: [64, 1] {
      coordinate_set = #set, memory_space = #ktdp.memory_space<global>
    } : memref<1x64xf16>
    %output_tile = ktdp.construct_access_tile %output_view[%c0, %c0] {
      access_tile_order = #map, access_tile_set = #set
    } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
    ktdp.store %result, %output_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    return
  }
}

// -----

// Four rows of 128 elements require eight 64-element tiles. Both row and column
// tile offsets must be preserved when the intermediate uses two f32 transfers.
// The two outer loops enumerate four rows; the inner loop visits columns 0 and
// 64. The same 64-element scratch is reused at offsets [0, 0] and [0, 32].
// CHECK-DAG: #[[GLOBAL_LAYOUT:.+]] = affine_map<(d0, d1) -> (d0 * 128 + d1)>
// CHECK-DAG: #[[MID_LAYOUT:.+]] = affine_map<(d0, d1) -> (d0 * 64 + d1)>
// CHECK-DAG: #[[HALF:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 31 >= 0)>
// CHECK-LABEL: func.func @dl16_fp32_dl16_tiles
// CHECK: func.func @local_schedule_0_body() attributes
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[LOAD_UNIT:.+]] = dataflow.get_unit {{.*}}type = "l1lu"{{.*}} : index
// CHECK-DAG: %[[STORE_UNIT:.+]] = dataflow.get_unit {{.*}}type = "l1su"{{.*}} : index
// CHECK-DAG: %[[MEMORY_UNIT:.+]] = dataflow.get_unit {{.*}}type = "l1"{{.*}} : index
// CHECK: dataflow.program_unit
// CHECK: %[[GLOBAL_INPUT:.+]] = dataflow.get_logical_memory_view %{{.+}} {layout_map = #[[GLOBAL_LAYOUT]]} : index, index, memref<4x128xf16>
// CHECK: scf.for %[[LOAD_ROW_HI:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT: scf.for %[[LOAD_ROW_LO:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT: scf.for %[[LOAD_COL:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: agen.composite_load_and_store src:%[[GLOBAL_INPUT]][%[[LOAD_ROW_HI]] * 2 + %[[LOAD_ROW_LO]], %[[LOAD_COL]] * 64] dst:%{{.+}}[0, 0]
// CHECK-NEXT: time_symbols(), load_iv(%{{.+}}:vector<64xf16>)
// CHECK: dataflow.program_unit iter_arg : %[[LOAD_KEY:.+]] -> (%[[LOAD_UNIT]]) : {
// CHECK: %[[LOAD_MEMORY_MAP:.+]] = uniform.def_immutable_mapping([%[[LOAD_UNIT]] -> %[[MEMORY_UNIT]]]):index
// CHECK-NEXT: %[[LOAD_MEMORY:.+]] = uniform.query_map(map:%[[LOAD_MEMORY_MAP]], key:%[[LOAD_KEY]]) : index
// CHECK: scf.for %{{.+}} = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT: scf.for %{{.+}} = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT: scf.for %{{.+}} = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: %[[LOAD_BUFFER:.+]] = dataflow.get_logical_memory_view %[[LOAD_MEMORY]], %[[MID_BASE:[^ ]+]] {layout_map = #[[MID_LAYOUT]]} : index, index, memref<1x64xf32>
// CHECK: %[[INPUT:.+]] = agen.vector_load %{{.+}}[0, 0] {{.+}} : memref<1x64xf16>, vector<64xf16>
// CHECK-NEXT: dataflow.send %{{.+}}, %[[INPUT]] : vector<64xf16>
// CHECK-NEXT: %[[WAIT_MAP:.+]] = uniform.def_immutable_mapping([%[[LOAD_UNIT]] -> %[[STORE_UNIT]]]):index
// CHECK-NEXT: %[[WAIT_PEER:.+]] = uniform.query_map(map:%[[WAIT_MAP]], key:%[[LOAD_KEY]]) : index
// CHECK-NEXT: dataflow.sync_send %[[WAIT_PEER]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT: dataflow.sync_recv %[[WAIT_PEER]] : index
// CHECK: %[[L0:.+]] = agen.vector_load %[[LOAD_BUFFER]][0, 0] {load_order = #{{.+}}, load_set = #[[HALF]]} : memref<1x64xf32>, vector<32xf32>
// CHECK-NEXT: dataflow.send %[[LOAD_DEST:.+]], %[[L0]] : vector<32xf32>
// CHECK-NEXT: %[[L1:.+]] = agen.vector_load %[[LOAD_BUFFER]][0, 32] {load_order = #{{.+}}, load_set = #[[HALF]]} : memref<1x64xf32>, vector<32xf32>
// CHECK-NEXT: dataflow.send %[[LOAD_DEST]], %[[L1]] : vector<32xf32>
// CHECK-NOT: agen.vector_load
// CHECK: dataflow.program_unit
// CHECK: scf.for %{{.+}} = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT: scf.for %{{.+}} = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT: scf.for %{{.+}} = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: %[[IN:.+]] = dataflow.receive %{{.+}} : vector<64xf16>
// CHECK: %[[W0:.+]] = vectorchain.cast %[[IN]] : vector<64xf16>, vector<64xf32>
// CHECK-NEXT: %[[LO:.+]] = vectorchain.shuffle input(%[[W0]]) {indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32], repetition = 8 : i32} : vector<64xf32>, vector<32xf32>
// CHECK-NEXT: dataflow.send %[[WIDE_DEST:.+]], %[[LO]] : vector<32xf32>
// CHECK-NEXT: %[[W1:.+]] = vectorchain.cast %[[IN]] : vector<64xf16>, vector<64xf32>
// CHECK-NEXT: %[[HI:.+]] = vectorchain.shuffle input(%[[W1]]) {indices = [4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : i32} : vector<64xf32>, vector<32xf32>
// CHECK-NEXT: dataflow.send %[[WIDE_DEST]], %[[HI]] : vector<32xf32>
// CHECK: %[[R0:.+]] = dataflow.receive %[[NARROW_SRC:.+]] : vector<32xf32>
// CHECK-NEXT: %[[R1:.+]] = dataflow.receive %[[NARROW_SRC]] : vector<32xf32>
// CHECK-NEXT: %[[N0:.+]] = vectorchain.cast %[[R0]] : vector<32xf32>, vector<32xf16>
// CHECK-NEXT: %[[N1:.+]] = vectorchain.cast %[[R1]] : vector<32xf32>, vector<32xf16>
// CHECK-NEXT: %[[PACKED:.+]] = vectorchain.pack %[[N0]], %[[N1]] {indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32, 5 : i32, 6 : i32, 7 : i32], repetition = 8 : index} : vector<32xf16>, vector<32xf16>, vector<64xf16>
// CHECK: dataflow.send %{{.+}}, %[[PACKED]] : vector<64xf16>
// CHECK-NOT: vectorchain.cast
// CHECK: dataflow.program_unit iter_arg : %[[STORE_KEY:.+]] -> (%[[STORE_UNIT]]) : {
// CHECK: %[[STORE_MEMORY_MAP:.+]] = uniform.def_immutable_mapping([%[[STORE_UNIT]] -> %[[MEMORY_UNIT]]]):index
// CHECK-NEXT: %[[STORE_MEMORY:.+]] = uniform.query_map(map:%[[STORE_MEMORY_MAP]], key:%[[STORE_KEY]]) : index
// CHECK: scf.for %{{.+}} = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT: scf.for %{{.+}} = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT: scf.for %{{.+}} = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: %[[STORE_BUFFER:.+]] = dataflow.get_logical_memory_view %[[STORE_MEMORY]], %[[MID_BASE]] {layout_map = #[[MID_LAYOUT]]} : index, index, memref<1x64xf32>
// CHECK: %[[S0:.+]] = dataflow.receive %[[STORE_SRC:.+]] : vector<32xf32>
// CHECK-NEXT: agen.vector_store %[[S0]], %[[STORE_BUFFER]][0, 0] {store_order = #{{.+}}, store_set = #[[HALF]]} : memref<1x64xf32>, vector<32xf32>
// CHECK-NEXT: %[[S1:.+]] = dataflow.receive %[[STORE_SRC]] : vector<32xf32>
// CHECK-NEXT: agen.vector_store %[[S1]], %[[STORE_BUFFER]][0, 32] {store_order = #{{.+}}, store_set = #[[HALF]]} : memref<1x64xf32>, vector<32xf32>
// CHECK-NEXT: %[[READY_MAP:.+]] = uniform.def_immutable_mapping([%[[STORE_UNIT]] -> %[[LOAD_UNIT]]]):index
// CHECK-NEXT: %[[READY_PEER:.+]] = uniform.query_map(map:%[[READY_MAP]], key:%[[STORE_KEY]]) : index
// CHECK-NEXT: dataflow.sync_send %[[READY_PEER]] {wait_immediately_for_async_transfers = true} : index
// CHECK-NEXT: dataflow.sync_recv %[[READY_PEER]] : index
// CHECK: %[[OUTPUT:.+]] = dataflow.receive %{{.+}} : vector<64xf16>
// CHECK-NEXT: agen.vector_store %[[OUTPUT]], %{{.+}}[0, 0] {{.+}} : memref<1x64xf16>, vector<64xf16>
// CHECK-NOT: dataflow.receive
// CHECK: dataflow.program_unit
// CHECK: %[[GLOBAL_OUTPUT:.+]] = dataflow.get_logical_memory_view %{{.+}} {layout_map = #[[GLOBAL_LAYOUT]]} : index, index, memref<4x128xf16>
// CHECK: scf.for %[[STORE_ROW_HI:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT: scf.for %[[STORE_ROW_LO:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK-NEXT: scf.for %[[STORE_COL:.+]] = %[[C0]] to %[[C2]] step %[[C1]] {
// CHECK: agen.composite_load_and_store src:%{{.+}}[0, 0] dst:%[[GLOBAL_OUTPUT]][%[[STORE_ROW_HI]] * 2 + %[[STORE_ROW_LO]], %[[STORE_COL]] * 64]
// CHECK-NEXT: time_symbols(), load_iv(%{{.+}}:vector<64xf16>)
// CHECK: return

#map = affine_map<(d0, d1) -> (d0, d1)>
#set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 3 >= 0, d1 >= 0, -d1 + 127 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @dl16_fp32_dl16_tiles(%input: index, %output: index) attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %input_view = ktdp.construct_memory_view %input, sizes: [4, 128], strides: [128, 1] {
      coordinate_set = #set, memory_space = #ktdp.memory_space<global>
    } : memref<4x128xf16>
    %input_tile = ktdp.construct_access_tile %input_view[%c0, %c0] {
      access_tile_order = #map, access_tile_set = #set
    } : memref<4x128xf16> -> !ktdp.access_tile<4x128xindex>
    %values = ktdp.load %input_tile : !ktdp.access_tile<4x128xindex> -> tensor<4x128xf16>
    %wide_empty = tensor.empty() : tensor<4x128xf32>
    %wide = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%values : tensor<4x128xf16>) outs(%wide_empty : tensor<4x128xf32>) {
    ^bb0(%in: f16, %out: f32):
      %cast = arith.extf %in : f16 to f32
      linalg.yield %cast : f32
    } -> tensor<4x128xf32>
    %narrow_empty = tensor.empty() : tensor<4x128xf16>
    %result = linalg.generic {
        indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
      ins(%wide : tensor<4x128xf32>) outs(%narrow_empty : tensor<4x128xf16>) {
    ^bb0(%in: f32, %out: f16):
      %cast = arith.truncf %in : f32 to f16
      linalg.yield %cast : f16
    } -> tensor<4x128xf16>
    %output_view = ktdp.construct_memory_view %output, sizes: [4, 128], strides: [128, 1] {
      coordinate_set = #set, memory_space = #ktdp.memory_space<global>
    } : memref<4x128xf16>
    %output_tile = ktdp.construct_access_tile %output_view[%c0, %c0] {
      access_tile_order = #map, access_tile_set = #set
    } : memref<4x128xf16> -> !ktdp.access_tile<4x128xindex>
    ktdp.store %result, %output_tile : tensor<4x128xf16>, !ktdp.access_tile<4x128xindex>
    return
  }
}
