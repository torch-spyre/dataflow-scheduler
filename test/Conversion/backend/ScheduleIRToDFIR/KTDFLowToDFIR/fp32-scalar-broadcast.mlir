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

// RUN: dataflow-scheduler-opt %s -splat-legalization -ktdflowering-to-dfir | \
// RUN:   FileCheck %s --implicit-check-not=ktdf.read_from_fifo \
// RUN:   --implicit-check-not=ktdf.write_to_fifo \
// RUN:   --implicit-check-not=ktdf.data_transfer \
// RUN:   --implicit-check-not=bufferization.to_tensor

// A 16B read starts at element 4, not at the beginning of the backing stick.
// The same scalar must reach all 32 lanes of a real output store. In particular,
// the three neighbours included in the load must not escape the shuffle in
// the compute unit.
// CHECK-LABEL: func.func @fp32_scalar_broadcast()
// CHECK: %[[OFFSET:.*]] = arith.constant 4 : index
// CHECK: dataflow.program_unit
// CHECK: %[[LOAD:.*]] = agen.vector_load %{{.*}}[%[[OFFSET]]]
// CHECK-SAME: vector<4xf32>
// CHECK-NEXT: %[[PACKET:.*]] = vectorchain.shuffle input(%[[LOAD]])
// CHECK-SAME: indices = [0 : i32, 1 : i32, 2 : i32, 3 : i32], repetition = 8 : i32
// CHECK: dataflow.send %{{.*}}, %[[PACKET]] : vector<32xf32>
// CHECK: dataflow.program_unit
// CHECK: %[[RECEIVED:.*]] = dataflow.receive %{{.*}} : vector<32xf32>
// CHECK-NEXT: %[[BROADCAST:.*]] = vectorchain.shuffle input(%[[RECEIVED]])
// CHECK-SAME: indices = [0 : i32, 0 : i32, 0 : i32, 0 : i32], repetition = 8 : i32
// CHECK-NOT: agen.vector_store
// CHECK: dataflow.send %{{.*}}, %[[BROADCAST]] : vector<32xf32>
// CHECK: dataflow.program_unit
// CHECK: %[[OUTPUT:.*]] = dataflow.receive %{{.*}} : vector<32xf32>
// CHECK-NEXT: agen.vector_store %[[OUTPUT]],
// CHECK-SAME: vector<32xf32>

module {
  ktdf_arch.device @sample_device
      import("../../../../Transforms/Inputs/fp32-splat-device.mlir")
  func.func @fp32_scalar_broadcast() attributes {grid = [1]} {
    %load_resource = dataflow.get_unit {
      core = 0 : i32, name = "C0-L1LU", type = "L1LU"
    } : index
    %compute_resource = dataflow.get_unit {
      core = 0 : i32, name = "C0-SFU", type = "SFU"
    } : index
    %store_resource = dataflow.get_unit {
      core = 0 : i32, name = "C0-L1SU", type = "L1SU"
    } : index
    %tile = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %load_map = uniform.def_immutable_mapping([%c0 -> %load_resource]) : index
    %load_unit = uniform.query_map(map: %load_map, key: %tile) : index
    %compute_map = uniform.def_immutable_mapping([%c0 -> %compute_resource])
        : index
    %compute_unit = uniform.query_map(map: %compute_map, key: %tile) : index
    %store_map = uniform.def_immutable_mapping([%c0 -> %store_resource]) : index
    %store_unit = uniform.query_map(map: %store_map, key: %tile) : index
    %input = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    %output = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
    ktdf_lowering.execute_on %load_unit {
      %source = memref.alloc() : memref<32xf32, "L1">
      ktdf.data_transfer from %source[%c4] size [1]
          to %input size [32] {transfer_mode = #ktdf.transfer_mode<splat>}
          : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    }
    ktdf_lowering.execute_on %compute_unit {
      %value = ktdf.read_from_fifo %input
          : <"L1LU" -> "SFU", 32xf32> -> tensor<32xf32>
      ktdf.write_to_fifo %value, %output
          : tensor<32xf32>, <"SFU" -> "L1SU", 32xf32>
    }
    ktdf_lowering.execute_on %store_unit {
      %destination = memref.alloc() : memref<32xf32, "L1">
      ktdf.data_transfer from %output size [32]
          to %destination[%c0] size [32]
          : !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, memref<32xf32, "L1">
    }
    return
  }
}
