// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   func.func @tensor_typed_case() attributes {grid = [2]} {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
// CHECK-NEXT:     %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
// CHECK-NEXT:     %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_0:.*]] -> (%[[GET_UNIT_0]], %[[GET_UNIT_1]]) : {
// CHECK-NEXT:       %[[ALLOC_0:.*]] = memref.alloc() : memref<1x64xf16, "L1">
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       %[[VECTOR_LOAD_0:.*]] = agen.vector_load %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {load_order = #[[$ATTR_0]], load_set = #[[$ATTR_2]]} : memref<1x64xf16, "L1">, vector<64xf16>
// CHECK-NEXT:       dataflow.send %[[QUERY_MAP_0]], %[[VECTOR_LOAD_0]] : vector<64xf16>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_0]] -> %[[GET_UNIT_2]]], {{\[}}%[[GET_UNIT_1]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[VAL_0]]) : index
// CHECK-NEXT:       %[[RECEIVE_0:.*]] = dataflow.receive %[[QUERY_MAP_1]] : vector<64xf16>
// CHECK-NEXT:       agen.vector_store %[[RECEIVE_0]], %[[ALLOC_0]]{{\[}}%[[CONSTANT_0]], %[[CONSTANT_0]]] {store_order = #[[$ATTR_0]], store_set = #[[$ATTR_2]]} : memref<1x64xf16, "L1">, vector<64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     dataflow.program_unit iter_arg : %[[VAL_1:.*]] -> (%[[GET_UNIT_2]], %[[GET_UNIT_3]]) : {
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_2:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_2]], key:%[[VAL_1]]) : index
// CHECK-NEXT:       %[[RECEIVE_1:.*]] = dataflow.receive %[[QUERY_MAP_2]] : vector<64xf16>
// CHECK-NEXT:       %[[BINARY_0:.*]] = vectorchain.binary %[[RECEIVE_1]], %[[RECEIVE_1]] {binary_op = #vectorchain<binary_operator mul>, op_specific_map = #[[$ATTR_1]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_3:.*]] = uniform.def_immutable_mapping({{\[}}%[[GET_UNIT_2]] -> %[[GET_UNIT_0]]], {{\[}}%[[GET_UNIT_3]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_3:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_3]], key:%[[VAL_1]]) : index
// CHECK-NEXT:       dataflow.send %[[QUERY_MAP_3]], %[[BINARY_0]] : vector<64xf16>
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }



// Verifies the tensor-semantics linalg.generic lowering path:
//   - input block arg replaced by the received vector (ktdf.read_from_fifo →
//     dataflow.receive on the SFU side)
//   - arith.mulf in the body → vectorchain.binary {mul}
//   - the generic result (tensor) is replaced by the binary output vector,
//     which flows directly into write_to_fifo → dataflow.send
//
// This exercises LowerLinalgGenericPattern::matchAndRewrite when
// hasPureTensorSemantics() is true and there are no reduction iterators.

// The SFU program_unit must contain a receive, a mul binary, and a send —
// no alloc, no vector_load/store from a register.

#id = affine_map<(d0) -> (d0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @tensor_typed_case() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x64xf16, "L1">
    %in_fifo  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    %out_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %alloc_l1[%c0, %c0] size [1, 64] to %in_fifo size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf_lowering.execute_on %u_sfu {
      %input  = ktdf.read_from_fifo %in_fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %init   = tensor.empty() : tensor<64xf16>
      %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel"]
      } ins(%input : tensor<64xf16>) outs(%init : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %v = arith.mulf %in, %in : f16
        linalg.yield %v : f16
      } -> tensor<64xf16>
      ktdf.write_to_fifo %result, %out_fifo : tensor<64xf16>, <"SFU" -> "L1LU", 64xf16>
    }
    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %out_fifo size [64] to %alloc_l1[%c0, %c0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>, memref<1x64xf16, "L1">
    }
    return
  }
}
