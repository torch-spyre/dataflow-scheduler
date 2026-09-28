// RUN: dataflow-scheduler-opt --ktir-map-and-tile --ktir-bufferize --ktir-pipeline %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   func.func @local_schedule_1(
// CHECK-SAME:      %[[ARG0:.*]]: index) {
// CHECK-NEXT:     %[[CONSTANT_0:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[CONSTANT_3:.*]] = arith.constant 3 : index
// CHECK-NEXT:     %[[CONSTANT_4:.*]] = arith.constant 1024 : index
// CHECK-NEXT:     %[[CONSTANT_5:.*]] = arith.constant 12288 : index
// CHECK-NEXT:     %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:     %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_3]] : index
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:     %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_5]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:     %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[ARG0]] : index
// CHECK-NEXT:     %[[MULI_1:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_0]] : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<96x64xf16> to memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: {{\[}}%[[MULI_1]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     %[[MULI_2:.*]] = arith.muli %[[ADDI_0]], %[[CONSTANT_0]] : index
// CHECK-NEXT:     %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<96x64xf16> to memref<96x64xf16, "DDR">
// CHECK-NEXT:     %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: {{\[}}%[[MULI_2]]], sizes: [1, 64], strides: [64, 1] : memref<96x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:     scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_1]] step %[[CONSTANT_1]] {
// CHECK-NEXT:       scf.for %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_0]] {
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[PRIVATE_0:.*]]:10 = ktdf.private -> (!ktdf.fifo.slot<"MNISU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"SFU" -> "MNISU", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"MNILU" -> "L1", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"DDR" -> "MNILU", 64xf16>, !ktdf.token) {
// CHECK-NEXT:             %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNISU" -> "DDR", 64xf16>
// CHECK-NEXT:             %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "MNISU", 64xf16>
// CHECK-NEXT:             %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[FIFO_2:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
// CHECK-NEXT:             %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[FIFO_3:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"MNILU" -> "L1", 64xf16>
// CHECK-NEXT:             %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[FIFO_4:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "MNILU", 64xf16>
// CHECK-NEXT:             %[[CREATE_TOKEN_4:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[FIFO_0]], %[[CREATE_TOKEN_0]], %[[FIFO_1]], %[[CREATE_TOKEN_1]], %[[FIFO_2]], %[[CREATE_TOKEN_2]], %[[FIFO_3]], %[[CREATE_TOKEN_3]], %[[FIFO_4]], %[[CREATE_TOKEN_4]] : !ktdf.fifo.slot<"MNISU" -> "DDR", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"SFU" -> "MNISU", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"MNILU" -> "L1", 64xf16>, !ktdf.token, !ktdf.fifo.slot<"DDR" -> "MNILU", 64xf16>, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[PRIVATE_0:.*]]#9) {
// CHECK-NEXT:             ktdf.data_transfer from %[[REINTERPRET_CAST_0]]{{\[}}%[[VAL_0]], %[[VAL_1]]] size [1, 64] to %[[PRIVATE_0]]#8 size [] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"DDR" -> "MNILU", 64xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[PRIVATE_0:.*]]#9) depends_out(%[[PRIVATE_0]]#7) {
// CHECK-NEXT:             ktdf.data_transfer from %[[PRIVATE_0]]#8 size [] to %[[PRIVATE_0]]#6 size [] : !ktdf.fifo.slot<"DDR" -> "MNILU", 64xf16>, !ktdf.fifo.slot<"MNILU" -> "L1", 64xf16>
// CHECK-NEXT:           } {applicable_units = ["MNILU"]}
// CHECK-NEXT:           ktdf.stage depends_in(%[[PRIVATE_0:.*]]#7) depends_out(%[[PRIVATE_0]]#5) {
// CHECK-NEXT:             ktdf.data_transfer from %[[PRIVATE_0]]#6 size [] to %[[PRIVATE_0]]#4 size [] : !ktdf.fifo.slot<"MNILU" -> "L1", 64xf16>, !ktdf.fifo.slot<"L1" -> "SFU", 64xf16>
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[PRIVATE_0:.*]]#5) depends_out(%[[PRIVATE_0]]#3) {
// CHECK-NEXT:             %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[PRIVATE_0]]#4 : <"L1" -> "SFU", 64xf16> -> tensor<1x64xf16>
// CHECK-NEXT:             %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:             %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<1x64xf16>) outs(%[[EMPTY_0]] : tensor<1x64xf16>) attrs =  {dataflow_scheduler.throttle = 64 : i64, ktdf_arch.maps_to = "SFU"} {
// CHECK-NEXT:             ^bb0(%[[VAL_6:.*]]: f16, %[[VAL_7:.*]]: f16):
// CHECK-NEXT:               %[[SQRT_0:.*]] = math.sqrt %[[VAL_6]] : f16
// CHECK-NEXT:               linalg.yield %[[SQRT_0]] : f16
// CHECK-NEXT:             } -> tensor<1x64xf16>
// CHECK-NEXT:             ktdf.write_to_fifo %[[GENERIC_0]], %[[PRIVATE_0]]#2 : tensor<1x64xf16>, <"SFU" -> "MNISU", 64xf16>
// CHECK-NEXT:           } {applicable_units = ["SFU"]}
// CHECK-NEXT:           ktdf.stage depends_in(%[[PRIVATE_0:.*]]#3) depends_out(%[[PRIVATE_0]]#1) {
// CHECK-NEXT:             ktdf.data_transfer from %[[PRIVATE_0]]#2 size [] to %[[PRIVATE_0]]#0 size [] : !ktdf.fifo.slot<"SFU" -> "MNISU", 64xf16>, !ktdf.fifo.slot<"MNISU" -> "DDR", 64xf16>
// CHECK-NEXT:           } {applicable_units = ["MNISU"]}
// CHECK-NEXT:           ktdf.stage depends_in(%[[PRIVATE_0:.*]]#1) depends_out(none) {
// CHECK-NEXT:             ktdf.data_transfer from %[[PRIVATE_0]]#0 size [] to %[[REINTERPRET_CAST_1]]{{\[}}%[[VAL_0]], %[[VAL_1]]] size [1, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"MNISU" -> "DDR", 64xf16>, memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#id = affine_map<(d0, d1) -> (d0, d1)>

module {
    ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../../../Dialect/KTDFArch/sample_device.mlir")
    func.func @local_schedule_1(%i: index) {
        %c0 = arith.constant 0 : index
        %tile_size = arith.constant 3 : index
        %A_start_address = arith.constant 1024 : index
        %B_start_address = arith.constant 12288 : index
    
        %id = ktdp.get_compute_tile_id : index
        %start_row = arith.muli %id, %tile_size : index
    
        // Construct a memory view of A from a given address
        %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.memory_space<global>
        } : memref<96x64xf16>
        // Construct a memory view of B from a given address
        %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
            coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            memory_space = #ktdp.memory_space<global>
        } : memref<96x64xf16>
        
        %start_row_i = arith.addi %start_row, %i : index
        
        // Construct an access tile from the memory view of A
        %A_access_tile = ktdp.construct_access_tile %A_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Load data from the corresponding access tile
        %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

        %A_hop = ktdf.via["MNILU"] %A_data_tile : tensor<1x64xf16>
        %A_hop_2 = ktdf.via["L1"] %A_hop : tensor<1x64xf16>

        %B_data_tile = tensor.empty() : tensor<1x64xf16>
        %result = linalg.generic {indexing_maps=[#id, #id], iterator_types=["parallel", "parallel"]}
            ins(%A_hop_2 : tensor<1x64xf16>)
            outs(%B_data_tile : tensor<1x64xf16>) {
        ^bb0(%a: f16, %b: f16):
            %sqrt = math.sqrt %a : f16
            linalg.yield %sqrt : f16
        } -> tensor<1x64xf16>

        %result_hop = ktdf.via["MNISU"] %result : tensor<1x64xf16>

        // Construct an access tile from the memory view of B
        %B_access_tile = ktdp.construct_access_tile %B_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

        // Store data into the access tile.
        ktdp.store %result_hop, %B_access_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
        return
    }
}
