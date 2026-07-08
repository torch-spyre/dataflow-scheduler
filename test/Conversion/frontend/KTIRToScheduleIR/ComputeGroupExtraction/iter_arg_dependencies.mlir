// RUN: dataflow-scheduler-opt --compute-group-extraction %s | FileCheck %s



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK:   module {
// CHECK:           func.func @test_loops() {
// CHECK:             %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:             %[[CONSTANT_1:.*]] = arith.constant 5 : index
// CHECK:             scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_0]] {
// CHECK:               func.call @"local-schedule-0"(%[[VAL_0]]) : (index) -> ()
// CHECK:             }
// CHECK:             return
// CHECK:           }
// CHECK:           func.func private @"local-schedule-0"(index)
// CHECK:         }

// CHECK:   module {
// CHECK:           func.func @"local-schedule-0"(%[[VAL_0:.*]]: index) {
// CHECK:             %[[CONSTANT_0:.*]] = arith.constant 1024 : index
// CHECK:             %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16>
// CHECK:             %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK:             %[[CONSTANT_1:.*]] = arith.constant 3 : index
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[MULI_0]], %[[VAL_0]] : index
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:             %[[CONSTRUCT_ACCESS_TILE_0:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_0]]{{\[}}%[[ADDI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
// CHECK:             %[[CONSTANT_3:.*]] = arith.constant 12288 : index
// CHECK:             %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16>
// CHECK:             %[[CONSTRUCT_ACCESS_TILE_1:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_1]]{{\[}}%[[ADDI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
// CHECK:             %[[LOAD_0:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_0]] : <3x64xindex> -> tensor<3x64xf16>
// CHECK:             %[[LOAD_1:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_1]] : <3x64xindex> -> tensor<3x64xf16>
// CHECK:             %[[EMPTY_0:.*]] = tensor.empty() : tensor<3x64xf16>
// CHECK:             %[[ADD_0:.*]] = linalg.add ins(%[[LOAD_0]], %[[LOAD_1]] : tensor<3x64xf16>, tensor<3x64xf16>) outs(%[[EMPTY_0]] : tensor<3x64xf16>) -> tensor<3x64xf16>
// CHECK:             %[[CONSTANT_4:.*]] = arith.constant 18432 : index
// CHECK:             %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16>
// CHECK:             %[[CONSTRUCT_ACCESS_TILE_2:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_2]]{{\[}}%[[ADDI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
// CHECK:             ktdp.store %[[ADD_0]], %[[CONSTRUCT_ACCESS_TILE_2]] : tensor<3x64xf16>, <3x64xindex>
// CHECK:             return
// CHECK:           }
// CHECK:         }


module {
  func.func @test_loops() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %tile_size = arith.constant 3 : index
    %A_addr = arith.constant 1024 : index
    %B_addr = arith.constant 12288 : index
    %C_addr = arith.constant 18432 : index
    %E_addr = arith.constant 24576 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    // Construct memory views for A, B, C
    %A_view = ktdp.construct_memory_view %A_addr, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>
    %B_view = ktdp.construct_memory_view %B_addr, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>
    %C_view = ktdp.construct_memory_view %C_addr, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>

    scf.for %i = %c1 to %c5 step %c1 {
        %start_row_i = arith.addi %start_row, %i : index
        %A_access_tile = ktdp.construct_access_tile %A_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

        %B_access_tile = ktdp.construct_access_tile %B_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

        %C_access_tile = ktdp.construct_access_tile %C_view[%start_row_i, %c0] {
            access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
            access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
        } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

        // Compute group: A + B = C
        %A = ktdp.load %A_access_tile : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>
        %B = ktdp.load %B_access_tile : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>
        %C_empty = tensor.empty() : tensor<3x64xf16>
        %C = linalg.add ins(%A, %B : tensor<3x64xf16>, tensor<3x64xf16>) 
                        outs(%C_empty : tensor<3x64xf16>) -> tensor<3x64xf16>
        ktdp.store %C, %C_access_tile : tensor<3x64xf16>, !ktdp.access_tile<3x64xindex>
    }
    return
  }
}