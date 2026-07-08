// RUN: dataflow-scheduler-opt --compute-group-extraction %s | FileCheck %s



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK:   module {
// CHECK:           func.func @test_nested_loops() {
// CHECK:             %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:             %[[CONSTANT_1:.*]] = arith.constant 4 : index
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 3 : index
// CHECK:             scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_0]] {
// CHECK:               scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] {
// CHECK:                 func.call @"local-schedule-0"() : () -> ()
// CHECK:               }
// CHECK:             }
// CHECK:             return
// CHECK:           }
// CHECK:           func.func private @"local-schedule-0"()
// CHECK:         }

// CHECK:   module {
// CHECK:           func.func @"local-schedule-0"() {
// CHECK:             %[[CONSTANT_0:.*]] = arith.constant 2048 : index
// CHECK:             %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16>
// CHECK:             %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK:             %[[CONSTANT_1:.*]] = arith.constant 3 : index
// CHECK:             %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_1]] : index
// CHECK:             %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:             %[[CONSTRUCT_ACCESS_TILE_0:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_0]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
// CHECK:             %[[CONSTANT_3:.*]] = arith.constant 4096 : index
// CHECK:             %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16>
// CHECK:             %[[CONSTRUCT_ACCESS_TILE_1:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_1]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
// CHECK:             %[[LOAD_0:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_0]] : <3x64xindex> -> tensor<3x64xf16>
// CHECK:             %[[LOAD_1:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_1]] : <3x64xindex> -> tensor<3x64xf16>
// CHECK:             %[[EMPTY_0:.*]] = tensor.empty() : tensor<3x64xf16>
// CHECK:             %[[ADD_0:.*]] = linalg.add ins(%[[LOAD_0]], %[[LOAD_1]] : tensor<3x64xf16>, tensor<3x64xf16>) outs(%[[EMPTY_0]] : tensor<3x64xf16>) -> tensor<3x64xf16>
// CHECK:             %[[CONSTANT_4:.*]] = arith.constant 8192 : index
// CHECK:             %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<96x64xf16>
// CHECK:             %[[CONSTRUCT_ACCESS_TILE_2:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_2]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>
// CHECK:             ktdp.store %[[ADD_0]], %[[CONSTRUCT_ACCESS_TILE_2]] : tensor<3x64xf16>, <3x64xindex>
// CHECK:             return
// CHECK:           }
// CHECK:         }


module {
  func.func @test_nested_loops() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %tile_size = arith.constant 3 : index
    %A_addr = arith.constant 2048 : index
    %B_addr = arith.constant 4096 : index
    %C_addr = arith.constant 8192 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    // Construct memory views and access tiles for A, B, C
    %A_view = ktdp.construct_memory_view %A_addr, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>
    %A_access_tile = ktdp.construct_access_tile %A_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

    %B_view = ktdp.construct_memory_view %B_addr, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>
    %B_access_tile = ktdp.construct_access_tile %B_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

    %C_view = ktdp.construct_memory_view %C_addr, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.spyre_memory_space<HBM>
    } : memref<96x64xf16>
    %C_access_tile = ktdp.construct_access_tile %C_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 2 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<3x64xindex>

    // Nested loops: outer loop iterates 4 times, inner loop iterates 3 times
    // The compute group inside the inner loop should be extracted
    scf.for %i = %c1 to %c4 step %c1 {
      scf.for %j = %c1 to %c3 step %c1 {
        %A_data = ktdp.load %A_access_tile : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>
        %B_data = ktdp.load %B_access_tile : !ktdp.access_tile<3x64xindex> -> tensor<3x64xf16>
        %C_empty = tensor.empty() : tensor<3x64xf16>
        %C_data = linalg.add ins(%A_data, %B_data : tensor<3x64xf16>, tensor<3x64xf16>) 
                             outs(%C_empty : tensor<3x64xf16>) -> tensor<3x64xf16>
        ktdp.store %C_data, %C_access_tile : tensor<3x64xf16>, !ktdp.access_tile<3x64xindex>
      }
    }
    
    return
  }
}
