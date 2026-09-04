// RUN: dataflow-scheduler-opt --compute-group-extraction %s | FileCheck %s



// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_2:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   module {
// CHECK:     func.func @add() attributes {grid = [4]} {
// CHECK:       call @local_schedule_0() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func @mulf() attributes {grid = [4]} {
// CHECK:       call @local_schedule_1() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func @max() attributes {grid = [4]} {
// CHECK:       call @local_schedule_2() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func @min() attributes {grid = [4]} {
// CHECK:       call @local_schedule_3() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func @absmax_maxnumf() attributes {grid = [4]} {
// CHECK:       call @local_schedule_4() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func private @local_schedule_0()
// CHECK:     func.func private @local_schedule_1()
// CHECK:     func.func private @local_schedule_2()
// CHECK:     func.func private @local_schedule_3()
// CHECK:     func.func private @local_schedule_4()
// CHECK:   }

// CHECK-LABEL:   module @local_schedule_0 {
// CHECK-NEXT:     func.func @local_schedule_0() attributes {grid = [4]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 1024 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_0:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_0]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 12288 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_1:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_1]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       %[[LOAD_0:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_0]] : <2x64xindex> -> tensor<2x64xf16>
// CHECK-NEXT:       %[[LOAD_1:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_1]] : <2x64xindex> -> tensor<2x64xf16>
// CHECK-NEXT:       %[[EMPTY_0:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 18432 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[ADD_0:.*]] = linalg.add ins(%[[LOAD_0]], %[[LOAD_1]] : tensor<2x64xf16>, tensor<2x64xf16>) outs(%[[EMPTY_0]] : tensor<2x64xf16>) -> tensor<2x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_2:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_2]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       ktdp.store %[[ADD_0]], %[[CONSTRUCT_ACCESS_TILE_2]] : tensor<2x64xf16>, <2x64xindex>
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }

// CHECK-LABEL:   module @local_schedule_1 {
// CHECK-NEXT:     func.func @local_schedule_1() attributes {grid = [4]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 1024 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_0:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_0]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 12288 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_1:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_1]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       %[[LOAD_0:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_0]] : <2x64xindex> -> tensor<2x64xf16>
// CHECK-NEXT:       %[[LOAD_1:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_1]] : <2x64xindex> -> tensor<2x64xf16>
// CHECK-NEXT:       %[[EMPTY_0:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 18432 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[MUL_0:.*]] = linalg.mul ins(%[[LOAD_0]], %[[LOAD_1]] : tensor<2x64xf16>, tensor<2x64xf16>) outs(%[[EMPTY_0]] : tensor<2x64xf16>) -> tensor<2x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_2:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_2]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       ktdp.store %[[MUL_0]], %[[CONSTRUCT_ACCESS_TILE_2]] : tensor<2x64xf16>, <2x64xindex>
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }

// CHECK-LABEL:   module @local_schedule_2 {
// CHECK-NEXT:     func.func @local_schedule_2() attributes {grid = [4]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 1024 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_0:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_0]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 12288 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_1:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_1]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       %[[LOAD_0:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_0]] : <2x64xindex> -> tensor<2x64xf16>
// CHECK-NEXT:       %[[LOAD_1:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_1]] : <2x64xindex> -> tensor<2x64xf16>
// CHECK-NEXT:       %[[EMPTY_0:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 18432 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[MAX_0:.*]] = linalg.max ins(%[[LOAD_0]], %[[LOAD_1]] : tensor<2x64xf16>, tensor<2x64xf16>) outs(%[[EMPTY_0]] : tensor<2x64xf16>) -> tensor<2x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_2:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_2]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       ktdp.store %[[MAX_0]], %[[CONSTRUCT_ACCESS_TILE_2]] : tensor<2x64xf16>, <2x64xindex>
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }

// CHECK-LABEL:   module @local_schedule_3 {
// CHECK-NEXT:     func.func @local_schedule_3() attributes {grid = [4]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 1024 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_0:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_0]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 12288 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_1:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_1]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       %[[LOAD_0:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_0]] : <2x64xindex> -> tensor<2x64xf16>
// CHECK-NEXT:       %[[LOAD_1:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_1]] : <2x64xindex> -> tensor<2x64xf16>
// CHECK-NEXT:       %[[EMPTY_0:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 18432 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[MIN_0:.*]] = linalg.min ins(%[[LOAD_0]], %[[LOAD_1]] : tensor<2x64xf16>, tensor<2x64xf16>) outs(%[[EMPTY_0]] : tensor<2x64xf16>) -> tensor<2x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_2:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_2]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       ktdp.store %[[MIN_0]], %[[CONSTRUCT_ACCESS_TILE_2]] : tensor<2x64xf16>, <2x64xindex>
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }

// CHECK-LABEL:   module @local_schedule_4 {
// CHECK-NEXT:     func.func @local_schedule_4() attributes {grid = [4]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 1024 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[MULI_0:.*]] = arith.muli %[[GET_COMPUTE_TILE_ID_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_0:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_0]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 12288 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_3]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_1:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_1]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       %[[LOAD_0:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_0]] : <2x64xindex> -> tensor<2x64xf16>
// CHECK-NEXT:       %[[LOAD_1:.*]] = ktdp.load %[[CONSTRUCT_ACCESS_TILE_1]] : <2x64xindex> -> tensor<2x64xf16>
// CHECK-NEXT:       %[[EMPTY_0:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 18432 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_4]], sizes: [96, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_1]], memory_space = #ktdp.memory_space<global>} : memref<96x64xf16>
// CHECK-NEXT:       %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[LOAD_0]], %[[LOAD_1]] : tensor<2x64xf16>, tensor<2x64xf16>) outs(%[[EMPTY_0]] : tensor<2x64xf16>) {
// CHECK-NEXT:       ^bb0(%[[VAL_0:.*]]: f16, %[[VAL_1:.*]]: f16, %[[VAL_2:.*]]: f16):
// CHECK-NEXT:         %[[ABSF_0:.*]] = math.absf %[[VAL_0]] : f16
// CHECK-NEXT:         %[[ABSF_1:.*]] = math.absf %[[VAL_1]] : f16
// CHECK-NEXT:         %[[MAXNUMF_0:.*]] = arith.maxnumf %[[ABSF_0]], %[[ABSF_1]] : f16
// CHECK-NEXT:         linalg.yield %[[MAXNUMF_0]] : f16
// CHECK-NEXT:       } -> tensor<2x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_ACCESS_TILE_2:.*]] = ktdp.construct_access_tile %[[CONSTRUCT_MEMORY_VIEW_2]]{{\[}}%[[MULI_0]], %[[CONSTANT_2]]] {access_tile_order = #[[$ATTR_0]], access_tile_set = #[[$ATTR_2]]} : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>
// CHECK-NEXT:       ktdp.store %[[GENERIC_0]], %[[CONSTRUCT_ACCESS_TILE_2]] : tensor<2x64xf16>, <2x64xindex>
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }

module {
  func.func @add() attributes {grid = [4]} {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 2 : index
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    // Construct a memory view of A from a given address
    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    // Construct an access tile set from the memory view of A
    %A_access_tile = ktdp.construct_access_tile %A_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    // Construct a memory view of B from a given address
    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    // Construct an access tile set from the memory view of B
    %B_access_tile = ktdp.construct_access_tile %B_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    // Load data from the corresponding access tile
    %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>

    %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>

    // Perform add operation on the data tiles.
    %C_data_tile = tensor.empty() : tensor<2x64xf16>
    %result = linalg.add ins(%A_data_tile, %B_data_tile : tensor<2x64xf16>, tensor<2x64xf16>)
                outs(%C_data_tile: tensor<2x64xf16>) -> tensor<2x64xf16>

    // Construct a memory view of C from a given address
    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    // Construct an access tile set from the memory view of C
    %C_access_tile = ktdp.construct_access_tile %C_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    // Store data into the access tile.
    ktdp.store %result, %C_access_tile : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>

    return
  }

  func.func @mulf() attributes {grid = [4]} {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 2 : index
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %A_access_tile = ktdp.construct_access_tile %A_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %B_access_tile = ktdp.construct_access_tile %B_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>
    %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>

    %C_data_tile = tensor.empty() : tensor<2x64xf16>
    %result = linalg.mul ins(%A_data_tile, %B_data_tile : tensor<2x64xf16>, tensor<2x64xf16>)
                outs(%C_data_tile: tensor<2x64xf16>) -> tensor<2x64xf16>

    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %C_access_tile = ktdp.construct_access_tile %C_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    ktdp.store %result, %C_access_tile : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>

    return
  }

  func.func @max() attributes {grid = [4]} {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 2 : index
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %A_access_tile = ktdp.construct_access_tile %A_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %B_access_tile = ktdp.construct_access_tile %B_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>
    %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>

    %C_data_tile = tensor.empty() : tensor<2x64xf16>
    %result = linalg.max ins(%A_data_tile, %B_data_tile : tensor<2x64xf16>, tensor<2x64xf16>)
                outs(%C_data_tile: tensor<2x64xf16>) -> tensor<2x64xf16>

    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %C_access_tile = ktdp.construct_access_tile %C_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    ktdp.store %result, %C_access_tile : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>

    return
  }

  func.func @min() attributes {grid = [4]} {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 2 : index
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %A_access_tile = ktdp.construct_access_tile %A_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %B_access_tile = ktdp.construct_access_tile %B_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>
    %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>

    %C_data_tile = tensor.empty() : tensor<2x64xf16>
    %result = linalg.min ins(%A_data_tile, %B_data_tile : tensor<2x64xf16>, tensor<2x64xf16>)
                outs(%C_data_tile: tensor<2x64xf16>) -> tensor<2x64xf16>

    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %C_access_tile = ktdp.construct_access_tile %C_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    ktdp.store %result, %C_access_tile : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>

    return
  }

  func.func @absmax_maxnumf() attributes {grid = [4]} {
    %c0 = arith.constant 0 : index
    %tile_size = arith.constant 2 : index
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %A_access_tile = ktdp.construct_access_tile %A_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %B_access_tile = ktdp.construct_access_tile %B_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>
    %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<2x64xindex> -> tensor<2x64xf16>

    // ABSMax using maxnumf: element-wise maxnum(|A|, |B|)
    %C_data_tile = tensor.empty() : tensor<2x64xf16>
    %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>,
                         affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%A_data_tile, %B_data_tile : tensor<2x64xf16>, tensor<2x64xf16>)
        outs(%C_data_tile : tensor<2x64xf16>) {
    ^bb0(%a: f16, %b: f16, %c: f16):
        %abs_a = math.absf %a : f16
        %abs_b = math.absf %b : f16
        %mx = arith.maxnumf %abs_a, %abs_b : f16
        linalg.yield %mx : f16
    } -> tensor<2x64xf16>

    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %C_access_tile = ktdp.construct_access_tile %C_view[%start_row, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<2x64xindex>

    ktdp.store %result, %C_access_tile : tensor<2x64xf16>, !ktdp.access_tile<2x64xindex>

    return
  }
}
