// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdf-to-ktdflowering)"  -allow-unregistered-dialect %s | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d4)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_6:.+]] = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>
// CHECK:   module {
// CHECK:     module {
// CHECK:     func.func @Add_1414() attributes {grid = [2]} {
// CHECK-NEXT:       call @"local-schedule-0"() : () -> ()
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:     func.func private @"local-schedule-0"()
// CHECK-NEXT:   }

// CHECK:     module {
// CHECK:     func.func private @"local-schedule-0"() attributes {grid = [2]} {
// CHECK-NEXT:       %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnilu", type = "mnilu"} : index
// CHECK-NEXT:       %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// CHECK-NEXT:       %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// CHECK-NEXT:       %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnisu", type = "mnisu"} : index
// CHECK-NEXT:       %[[GET_UNIT_4:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu-CL0", type = "l1lu"} : index
// CHECK-NEXT:       %[[GET_UNIT_5:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1lu-CL0", type = "l1lu"} : index
// CHECK-NEXT:       %[[GET_UNIT_6:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-l1lu-CL1", type = "l1lu"} : index
// CHECK-NEXT:       %[[GET_UNIT_7:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-l1lu-CL1", type = "l1lu"} : index
// CHECK-NEXT:       %[[GET_UNIT_8:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu-CL0", type = "sfu"} : index
// CHECK-NEXT:       %[[GET_UNIT_9:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-sfu-CL0", type = "sfu"} : index
// CHECK-NEXT:       %[[GET_UNIT_10:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfu-CL1", type = "sfu"} : index
// CHECK-NEXT:       %[[GET_UNIT_11:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-sfu-CL1", type = "sfu"} : index
// CHECK-NEXT:       %[[GET_UNIT_12:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su-CL0", type = "l1su"} : index
// CHECK-NEXT:       %[[GET_UNIT_13:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1su-CL0", type = "l1su"} : index
// CHECK-NEXT:       %[[GET_UNIT_14:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-l1su-CL1", type = "l1su"} : index
// CHECK-NEXT:       %[[GET_UNIT_15:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-l1su-CL1", type = "l1su"} : index
// CHECK-NEXT:       %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_0]] -> %[[GET_UNIT_0]]], {{\[}}%[[CONSTANT_1]] -> %[[GET_UNIT_1]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_2]] -> %[[GET_UNIT_2]]], {{\[}}%[[CONSTANT_3]] -> %[[GET_UNIT_3]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_4]] -> %[[GET_UNIT_4]]], {{\[}}%[[CONSTANT_5]] -> %[[GET_UNIT_5]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_2:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_2]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_7:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_3:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_6]] -> %[[GET_UNIT_6]]], {{\[}}%[[CONSTANT_7]] -> %[[GET_UNIT_7]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_3:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_3]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_8:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_9:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_4:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_8]] -> %[[GET_UNIT_8]]], {{\[}}%[[CONSTANT_9]] -> %[[GET_UNIT_9]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_4:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_4]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_10:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_11:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_5:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_10]] -> %[[GET_UNIT_10]]], {{\[}}%[[CONSTANT_11]] -> %[[GET_UNIT_11]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_5:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_5]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_12:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_13:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_6:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_12]] -> %[[GET_UNIT_12]]], {{\[}}%[[CONSTANT_13]] -> %[[GET_UNIT_13]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_6:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_6]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_14:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_15:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[DEF_IMMUTABLE_MAPPING_7:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_14]] -> %[[GET_UNIT_14]]], {{\[}}%[[CONSTANT_15]] -> %[[GET_UNIT_15]]]):index
// CHECK-NEXT:       %[[QUERY_MAP_7:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_7]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK-NEXT:       %[[CONSTANT_16:.*]] = arith.constant 32 : index
// CHECK-NEXT:       %[[CONSTANT_17:.*]] = arith.constant 6 : index
// CHECK-NEXT:       %[[CONSTANT_18:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[CONSTANT_19:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_20:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_21:.*]] = arith.constant 64 : index
// CHECK-NEXT:       %[[CONSTANT_22:.*]] = arith.constant 113216 : index
// CHECK-NEXT:       %[[CONSTANT_23:.*]] = arith.constant 113152 : index
// CHECK-NEXT:       %[[CONSTANT_24:.*]] = arith.constant 64000 : index
// CHECK-NEXT:       %[[CONSTANT_25:.*]] = arith.constant 12 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_24]], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<12x1x64x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_23]], sizes: [1, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_5]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<1x64xf16> to memref<1x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<1x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<1x64xf16, strided<[64, 1]>, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_22]], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #[[$ATTR_6]], memory_space = #ktdp.spyre_memory_space<HBM>} : memref<12x1x1x64x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_2:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_2]] : memref<12x1x1x64x64xf16> to memref<12x1x1x64x64xf16, "DDR">
// CHECK-NEXT:       %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_2]] to offset: [0], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] : memref<12x1x1x64x64xf16, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR">
// CHECK-NEXT:       %[[CAST_2:.*]] = memref.cast %[[REINTERPRET_CAST_2]] : memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:       %[[CONSTANT_26:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_26]] : index to memref<1x64xf16, "L1">
// CHECK-NEXT:       ktdf_lowering.execute_on %[[QUERY_MAP_0]] {
// CHECK-NEXT:         ktdf_lowering.execute_on %[[QUERY_MAP_0]] {
// CHECK-NEXT:           ktdf.data_transfer from %[[CAST_1]]{{\[}}%[[CONSTANT_19]], %[[CONSTANT_19]]] size [1, 64] to %[[UNREALIZED_CONVERSION_CAST_0]][0, 0] size [1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       %[[CONSTANT_27:.*]] = arith.constant 128 : index
// CHECK-NEXT:       %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_27]] : index to memref<2x2x1x1x1x64xf16, "L1">
// CHECK-NEXT:       %[[CAST_3:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_1]] : memref<2x2x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x64xf16, "L1">
// CHECK-NEXT:       %[[CONSTANT_28:.*]] = arith.constant 640 : index
// CHECK-NEXT:       %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_28]] : index to memref<2x2x1x1x1x64xf16, "L1">
// CHECK-NEXT:       %[[CAST_4:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_2]] : memref<2x2x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x64xf16, "L1">
// CHECK-NEXT:       %[[CONSTANT_29:.*]] = arith.constant 1152 : index
// CHECK-NEXT:       %[[UNREALIZED_CONVERSION_CAST_3:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_29]] : index to memref<2x2x1x1x1x1x64xf16, "L1">
// CHECK-NEXT:       %[[CAST_5:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_3]] : memref<2x2x1x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x1x64xf16, "L1">
// CHECK-NEXT:       %[[CONSTANT_30:.*]] = arith.constant 1664 : index
// CHECK-NEXT:       %[[UNREALIZED_CONVERSION_CAST_4:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_30]] : index to memref<2x2x1x1x1x1x64xf16, "L1">
// CHECK-NEXT:       %[[CAST_6:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_4]] : memref<2x2x1x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x1x64xf16, "L1">
// CHECK-NEXT:       scf.for %[[VAL_0:.*]] = %[[CONSTANT_19]] to %[[CONSTANT_17]] step %[[CONSTANT_20]] {
// CHECK-NEXT:         %[[TILING_0:.*]] = ktdf.tiling.derive_size {{\[}}%[[VAL_0]] : %[[CONSTANT_18]]], total_size = %[[CONSTANT_25]] : index
// CHECK-NEXT:         scf.for %[[VAL_1:.*]] = %[[CONSTANT_19]] to %[[CONSTANT_16]] step %[[CONSTANT_20]] {
// CHECK-NEXT:           %[[TILING_1:.*]] = ktdf.tiling.derive_size {{\[}}%[[VAL_1]] : %[[CONSTANT_18]]], total_size = %[[CONSTANT_21]] : index
// CHECK-NEXT:           %[[BUFFER_PHASE_0:.*]] = ktdf.buffer_phase(%[[VAL_0]], %[[VAL_1]]) {num_phases = 2 : i64} : index
// CHECK-NEXT:           %[[SELECT_MEMREF_0:.*]] = ktdf.select_memref %[[BUFFER_PHASE_0]]{{\[}}%[[CAST_3]], %[[CAST_4]]] : memref<?x?x1x1x1x64xf16, "L1">
// CHECK-NEXT:           %[[SELECT_MEMREF_1:.*]] = ktdf.select_memref %[[BUFFER_PHASE_0]]{{\[}}%[[CAST_5]], %[[CAST_6]]] : memref<?x?x1x1x1x1x64xf16, "L1">
// CHECK-NEXT:           ktdf_lowering.execute_on %[[QUERY_MAP_0]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]], %[[QUERY_MAP_6]], %[[QUERY_MAP_7]], %[[QUERY_MAP_1]] {
// CHECK-NEXT:             %[[CONSTANT_31:.*]] = arith.constant 2176 : index
// CHECK-NEXT:             %[[UNREALIZED_CONVERSION_CAST_5:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_31]] : index to memref<1x64xf16, "L1">
// CHECK-NEXT:             %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf_lowering.execute_on %[[QUERY_MAP_0]] {
// CHECK-NEXT:               scf.for %[[VAL_2:.*]] = %[[CONSTANT_19]] to %[[TILING_0]] step %[[CONSTANT_20]] {
// CHECK-NEXT:                 scf.for %[[VAL_3:.*]] = %[[CONSTANT_19]] to %[[TILING_1]] step %[[CONSTANT_20]] {
// CHECK-NEXT:                   %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_0]], %[[VAL_2]])
// CHECK-NEXT:                   %[[APPLY_1:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_1]], %[[VAL_3]])
// CHECK-NEXT:                   ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[VAL_0]] * 2 + %[[VAL_2]], 0, %[[VAL_1]] * 2 + %[[VAL_3]], 0] size [1, 1, 1, 64] to %[[SELECT_MEMREF_0]]{{\[}}%[[VAL_2]], %[[VAL_3]], 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<?x?x1x1x1x64xf16, "L1">
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf_lowering.signal %[[QUERY_MAP_0]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]]
// CHECK-NEXT:             ktdf_lowering.execute_on %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]], %[[QUERY_MAP_6]], %[[QUERY_MAP_7]] {
// CHECK-NEXT:               ktdf.parallel (%[[VAL_4:.*]], %[[VAL_5:.*]]) = (%[[CONSTANT_19]]) to (%[[TILING_0]]) step (%[[CONSTANT_20]]) distribute(num_instances = 2) {
// CHECK-NEXT:                 scf.for %[[VAL_6:.*]] = %[[CONSTANT_19]] to %[[TILING_1]] step %[[CONSTANT_20]] {
// CHECK-NEXT:                   ktdf_lowering.execute_on %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]], %[[QUERY_MAP_6]], %[[QUERY_MAP_7]] {
// CHECK-NEXT:                     %[[FIFO_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                     %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                     %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     ktdf_lowering.execute_on %[[QUERY_MAP_2]], %[[QUERY_MAP_3]] {
// CHECK-NEXT:                       ktdf.data_transfer from %[[SELECT_MEMREF_0]]{{\[}}%[[VAL_4]], %[[VAL_6]], 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] to %[[FIFO_0]]#0 size [64] : memref<?x?x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                       ktdf.data_transfer from %[[UNREALIZED_CONVERSION_CAST_0]][0, 0] size [1, 64] to %[[FIFO_0]]#1 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                     }
// CHECK-NEXT:                     ktdf_lowering.execute_on %[[QUERY_MAP_4]], %[[QUERY_MAP_5]] {
// CHECK-NEXT:                       %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[FIFO_0]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
// CHECK-NEXT:                       %[[READ_FROM_FIFO_1:.*]] = ktdf.read_from_fifo %[[FIFO_0]]#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
// CHECK-NEXT:                       %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x1x1x1x64xf16>
// CHECK-NEXT:                       %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]], #[[$ATTR_3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[READ_FROM_FIFO_0]], %[[READ_FROM_FIFO_1]] : tensor<1x1x1x64xf16>, tensor<1x64xf16>) outs(%[[EMPTY_0]] : tensor<1x1x1x1x64xf16>) {
// CHECK-NEXT:                       ^bb0(%[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16):
// CHECK-NEXT:                         %[[ADDF_0:.*]] = arith.addf %[[VAL_7]], %[[VAL_8]] : f16
// CHECK-NEXT:                         linalg.yield %[[ADDF_0]] : f16
// CHECK-NEXT:                       } -> tensor<1x1x1x1x64xf16>
// CHECK-NEXT:                       ktdf.write_to_fifo %[[GENERIC_0]], %[[FIFO_1]] : tensor<1x1x1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                     }
// CHECK-NEXT:                     ktdf_lowering.execute_on %[[QUERY_MAP_6]], %[[QUERY_MAP_7]] {
// CHECK-NEXT:                       ktdf.data_transfer from %[[FIFO_1]] size [64] to %[[SELECT_MEMREF_1]]{{\[}}%[[VAL_4]], %[[VAL_6]], 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<?x?x1x1x1x1x64xf16, "L1">
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:                 ktdf.parallel_yield
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:             ktdf_lowering.signal %[[QUERY_MAP_6]], %[[QUERY_MAP_7]], %[[QUERY_MAP_1]]
// CHECK-NEXT:             ktdf_lowering.execute_on %[[QUERY_MAP_1]] {
// CHECK-NEXT:               scf.for %[[VAL_10:.*]] = %[[CONSTANT_19]] to %[[TILING_0]] step %[[CONSTANT_20]] {
// CHECK-NEXT:                 scf.for %[[VAL_11:.*]] = %[[CONSTANT_19]] to %[[TILING_1]] step %[[CONSTANT_20]] {
// CHECK-NEXT:                   %[[APPLY_2:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_0]], %[[VAL_10]])
// CHECK-NEXT:                   %[[APPLY_3:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_1]], %[[VAL_11]])
// CHECK-NEXT:                   ktdf.data_transfer from %[[SELECT_MEMREF_1]]{{\[}}%[[VAL_10]], %[[VAL_11]], 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] to %[[CAST_2]]{{\[}}%[[VAL_0]] * 2 + %[[VAL_10]], 0, 0, %[[VAL_1]] * 2 + %[[VAL_11]], 0] size [1, 1, 1, 1, 64] : memref<?x?x1x1x1x1x64xf16, "L1">, memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }



// Example with nested pipelines and a parallel region in the innermost pipeline.
// Any component types occuring in the parallel region should be treated as
// parallel components in ancestor execute_on operations (outside the parallel region).

#map = affine_map<(d0, d1) -> (d0 * 2 + d1)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @Add_1414() attributes {grid = [2]} {
      call @"local-schedule-0"() : () -> ()
      return
    }
    func.func private @"local-schedule-0"()
  }
  module {
    func.func private @"local-schedule-0"() attributes {grid = [2]} {
      %c32 = arith.constant 32 : index
      %c6 = arith.constant 6 : index
      %c2 = arith.constant 2 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c113216 = arith.constant 113216 : index
      %c113152 = arith.constant 113152 : index
      %c64000 = arith.constant 64000 : index
      %c12 = arith.constant 12 : index
      %0 = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<12x1x64x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
      %1 = ktdp.construct_memory_view %c113152, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<1x64xf16>
      %memspacecast_0 = memref.memory_space_cast %1 : memref<1x64xf16> to memref<1x64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<1x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<1x64xf16, strided<[64, 1]>, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
      %2 = ktdp.construct_memory_view %c113216, sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.spyre_memory_space<HBM>} : memref<12x1x1x64x64xf16>
      %memspacecast_3 = memref.memory_space_cast %2 : memref<12x1x1x64x64xf16> to memref<12x1x1x64x64xf16, "DDR">
      %reinterpret_cast_4 = memref.reinterpret_cast %memspacecast_3 to offset: [0], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] : memref<12x1x1x64x64xf16, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR">
      %cast_5 = memref.cast %reinterpret_cast_4 : memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
      %c0_6 = arith.constant 0 : index
      %3 = builtin.unrealized_conversion_cast %c0_6 : index to memref<1x64xf16, "L1">
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          ktdf.data_transfer from %cast_2[%c0, %c0] size [1, 64] to %3[0, 0] size [1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
        } {applicable_units = ["MNILU"]}
      }
      %c128 = arith.constant 128 : index
      %4 = builtin.unrealized_conversion_cast %c128 : index to memref<2x2x1x1x1x64xf16, "L1">
      %cast_7 = memref.cast %4 : memref<2x2x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x64xf16, "L1">
      %c640 = arith.constant 640 : index
      %5 = builtin.unrealized_conversion_cast %c640 : index to memref<2x2x1x1x1x64xf16, "L1">
      %cast_8 = memref.cast %5 : memref<2x2x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x64xf16, "L1">
      %c1152 = arith.constant 1152 : index
      %6 = builtin.unrealized_conversion_cast %c1152 : index to memref<2x2x1x1x1x1x64xf16, "L1">
      %cast_9 = memref.cast %6 : memref<2x2x1x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x1x64xf16, "L1">
      %c1664 = arith.constant 1664 : index
      %7 = builtin.unrealized_conversion_cast %c1664 : index to memref<2x2x1x1x1x1x64xf16, "L1">
      %cast_10 = memref.cast %7 : memref<2x2x1x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x1x64xf16, "L1">
      scf.for %arg0 = %c0 to %c6 step %c1 {
        %8 = ktdf.tiling.derive_size [%arg0 : %c2], total_size = %c12 : index
        scf.for %arg1 = %c0 to %c32 step %c1 {
          %9 = ktdf.tiling.derive_size [%arg1 : %c2], total_size = %c64 : index
          %10 = ktdf.buffer_phase(%arg0, %arg1) {num_phases = 2 : i64} : index
          %11 = ktdf.select_memref %10[%cast_7, %cast_8] : memref<?x?x1x1x1x64xf16, "L1">
          %12 = ktdf.select_memref %10[%cast_9, %cast_10] : memref<?x?x1x1x1x1x64xf16, "L1">
          ktdf.pipeline modulo(size : 2) {
            %13:3 = ktdf.private -> (memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token) {
              %c2176 = arith.constant 2176 : index
              %14 = builtin.unrealized_conversion_cast %c2176 : index to memref<1x64xf16, "L1">
              %15 = ktdf.create_token : !ktdf.token
              %16 = ktdf.create_token : !ktdf.token
              ktdf.private_yield %14, %15, %16 : memref<1x64xf16, "L1">, !ktdf.token, !ktdf.token
            }
            ktdf.stage depends_in(none) depends_out(%13#1) {
              scf.for %arg2 = %c0 to %8 step %c1 {
                scf.for %arg3 = %c0 to %9 step %c1 {
                  %14 = affine.apply #map(%arg0, %arg2)
                  %15 = affine.apply #map(%arg1, %arg3)
                  ktdf.data_transfer from %cast[%arg0 * 2 + %arg2, 0, %arg1 * 2 + %arg3, 0] size [1, 1, 1, 64] to %11[%arg2, %arg3, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<?x?x1x1x1x64xf16, "L1">
                } {loop_type = #ktdf.loop_type<parallel_loop>}
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            } {applicable_units = ["MNILU"]}
            ktdf.stage depends_in(%13#1) depends_out(%13#2) {
              ktdf.parallel (%arg2, %arg3) = (%c0) to (%8) step (%c1) distribute(num_instances = 2) {
                scf.for %arg4 = %c0 to %9 step %c1 {
                  ktdf.pipeline {
                    %14:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                      %15:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                      %16 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                      %17 = ktdf.create_token : !ktdf.token
                      %18 = ktdf.create_token : !ktdf.token
                      ktdf.private_yield %15#0, %15#1, %16, %17, %18 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
                    }
                    ktdf.stage depends_in(none) depends_out(%14#3) {
                      ktdf.data_transfer from %11[%arg2, %arg4, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] to %14#0 size [64] : memref<?x?x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                      ktdf.data_transfer from %3[0, 0] size [1, 64] to %14#1 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                    } {applicable_units = ["L1LU"]}
                    ktdf.stage depends_in(%14#3) depends_out(%14#4) {
                      %15 = ktdf.read_from_fifo %14#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
                      %16 = ktdf.read_from_fifo %14#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
                      %17 = tensor.empty() : tensor<1x1x1x1x64xf16>
                      %18 = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%15, %16 : tensor<1x1x1x64xf16>, tensor<1x64xf16>) outs(%17 : tensor<1x1x1x1x64xf16>) {
                      ^bb0(%in: f16, %in_11: f16, %out: f16):
                        %19 = arith.addf %in, %in_11 : f16
                        linalg.yield %19 : f16
                      } -> tensor<1x1x1x1x64xf16>
                      ktdf.write_to_fifo %18, %14#2 : tensor<1x1x1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
                    } {applicable_units = ["SFU"]}
                    ktdf.stage depends_in(%14#4) depends_out(none) {
                      ktdf.data_transfer from %14#2 size [64] to %12[%arg2, %arg4, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<?x?x1x1x1x1x64xf16, "L1">
                    } {applicable_units = ["L1SU"]}
                  }
                } {loop_type = #ktdf.loop_type<parallel_loop>}
                ktdf.parallel_yield
              }
            } {applicable_units = ["L1LU", "SFU", "L1SU"]}
            ktdf.stage depends_in(%13#2) depends_out(none) {
              scf.for %arg2 = %c0 to %8 step %c1 {
                scf.for %arg3 = %c0 to %9 step %c1 {
                  %14 = affine.apply #map(%arg0, %arg2)
                  %15 = affine.apply #map(%arg1, %arg3)
                  ktdf.data_transfer from %12[%arg2, %arg3, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] to %cast_5[%arg0 * 2 + %arg2, 0, 0, %arg1 * 2 + %arg3, 0] size [1, 1, 1, 1, 64] : memref<?x?x1x1x1x1x64xf16, "L1">, memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
                } {loop_type = #ktdf.loop_type<parallel_loop>}
              } {loop_type = #ktdf.loop_type<parallel_loop>}
            } {applicable_units = ["MNISU"]}
          }
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
      return
    }
  }
}
