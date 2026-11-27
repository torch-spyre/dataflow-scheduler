// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdf-to-ktdflowering)"  -allow-unregistered-dialect %s | FileCheck %s

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2, d4)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK: #[[$ATTR_6:.+]] = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>
// CHECK-LABEL:   ktdf_arch.device @sample_device import("../../../../Dialect/KTDFArch/sample_device.mlir")

// CHECK:   module {
// CHECK:     func.func private @local_schedule_0() attributes {grid = [2]} {
// CHECK:       %[[GET_UNIT_0:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnilu", type = "mnilu"} : index
// CHECK:       %[[GET_UNIT_1:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnilu", type = "mnilu"} : index
// CHECK:       %[[GET_UNIT_2:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-mnisu", type = "mnisu"} : index
// CHECK:       %[[GET_UNIT_3:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-mnisu", type = "mnisu"} : index
// CHECK:       %[[GET_UNIT_4:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu-CL0", type = "l1lu"} : index
// CHECK:       %[[GET_UNIT_5:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1lu-CL0", type = "l1lu"} : index
// CHECK:       %[[GET_UNIT_6:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-l1lu-CL1", type = "l1lu"} : index
// CHECK:       %[[GET_UNIT_7:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-l1lu-CL1", type = "l1lu"} : index
// CHECK:       %[[GET_UNIT_8:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu-CL0", type = "sfu"} : index
// CHECK:       %[[GET_UNIT_9:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-sfu-CL0", type = "sfu"} : index
// CHECK:       %[[GET_UNIT_10:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-sfu-CL1", type = "sfu"} : index
// CHECK:       %[[GET_UNIT_11:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-sfu-CL1", type = "sfu"} : index
// CHECK:       %[[GET_UNIT_12:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su-CL0", type = "l1su"} : index
// CHECK:       %[[GET_UNIT_13:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 0 : i32, name = "C1-l1su-CL0", type = "l1su"} : index
// CHECK:       %[[GET_UNIT_14:.*]] = dataflow.get_unit {core = 0 : i32, corelet = 1 : i32, name = "C0-l1su-CL1", type = "l1su"} : index
// CHECK:       %[[GET_UNIT_15:.*]] = dataflow.get_unit {core = 1 : i32, corelet = 1 : i32, name = "C1-l1su-CL1", type = "l1su"} : index
// CHECK:       %[[GET_COMPUTE_TILE_ID_0:.*]] = ktdp.get_compute_tile_id : index
// CHECK:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK:       %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK:       %[[DEF_IMMUTABLE_MAPPING_0:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_0]] -> %[[GET_UNIT_0]]], {{\[}}%[[CONSTANT_1]] -> %[[GET_UNIT_1]]]):index
// CHECK:       %[[QUERY_MAP_0:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_0]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK:       %[[CONSTANT_2:.*]] = arith.constant 0 : index
// CHECK:       %[[CONSTANT_3:.*]] = arith.constant 1 : index
// CHECK:       %[[DEF_IMMUTABLE_MAPPING_1:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_2]] -> %[[GET_UNIT_2]]], {{\[}}%[[CONSTANT_3]] -> %[[GET_UNIT_3]]]):index
// CHECK:       %[[QUERY_MAP_1:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_1]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK:       %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:       %[[CONSTANT_5:.*]] = arith.constant 1 : index
// CHECK:       %[[DEF_IMMUTABLE_MAPPING_2:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_4]] -> %[[GET_UNIT_4]]], {{\[}}%[[CONSTANT_5]] -> %[[GET_UNIT_5]]]):index
// CHECK:       %[[QUERY_MAP_2:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_2]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK:       %[[CONSTANT_6:.*]] = arith.constant 0 : index
// CHECK:       %[[CONSTANT_7:.*]] = arith.constant 1 : index
// CHECK:       %[[DEF_IMMUTABLE_MAPPING_3:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_6]] -> %[[GET_UNIT_6]]], {{\[}}%[[CONSTANT_7]] -> %[[GET_UNIT_7]]]):index
// CHECK:       %[[QUERY_MAP_3:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_3]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK:       %[[CONSTANT_8:.*]] = arith.constant 0 : index
// CHECK:       %[[CONSTANT_9:.*]] = arith.constant 1 : index
// CHECK:       %[[DEF_IMMUTABLE_MAPPING_4:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_8]] -> %[[GET_UNIT_8]]], {{\[}}%[[CONSTANT_9]] -> %[[GET_UNIT_9]]]):index
// CHECK:       %[[QUERY_MAP_4:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_4]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK:       %[[CONSTANT_10:.*]] = arith.constant 0 : index
// CHECK:       %[[CONSTANT_11:.*]] = arith.constant 1 : index
// CHECK:       %[[DEF_IMMUTABLE_MAPPING_5:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_10]] -> %[[GET_UNIT_10]]], {{\[}}%[[CONSTANT_11]] -> %[[GET_UNIT_11]]]):index
// CHECK:       %[[QUERY_MAP_5:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_5]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK:       %[[CONSTANT_12:.*]] = arith.constant 0 : index
// CHECK:       %[[CONSTANT_13:.*]] = arith.constant 1 : index
// CHECK:       %[[DEF_IMMUTABLE_MAPPING_6:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_12]] -> %[[GET_UNIT_12]]], {{\[}}%[[CONSTANT_13]] -> %[[GET_UNIT_13]]]):index
// CHECK:       %[[QUERY_MAP_6:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_6]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK:       %[[CONSTANT_14:.*]] = arith.constant 0 : index
// CHECK:       %[[CONSTANT_15:.*]] = arith.constant 1 : index
// CHECK:       %[[DEF_IMMUTABLE_MAPPING_7:.*]] = uniform.def_immutable_mapping({{\[}}%[[CONSTANT_14]] -> %[[GET_UNIT_14]]], {{\[}}%[[CONSTANT_15]] -> %[[GET_UNIT_15]]]):index
// CHECK:       %[[QUERY_MAP_7:.*]] = uniform.query_map(map:%[[DEF_IMMUTABLE_MAPPING_7]], key:%[[GET_COMPUTE_TILE_ID_0]]) : index
// CHECK:       %[[CONSTANT_16:.*]] = arith.constant 32 : index
// CHECK:       %[[CONSTANT_17:.*]] = arith.constant 6 : index
// CHECK:       %[[CONSTANT_18:.*]] = arith.constant 2 : index
// CHECK:       %[[CONSTANT_19:.*]] = arith.constant 0 : index
// CHECK:       %[[CONSTANT_20:.*]] = arith.constant 1 : index
// CHECK:       %[[CONSTANT_21:.*]] = arith.constant 64 : index
// CHECK:       %[[CONSTANT_22:.*]] = arith.constant 113216 : index
// CHECK:       %[[CONSTANT_23:.*]] = arith.constant 113152 : index
// CHECK:       %[[CONSTANT_24:.*]] = arith.constant 64000 : index
// CHECK:       %[[CONSTANT_25:.*]] = arith.constant 12 : index
// CHECK:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_24]], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
// CHECK:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
// CHECK:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
// CHECK:       %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_23]], sizes: [1, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_5]], memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
// CHECK:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<1x64xf16> to memref<1x64xf16, "DDR">
// CHECK:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<1x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1]>, "DDR">
// CHECK:       %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<1x64xf16, strided<[64, 1]>, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
// CHECK:       %[[CONSTRUCT_MEMORY_VIEW_2:.*]] = ktdp.construct_memory_view %[[CONSTANT_22]], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #[[$ATTR_6]], memory_space = #ktdp.memory_space<global>} : memref<12x1x1x64x64xf16>
// CHECK:       %[[MEMORY_SPACE_CAST_2:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_2]] : memref<12x1x1x64x64xf16> to memref<12x1x1x64x64xf16, "DDR">
// CHECK:       %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_2]] to offset: [0], sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] : memref<12x1x1x64x64xf16, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR">
// CHECK:       %[[CAST_2:.*]] = memref.cast %[[REINTERPRET_CAST_2]] : memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1]>, "DDR"> to memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK:       %[[CONSTANT_26:.*]] = arith.constant 0 : index
// CHECK:       %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_26]] : index to memref<1x64xf16, "L1">
// CHECK:       ktdf_lowering.execute_on %[[QUERY_MAP_0]] {
// CHECK:         ktdf_lowering.execute_on %[[QUERY_MAP_0]] {
// CHECK:           ktdf.data_transfer from %[[CAST_1]]{{\[}}%[[CONSTANT_19]], %[[CONSTANT_19]]] size [1, 64] to %[[UNREALIZED_CONVERSION_CAST_0]][0, 0] size [1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
// CHECK:         }
// CHECK:       }
// CHECK:       %[[CONSTANT_27:.*]] = arith.constant 128 : index
// CHECK:       %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_27]] : index to memref<2x2x1x1x1x64xf16, "L1">
// CHECK:       %[[CAST_3:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_1]] : memref<2x2x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x64xf16, "L1">
// CHECK:       %[[CONSTANT_28:.*]] = arith.constant 640 : index
// CHECK:       %[[UNREALIZED_CONVERSION_CAST_2:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_28]] : index to memref<2x2x1x1x1x64xf16, "L1">
// CHECK:       %[[CAST_4:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_2]] : memref<2x2x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x64xf16, "L1">
// CHECK:       %[[CONSTANT_29:.*]] = arith.constant 1152 : index
// CHECK:       %[[UNREALIZED_CONVERSION_CAST_3:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_29]] : index to memref<2x2x1x1x1x1x64xf16, "L1">
// CHECK:       %[[CAST_5:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_3]] : memref<2x2x1x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x1x64xf16, "L1">
// CHECK:       %[[CONSTANT_30:.*]] = arith.constant 1664 : index
// CHECK:       %[[UNREALIZED_CONVERSION_CAST_4:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_30]] : index to memref<2x2x1x1x1x1x64xf16, "L1">
// CHECK:       %[[CAST_6:.*]] = memref.cast %[[UNREALIZED_CONVERSION_CAST_4]] : memref<2x2x1x1x1x1x64xf16, "L1"> to memref<?x?x1x1x1x1x64xf16, "L1">
// CHECK:       %[[CONSTANT_31:.*]] = arith.constant 2240 : index
// CHECK:       %[[UNREALIZED_CONVERSION_CAST_5:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_31]] : index to memref<6x32x64xf16, "L1">
// CHECK:       scf.for %[[VAL_0:.*]] = %[[CONSTANT_19]] to %[[CONSTANT_17]] step %[[CONSTANT_20]] {
// CHECK:         %[[TILING_0:.*]] = ktdf.tiling.derive_size {{\[}}%[[VAL_0]] : %[[CONSTANT_18]]], total_size = %[[CONSTANT_25]] : index
// CHECK:         scf.for %[[VAL_1:.*]] = %[[CONSTANT_19]] to %[[CONSTANT_16]] step %[[CONSTANT_20]] {
// CHECK:           %[[TILING_1:.*]] = ktdf.tiling.derive_size {{\[}}%[[VAL_1]] : %[[CONSTANT_18]]], total_size = %[[CONSTANT_21]] : index
// CHECK:           %[[BUFFER_PHASE_0:.*]] = ktdf.buffer_phase(%[[VAL_0]], %[[VAL_1]]) {num_phases = 2 : i64} : index
// CHECK:           %[[SELECT_MEMREF_0:.*]] = ktdf.select_memref %[[BUFFER_PHASE_0]]{{\[}}%[[CAST_3]], %[[CAST_4]]] : memref<?x?x1x1x1x64xf16, "L1">
// CHECK:           %[[SELECT_MEMREF_1:.*]] = ktdf.select_memref %[[BUFFER_PHASE_0]]{{\[}}%[[CAST_5]], %[[CAST_6]]] : memref<?x?x1x1x1x1x64xf16, "L1">
// CHECK:           ktdf_lowering.execute_on %[[QUERY_MAP_0]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]], %[[QUERY_MAP_6]], %[[QUERY_MAP_7]], %[[QUERY_MAP_1]] {
// CHECK:             %[[CONSTANT_32:.*]] = arith.constant 2176 : index
// CHECK:             %[[UNREALIZED_CONVERSION_CAST_6:.*]] = builtin.unrealized_conversion_cast %[[CONSTANT_32]] : index to memref<1x64xf16, "L1">
// CHECK:             %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK:             %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK:             ktdf_lowering.execute_on %[[QUERY_MAP_0]] {
// CHECK:               scf.for %[[VAL_2:.*]] = %[[CONSTANT_19]] to %[[TILING_0]] step %[[CONSTANT_20]] {
// CHECK:                 scf.for %[[VAL_3:.*]] = %[[CONSTANT_19]] to %[[TILING_1]] step %[[CONSTANT_20]] {
// CHECK:                   %[[APPLY_0:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_0]], %[[VAL_2]])
// CHECK:                   %[[APPLY_1:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_1]], %[[VAL_3]])
// CHECK:                   ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[VAL_0]] * 2 + %[[VAL_2]], 0, %[[VAL_1]] * 2 + %[[VAL_3]], 0] size [1, 1, 1, 64] to %[[SELECT_MEMREF_0]]{{\[}}%[[VAL_2]], %[[VAL_3]], 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<?x?x1x1x1x64xf16, "L1">
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:             ktdf_lowering.signal %[[QUERY_MAP_0]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]]
// CHECK:             ktdf_lowering.execute_on %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]], %[[QUERY_MAP_6]], %[[QUERY_MAP_7]] {
// CHECK:               ktdf.parallel (%[[VAL_4:.*]], %[[VAL_5:.*]]) = (%[[CONSTANT_19]]) to (%[[TILING_0]]) step (%[[CONSTANT_20]]) distribute(num_instances = 2) {
// CHECK:                 scf.for %[[VAL_6:.*]] = %[[CONSTANT_19]] to %[[TILING_1]] step %[[CONSTANT_20]] {
// CHECK:                   ktdf_lowering.execute_on %[[QUERY_MAP_2]], %[[QUERY_MAP_3]], %[[QUERY_MAP_4]], %[[QUERY_MAP_5]], %[[QUERY_MAP_6]], %[[QUERY_MAP_7]] {
// CHECK:                     %[[FIFO_0:.*]]:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:                     %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK:                     %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK:                     %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK:                     ktdf_lowering.execute_on %[[QUERY_MAP_2]], %[[QUERY_MAP_3]] {
// CHECK:                       %[[CMPI_0:.*]] = arith.cmpi ne, %[[VAL_6]], %[[CONSTANT_19]] : index
// CHECK:                       %[[CMPI_1:.*]] = arith.cmpi ne, %[[VAL_0]], %[[CONSTANT_19]] : index
// CHECK:                       %[[ORI_0:.*]] = arith.ori %[[CMPI_0]], %[[CMPI_1]] : i1
// CHECK:                       scf.if %[[ORI_0]] {
// CHECK:                         ktdf_lowering.signal %[[QUERY_MAP_6]], %[[QUERY_MAP_7]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]]
// CHECK:                       }
// CHECK:                       ktdf.data_transfer from %[[SELECT_MEMREF_0]]{{\[}}%[[VAL_4]], %[[VAL_6]], 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] to %[[FIFO_0]]#0 size [64] : memref<?x?x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:                       scf.for %[[VAL_7:.*]] = %[[CONSTANT_19]] to %[[CONSTANT_21]] step %[[CONSTANT_20]] {
// CHECK:                         ktdf.data_transfer from %[[UNREALIZED_CONVERSION_CAST_5]]{{\[}}%[[VAL_0]], %[[VAL_6]], %[[VAL_7]]] size [1, 1, 64] to %[[FIFO_0]]#1 size [64] : memref<6x32x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:                       }
// CHECK:                     }
// CHECK:                     ktdf_lowering.execute_on %[[QUERY_MAP_4]], %[[QUERY_MAP_5]] {
// CHECK:                       %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[FIFO_0]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
// CHECK:                       %[[READ_FROM_FIFO_1:.*]] = ktdf.read_from_fifo %[[FIFO_0]]#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
// CHECK:                       %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x1x1x1x64xf16>
// CHECK:                       %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_2]], #[[$ATTR_3]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%[[READ_FROM_FIFO_0]], %[[READ_FROM_FIFO_1]] : tensor<1x1x1x64xf16>, tensor<1x64xf16>) outs(%[[EMPTY_0]] : tensor<1x1x1x1x64xf16>) {
// CHECK:                       ^bb0(%[[VAL_8:.*]]: f16, %[[VAL_9:.*]]: f16, %[[VAL_10:.*]]: f16):
// CHECK:                         %[[ADDF_0:.*]] = arith.addf %[[VAL_8]], %[[VAL_9]] : f16
// CHECK:                         linalg.yield %[[ADDF_0]] : f16
// CHECK:                       } -> tensor<1x1x1x1x64xf16>
// CHECK:                       ktdf.write_to_fifo %[[GENERIC_0]], %[[FIFO_1]] : tensor<1x1x1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK:                     }
// CHECK:                     ktdf_lowering.execute_on %[[QUERY_MAP_6]], %[[QUERY_MAP_7]] {
// CHECK:                       ktdf.data_transfer from %[[FIFO_1]] size [64] to %[[SELECT_MEMREF_1]]{{\[}}%[[VAL_4]], %[[VAL_6]], 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<?x?x1x1x1x1x64xf16, "L1">
// CHECK:                       ktdf.data_transfer from %[[FIFO_1]] size [64] to %[[UNREALIZED_CONVERSION_CAST_5]]{{\[}}%[[VAL_0]], %[[VAL_6]], %[[CONSTANT_19]]] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<6x32x64xf16, "L1">
// CHECK:                       %[[SUBI_0:.*]] = arith.subi %[[TILING_1]], %[[CONSTANT_20]] : index
// CHECK:                       %[[CMPI_2:.*]] = arith.cmpi ne, %[[VAL_6]], %[[SUBI_0]] : index
// CHECK:                       %[[SUBI_1:.*]] = arith.subi %[[CONSTANT_17]], %[[CONSTANT_20]] : index
// CHECK:                       %[[CMPI_3:.*]] = arith.cmpi ne, %[[VAL_0]], %[[SUBI_1]] : index
// CHECK:                       %[[ORI_1:.*]] = arith.ori %[[CMPI_2]], %[[CMPI_3]] : i1
// CHECK:                       scf.if %[[ORI_1]] {
// CHECK:                         ktdf_lowering.signal %[[QUERY_MAP_6]], %[[QUERY_MAP_7]], %[[QUERY_MAP_2]], %[[QUERY_MAP_3]]
// CHECK:                       }
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:                 ktdf.parallel_yield
// CHECK:               }
// CHECK:             }
// CHECK:             ktdf_lowering.signal %[[QUERY_MAP_6]], %[[QUERY_MAP_7]], %[[QUERY_MAP_1]]
// CHECK:             ktdf_lowering.execute_on %[[QUERY_MAP_1]] {
// CHECK:               scf.for %[[VAL_11:.*]] = %[[CONSTANT_19]] to %[[TILING_0]] step %[[CONSTANT_20]] {
// CHECK:                 scf.for %[[VAL_12:.*]] = %[[CONSTANT_19]] to %[[TILING_1]] step %[[CONSTANT_20]] {
// CHECK:                   %[[APPLY_2:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_0]], %[[VAL_11]])
// CHECK:                   %[[APPLY_3:.*]] = affine.apply #[[$ATTR_0]](%[[VAL_1]], %[[VAL_12]])
// CHECK:                   ktdf.data_transfer from %[[SELECT_MEMREF_1]]{{\[}}%[[VAL_11]], %[[VAL_12]], 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] to %[[CAST_2]]{{\[}}%[[VAL_0]] * 2 + %[[VAL_11]], 0, 0, %[[VAL_1]] * 2 + %[[VAL_12]], 0] size [1, 1, 1, 1, 64] : memref<?x?x1x1x1x1x64xf16, "L1">, memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:       return
// CHECK:     }
// CHECK:   }

// CHECK: }



// Test that backedge signal guards check conditions only on IVs of scf.for
// loops that are ancestors of the pipeline.
//
// The load stage (L1LU) has two transfers:
//   - first transfer: reads from %11 (no loop-carried dependency, no backedge)
//   - second transfer: wrapped in an inner scf.for %arg5 inside the stage body;
//     reads %scratchpad[%arg0, %arg4, %arg5] — loop-carried RAW with the store
//     stage.  %arg5 lives inside the load stage so has no dependency with the store
//     stage and thus must NOT appear in the guard condition.
// The store stage (L1SU) writes back to %scratchpad[%arg0, %arg4, 0].
//
// The inner pipeline is nested inside:
//   scf.for %arg0 -> scf.for %arg1 -> ktdf.parallel -> scf.for %arg4 -> pipeline
// %arg0 and %arg4 are both ancestor loops, so both IVs are eligible for guard conditions.
// %arg5 (inside the load stage,) is not an ancestor of the pipeline and
// must be excluded from the guard.
//
// Expected backedge guards (condition on %arg4 OR %arg0 IVs):
//   recv guard at top of load_stage body:
//     cmpi ne(%arg4, lb) OR cmpi ne(%arg0, lb) => scf.if { signal }
//   send guard at bottom of store_stage body:
//     cmpi ne(%arg4, ub-step) OR cmpi ne(%arg0, ub-step) => scf.if { signal }



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
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  module @local_schedule_0 {
    func.func private @local_schedule_0() attributes {grid = [2]} {
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
      %0 = ktdp.construct_memory_view %c64000, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR">
      %cast = memref.cast %reinterpret_cast : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1]>, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
      %1 = ktdp.construct_memory_view %c113152, sizes: [1, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<1x64xf16>
      %memspacecast_0 = memref.memory_space_cast %1 : memref<1x64xf16> to memref<1x64xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<1x64xf16, "DDR"> to memref<1x64xf16, strided<[64, 1]>, "DDR">
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<1x64xf16, strided<[64, 1]>, "DDR"> to memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">
      %2 = ktdp.construct_memory_view %c113216, sizes: [12, 1, 1, 64, 64], strides: [4096, 4096, 4096, 64, 1] {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>} : memref<12x1x1x64x64xf16>
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
      // Shared L1 scratchpad indexed by the outer loop IV (%arg0) and the
      // inner scf.for IV (%arg4).  The load stage reads from it; the store
      // stage writes back to it, creating a two-loop-dimensional backedge.
      %c2240 = arith.constant 2240 : index
      %scratchpad = builtin.unrealized_conversion_cast %c2240 : index to memref<6x32x64xf16, "L1">
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
                    // load_stage: first transfer from %11 has no loop-carried
                    // dependency (no backedge signal); second transfer is
                    // wrapped in an inner scf.for %arg5 and reads
                    // %scratchpad[%arg0, %arg4, %arg5].  %arg5 is inside the
                    // stage (not an ancestor of the pipeline) so it must not
                    // appear in the backedge guard condition.
                    ktdf.stage depends_in(none) depends_out(%14#3) {
                      ktdf.data_transfer from %11[%arg2, %arg4, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 64] to %14#0 size [64] : memref<?x?x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                      scf.for %arg5 = %c0 to %c64 step %c1 {
                        ktdf.data_transfer from %scratchpad[%arg0, %arg4, %arg5] size [1, 1, 64] to %14#1 size [64] : memref<6x32x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                      }
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
                    // store_stage: first transfer to %12 has no loop-carried
                    // dependency (no backedge signal); second transfer writes back
                    // to %scratchpad[%arg0, %arg4, 0], closing the backedge.
                    ktdf.stage depends_in(%14#4) depends_out(none) {
                      ktdf.data_transfer from %14#2 size [64] to %12[%arg2, %arg4, 0, 0, 0, 0, 0] size [1, 1, 1, 1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<?x?x1x1x1x1x64xf16, "L1">
                      ktdf.data_transfer from %14#2 size [64] to %scratchpad[%arg0, %arg4, %c0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<6x32x64xf16, "L1">
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
