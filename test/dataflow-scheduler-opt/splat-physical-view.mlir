// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s --ktir-map-and-tile --ktir-bufferize | FileCheck %s --check-prefix=BUFFER
// RUN: dataflow-scheduler-opt %s -kEmitDFIR --split-dfir-output-dir=%t | FileCheck %s --check-prefix=DFIR
// The logical access selects one lane per row. The bufferized view must retain
// the other lanes so the architecture can load its minimum 32-bit granule.
// BUFFER-LABEL: func.func @fp32_splat_view_repro
// BUFFER: memref.reinterpret_cast {{.*}}sizes: [4, 32], strides: [32, 1]
// BUFFER: ktdp_lowering.load {{.*}} [1, 32] [1, 0]
// BUFFER-SAME: memref<4x32xf32, strided<[32, 1], offset: ?>, "L1">
// BUFFER-LABEL: func.func @fp32_splat_view_two_cores
// BUFFER: %[[CORE:[0-9]+]] = ktdp.get_compute_tile_id
// BUFFER: %[[ROW:[0-9]+]] = arith.muli %[[CORE]], %c4 : index
// BUFFER: %[[OFFSET:[0-9]+]] = arith.muli %[[ROW]], %c32 : index
// BUFFER: memref.reinterpret_cast {{.*}}offset: [%[[OFFSET]]], sizes: [4, 32], strides: [32, 1]
// BUFFER: ktdp_lowering.load {{.*}} [1, 32] [1, 0]
// BUFFER-SAME: memref<4x32xf32, strided<[32, 1], offset: ?>, "L1">
// BUFFER-LABEL: func.func @i32_splat_view_repro
// BUFFER: memref.reinterpret_cast {{.*}}sizes: [4, 32], strides: [32, 1]
// BUFFER: ktdp_lowering.load {{.*}} [1, 32] [1, 0]
// BUFFER-SAME: memref<4x32xi32, strided<[32, 1], offset: ?>, "L1">
// BUFFER-LABEL: func.func @i32_splat_view_two_cores
// BUFFER: %[[I32_CORE:[0-9]+]] = ktdp.get_compute_tile_id
// BUFFER: %[[I32_ROW:[0-9]+]] = arith.muli %[[I32_CORE]], %c4 : index
// BUFFER: %[[I32_OFFSET:[0-9]+]] = arith.muli %[[I32_ROW]], %c32 : index
// BUFFER: memref.reinterpret_cast {{.*}}offset: [%[[I32_OFFSET]]], sizes: [4, 32], strides: [32, 1]
// BUFFER: ktdp_lowering.load {{.*}} [1, 32] [1, 0]
// BUFFER-SAME: memref<4x32xi32, strided<[32, 1], offset: ?>, "L1">
// DFIR-LABEL: func.func @local_schedule_0_body
// DFIR: agen.vector_load {{.*}} : memref<8x32xf32>, vector<2xf32>
// DFIR: vectorchain.shuffle
// The two-core producer must be a compute schedule, not a bare data transfer.
// DFIR-LABEL: func.func @local_schedule_1_body() attributes {grid = [2]}
// DFIR: dataflow.program_unit
// DFIR: agen.composite_load_and_store
// DFIR: vectorchain.binary
// DFIR: agen.vector_store
// DFIR-LABEL: func.func @local_schedule_2(%arg0: index)
// DFIR: arith.addi %arg0, %c512 : index
// DFIR-LABEL: func.func @local_schedule_2_body() attributes {grid = [2]}
// DFIR: dataflow.get_unit {core = 0{{.*}}name = "C0-l1"
// DFIR: dataflow.get_unit {core = 1{{.*}}name = "C1-l1"
// DFIR: uniform.def_immutable_mapping([{{.*}} -> %c0], [{{.*}} -> %c128]
// DFIR: agen.vector_load {{.*}} : memref<8x32xf32>, vector<2xf32>
// DFIR: vectorchain.shuffle

// DFIR-LABEL: func.func @local_schedule_3_body
// DFIR: agen.vector_load {{.*}} : memref<8x32xi32>, vector<2xi32>
// DFIR: vectorchain.shuffle
// DFIR: %[[RECV_ONE:.+]] = dataflow.receive {{.*}} : vector<32xi32>
// DFIR-NEXT: %[[SPLAT_ONE:.+]] = vectorchain.shuffle input(%[[RECV_ONE]])
// DFIR-SAME: indices = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
// DFIR-SAME: repetition = 8
// The two-core producer must be a compute schedule, not a bare data transfer.
// DFIR-LABEL: func.func @local_schedule_4_body() attributes {grid = [2]}
// DFIR: dataflow.program_unit
// DFIR: agen.composite_load_and_store
// DFIR: vectorchain.binary
// DFIR: agen.vector_store
// DFIR-LABEL: func.func @local_schedule_5(%arg0: index)
// DFIR: arith.addi %arg0, %c512 : index
// DFIR-LABEL: func.func @local_schedule_5_body() attributes {grid = [2]}
// DFIR: dataflow.get_unit {core = 0{{.*}}name = "C0-l1"
// DFIR: dataflow.get_unit {core = 1{{.*}}name = "C1-l1"
// DFIR: uniform.def_immutable_mapping([{{.*}} -> %c0], [{{.*}} -> %c128]
// DFIR: agen.vector_load {{.*}} : memref<8x32xi32>, vector<2xi32>
// DFIR: vectorchain.shuffle
// DFIR: %[[RECV_TWO:.+]] = dataflow.receive {{.*}} : vector<32xi32>
// DFIR-NEXT: %[[SPLAT_TWO:.+]] = vectorchain.shuffle input(%[[RECV_TWO]])
// DFIR-SAME: indices = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
// DFIR-SAME: repetition = 8

// One core reads lane 0 of rows 4..7 in eight ct_local sticks.
// The caller supplies initialized ct_local storage at element offset 0.
// Expected result: out[row, lane] = 2 * local[row, 0], for rows 4..7.
#memory = affine_set<(r, l) : (r >= 0, 7 - r >= 0, l >= 0, 31 - l >= 0)>
#scalar = affine_set<(r, l) : (r >= 0, 3 - r >= 0, l == 0)>
#stick = affine_set<(r, l) : (r >= 0, 3 - r >= 0, l >= 0, 31 - l >= 0)>
#identity = affine_map<(r, l) -> (r, l)>
#broadcast = affine_map<(r, l) -> (r, 0)>

module {
  ktdf_arch.device @sample_device attributes {
    mem_space_mapping = #ktdf_arch.map<
      #ktdp.memory_space<global> = "DDR",
      #ktdp.memory_space<ct_local> = "L1">
  } import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @fp32_splat_view_repro(%out_addr: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %row = arith.constant 4 : index
    %src = ktdp.construct_memory_view %zero,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<ct_local>
    } : memref<8x32xf32>
    %src_tile = ktdp.construct_access_tile %src[%row, %zero] {
      access_tile_set = #scalar, access_tile_order = #identity
    } : memref<8x32xf32> -> !ktdp.access_tile<4x1xindex>
    %value = ktdp.load %src_tile : <4x1xindex> -> tensor<4x1xf32>
    %init = tensor.empty() : tensor<4x32xf32>
    %result = linalg.generic {
      indexing_maps = [#broadcast, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%value : tensor<4x1xf32>) outs(%init : tensor<4x32xf32>) {
    ^bb0(%x: f32, %unused: f32):
      %y = arith.addf %x, %x : f32
      linalg.yield %y : f32
    } -> tensor<4x32xf32>
    %dst = ktdp.construct_memory_view %out_addr,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<global>
    } : memref<8x32xf32>
    %dst_tile = ktdp.construct_access_tile %dst[%row, %zero] {
      access_tile_set = #stick, access_tile_order = #identity
    } : memref<8x32xf32> -> !ktdp.access_tile<4x32xindex>
    ktdp.store %result, %dst_tile : tensor<4x32xf32>, <4x32xindex>
    return
  }

  // Core 0 handles rows 0..3; core 1 handles rows 4..7. The first compute
  // group writes 2 * input to ct_local, then the second broadcasts lane 0.
  // Each output lane contains 4 * input[row, 0].
  func.func @fp32_splat_view_two_cores(%in_addr: index, %out_addr: index) attributes {grid = [2]} {
    %zero = arith.constant 0 : index
    %four = arith.constant 4 : index
    %core = ktdp.get_compute_tile_id : index
    %row = arith.muli %core, %four : index
    %src = ktdp.construct_memory_view %zero,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<ct_local>
    } : memref<8x32xf32>
    %input = ktdp.construct_memory_view %in_addr,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<global>
    } : memref<8x32xf32>
    %input_tile = ktdp.construct_access_tile %input[%row, %zero] {
      access_tile_set = #stick, access_tile_order = #identity
    } : memref<8x32xf32> -> !ktdp.access_tile<4x32xindex>
    %input_values = ktdp.load %input_tile : <4x32xindex> -> tensor<4x32xf32>
    %stage_init = tensor.empty() : tensor<4x32xf32>
    %stage_values = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%input_values : tensor<4x32xf32>) outs(%stage_init : tensor<4x32xf32>) {
    ^bb0(%x: f32, %unused_stage: f32):
      %twice = arith.addf %x, %x : f32
      linalg.yield %twice : f32
    } -> tensor<4x32xf32>
    %local_tile = ktdp.construct_access_tile %src[%row, %zero] {
      access_tile_set = #stick, access_tile_order = #identity
    } : memref<8x32xf32> -> !ktdp.access_tile<4x32xindex>
    ktdp.store %stage_values, %local_tile : tensor<4x32xf32>, <4x32xindex>
    %src_tile = ktdp.construct_access_tile %src[%row, %zero] {
      access_tile_set = #scalar, access_tile_order = #identity
    } : memref<8x32xf32> -> !ktdp.access_tile<4x1xindex>
    %value = ktdp.load %src_tile : <4x1xindex> -> tensor<4x1xf32>
    %init = tensor.empty() : tensor<4x32xf32>
    %result = linalg.generic {
      indexing_maps = [#broadcast, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%value : tensor<4x1xf32>) outs(%init : tensor<4x32xf32>) {
    ^bb0(%x: f32, %unused: f32):
      %y = arith.addf %x, %x : f32
      linalg.yield %y : f32
    } -> tensor<4x32xf32>
    %dst = ktdp.construct_memory_view %out_addr,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<global>
    } : memref<8x32xf32>
    %dst_tile = ktdp.construct_access_tile %dst[%row, %zero] {
      access_tile_set = #stick, access_tile_order = #identity
    } : memref<8x32xf32> -> !ktdp.access_tile<4x32xindex>
    ktdp.store %result, %dst_tile : tensor<4x32xf32>, <4x32xindex>
    return
  }

  // One core broadcasts lane 0 of rows 4..7 from ct_local.
  // Expected result: out[row, lane] = local[row, 0].
  func.func @i32_splat_view_repro(%out_addr: index) attributes {grid = [1]} {
    %zero = arith.constant 0 : index
    %row = arith.constant 4 : index
    %src = ktdp.construct_memory_view %zero,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<ct_local>
    } : memref<8x32xi32>
    %src_tile = ktdp.construct_access_tile %src[%row, %zero] {
      access_tile_set = #scalar, access_tile_order = #identity
    } : memref<8x32xi32> -> !ktdp.access_tile<4x1xindex>
    %value = ktdp.load %src_tile : <4x1xindex> -> tensor<4x1xi32>
    %init = tensor.empty() : tensor<4x32xi32>
    %result = linalg.generic {
      indexing_maps = [#broadcast, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%value : tensor<4x1xi32>) outs(%init : tensor<4x32xi32>) {
    ^bb0(%x: i32, %unused: i32):
      %y = spyreop.identity %x : i32
      linalg.yield %y : i32
    } -> tensor<4x32xi32>
    %dst = ktdp.construct_memory_view %out_addr,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<global>
    } : memref<8x32xi32>
    %dst_tile = ktdp.construct_access_tile %dst[%row, %zero] {
      access_tile_set = #stick, access_tile_order = #identity
    } : memref<8x32xi32> -> !ktdp.access_tile<4x32xindex>
    ktdp.store %result, %dst_tile : tensor<4x32xi32>, <4x32xindex>
    return
  }

  // Core 0 handles rows 0..3; core 1 handles rows 4..7. The first compute
  // group copies input to ct_local, then the second broadcasts lane 0.
  // Each output lane contains input[row, 0].
  func.func @i32_splat_view_two_cores(%in_addr: index, %out_addr: index) attributes {grid = [2]} {
    %zero = arith.constant 0 : index
    %four = arith.constant 4 : index
    %core = ktdp.get_compute_tile_id : index
    %row = arith.muli %core, %four : index
    %src = ktdp.construct_memory_view %zero,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<ct_local>
    } : memref<8x32xi32>
    %input = ktdp.construct_memory_view %in_addr,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<global>
    } : memref<8x32xi32>
    %input_tile = ktdp.construct_access_tile %input[%row, %zero] {
      access_tile_set = #stick, access_tile_order = #identity
    } : memref<8x32xi32> -> !ktdp.access_tile<4x32xindex>
    %input_values = ktdp.load %input_tile : <4x32xindex> -> tensor<4x32xi32>
    %stage_init = tensor.empty() : tensor<4x32xi32>
    %stage_values = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%input_values : tensor<4x32xi32>) outs(%stage_init : tensor<4x32xi32>) {
    ^bb0(%x: i32, %unused_stage: i32):
      %copy = spyreop.identity %x : i32
      linalg.yield %copy : i32
    } -> tensor<4x32xi32>
    %local_tile = ktdp.construct_access_tile %src[%row, %zero] {
      access_tile_set = #stick, access_tile_order = #identity
    } : memref<8x32xi32> -> !ktdp.access_tile<4x32xindex>
    ktdp.store %stage_values, %local_tile : tensor<4x32xi32>, <4x32xindex>
    %src_tile = ktdp.construct_access_tile %src[%row, %zero] {
      access_tile_set = #scalar, access_tile_order = #identity
    } : memref<8x32xi32> -> !ktdp.access_tile<4x1xindex>
    %value = ktdp.load %src_tile : <4x1xindex> -> tensor<4x1xi32>
    %init = tensor.empty() : tensor<4x32xi32>
    %result = linalg.generic {
      indexing_maps = [#broadcast, #identity],
      iterator_types = ["parallel", "parallel"]
    } ins(%value : tensor<4x1xi32>) outs(%init : tensor<4x32xi32>) {
    ^bb0(%x: i32, %unused: i32):
      %y = spyreop.identity %x : i32
      linalg.yield %y : i32
    } -> tensor<4x32xi32>
    %dst = ktdp.construct_memory_view %out_addr,
        sizes: [8, 32], strides: [32, 1] {
      coordinate_set = #memory, memory_space = #ktdp.memory_space<global>
    } : memref<8x32xi32>
    %dst_tile = ktdp.construct_access_tile %dst[%row, %zero] {
      access_tile_set = #stick, access_tile_order = #identity
    } : memref<8x32xi32> -> !ktdp.access_tile<4x32xindex>
    ktdp.store %result, %dst_tile : tensor<4x32xi32>, <4x32xindex>
    return
  }
}
