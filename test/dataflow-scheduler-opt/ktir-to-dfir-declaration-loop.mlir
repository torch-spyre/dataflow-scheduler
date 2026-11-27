// RUN: dataflow-scheduler-opt -ktir-to-dfir -split-input-file -verify-diagnostics "%s" | FileCheck "%s"

// A kernel whose program runs in a loop. After compute-group-extraction the loop
// is in the declaration module, around the program's call. The pipeline used to
// assert in ktir-pipeline, which took the loop for a pipeline to build.

// CHECK-LABEL: func.func @arith_math_test()
// CHECK:         scf.for
// CHECK:           call @local_schedule_0()
// CHECK:       module @local_schedule_0 attributes {dataflow_scheduler.scheduled}
// CHECK:         dataflow.program_unit

module {
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../Dialect/KTDFArch/sample_device.mlir")
  func.func @arith_math_test() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
    %tile_size = arith.constant 3 : index
    %A_start_address = arith.constant 1024 : index
    %B_start_address = arith.constant 12288 : index
    %C_start_address = arith.constant 18432 : index

    %id = ktdp.get_compute_tile_id : index
    %start_row = arith.muli %id, %tile_size : index

    // Construct memory views
    %A_view = ktdp.construct_memory_view %A_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %B_view = ktdp.construct_memory_view %B_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    %C_view = ktdp.construct_memory_view %C_start_address, sizes: [96, 64], strides: [64, 1] {
      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 95 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      memory_space = #ktdp.memory_space<global>
    } : memref<96x64xf16>

    
    // Construct access tiles
    %A_access_tile = ktdp.construct_access_tile %A_view[%start_row, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

    %B_access_tile = ktdp.construct_access_tile %B_view[%start_row, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

    // Load data
    %A_data_tile = ktdp.load %A_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
    %B_data_tile = ktdp.load %B_access_tile : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>

    // Add the results using linalg.add
    %result_empty = tensor.empty() : tensor<1x64xf16>
    %result = linalg.add 
      ins(%A_data_tile, %B_data_tile : tensor<1x64xf16>, tensor<1x64xf16>)
      outs(%result_empty: tensor<1x64xf16>) -> tensor<1x64xf16>

    // Construct access tile for output
    %C_access_tile = ktdp.construct_access_tile %C_view[%start_row, %c0] {
      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + 0 >= 0, d1 >= 0, -d1 + 63 >= 0)>,
      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
    } : memref<96x64xf16> -> !ktdp.access_tile<1x64xindex>

    // Store result
    ktdp.store %result, %C_access_tile : tensor<1x64xf16>, !ktdp.access_tile<1x64xindex>
    }
    return
  }
}
