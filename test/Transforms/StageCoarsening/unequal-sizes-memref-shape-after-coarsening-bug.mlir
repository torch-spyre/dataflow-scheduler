// RUN: dataflow-scheduler-opt --stage-coarsening %s | FileCheck %s

// Regression test: after stage coarsening expands a buffer with N new leading
// dimensions, data transfers to/from that buffer must have N extra leading sizes
// of 1 prepended.  Previously cloneDataTransfer omitted them, producing a size
// count that didn't match the expanded memref rank.

// The MNILU load stage transfers into the rank-6 expanded l1_A buffer.
// Size list must have 6 entries matching the rank.
// CHECK: ktdf.data_transfer from {{.*}} size [1, 1, 1, 64] to {{.*}} size [1, 1, 1, 1, 1, 64] : memref<12x1x64x64xf16,{{.*}}>, memref<?x?x1x1x1x64xf16,{{.*}}>

// The L1LU stage reads from the rank-6 l1_A buffer into the FIFO.
// Source side must have 6 sizes; FIFO dest has its own flat size.
// CHECK: ktdf.data_transfer from {{.*}} size [1, 1, 1, 1, 1, 64] to {{.*}} size [64] : memref<?x?x1x1x1x64xf16,{{.*}}>, !ktdf.fifo

// The L1SU stage stores from the FIFO into the rank-7 expanded l1_C buffer.
// Dest side must have 7 sizes.
// CHECK: ktdf.data_transfer from {{.*}} size [64] to {{.*}} size [1, 1, 1, 1, 1, 1, 64] : !ktdf.fifo{{.*}}, memref<?x?x1x1x1x1x64xf16,{{.*}}>

// The MNISU store stage transfers from the rank-7 l1_C buffer back to DDR.
// Source side must have 7 sizes.
// CHECK: ktdf.data_transfer from {{.*}} size [1, 1, 1, 1, 1, 1, 64] to {{.*}} size [1, 1, 1, 1, 64] : memref<?x?x1x1x1x1x64xf16,{{.*}}>, memref<12x1x1x64x64xf16,{{.*}}>

#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d2, d3, d4)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0)>
#set2 = affine_set<(d0, d1, d2, d3, d4) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 >= 0, d3 >= 0, -d3 + 63 >= 0, d4 >= 0, -d4 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @Add_1414() attributes {grid = [2]} {
      call @"local-schedule-0"() : () -> ()
      return
    }
    func.func private @"local-schedule-0"()
  }
  module {
    func.func private @"local-schedule-0"() attributes {grid = [2]} {
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
      %3 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %4 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
      %5 = arith.ceildivui %c12, %3 : index
      scf.for %arg0 = %c0 to %5 step %c1 {
        %6 = arith.ceildivui %c64, %4 : index
        %7 = ktdf.tiling.derive_size [%arg0 : %3], total_size = %c12 : index
        scf.for %arg1 = %c0 to %6 step %c1 {
          %8 = ktdf.tiling.derive_size [%arg1 : %4], total_size = %c64 : index
          scf.for %arg2 = %c0 to %7 step %c1 {
            %9 = ktdf.tiling.linearize_index [%arg0 : %3], [%arg2 : %c1] : index
            scf.for %arg3 = %c0 to %8 step %c1 {
              %10 = ktdf.tiling.linearize_index [%arg1 : %4], [%arg3 : %c1] : index
              ktdf.pipeline {
                %11:10 = ktdf.private -> (memref<1x1x1x64xf16, "L1">, memref<1x64xf16, "L1">, memref<1x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
                  %alloc = memref.alloc() : memref<1x1x1x64xf16, "L1">
                  %alloc_6 = memref.alloc() : memref<1x64xf16, "L1">
                  %alloc_7 = memref.alloc() : memref<1x1x1x1x64xf16, "L1">
                  %12:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  %13 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                  %14 = ktdf.create_token : !ktdf.token
                  %15 = ktdf.create_token : !ktdf.token
                  %16 = ktdf.create_token : !ktdf.token
                  %17 = ktdf.create_token : !ktdf.token
                  ktdf.private_yield %alloc, %alloc_6, %alloc_7, %12#0, %12#1, %13, %14, %15, %16, %17 : memref<1x1x1x64xf16, "L1">, memref<1x64xf16, "L1">, memref<1x1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
                }
                ktdf.stage depends_in(none) depends_out(%11#6) {
                  ktdf.data_transfer from %cast[%9, %c0, %10, %c0] size [1, 1, 1, 64] to %11#0[0, 0, 0, 0] size [1, 1, 1, 64] : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<1x1x1x64xf16, "L1">
                  ktdf.data_transfer from %cast_2[%c0, %c0] size [1, 64] to %11#1[0, 0] size [1, 64] : memref<1x64xf16, strided<[64, 1], offset: ?>, "DDR">, memref<1x64xf16, "L1">
                } {applicable_units = ["MNILU"]}
                ktdf.stage depends_in(%11#6) depends_out(%11#7) {
                  ktdf.data_transfer from %11#0[0, 0, 0, 0] size [1, 1, 1, 64] to %11#3 size [64] : memref<1x1x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  ktdf.data_transfer from %11#1[0, 0] size [1, 64] to %11#4 size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                } {applicable_units = ["L1LU"]}
                ktdf.stage depends_in(%11#7) depends_out(%11#8) {
                  %12 = ktdf.read_from_fifo %11#3 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x1x64xf16>
                  %13 = ktdf.read_from_fifo %11#4 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
                  %14 = tensor.empty() : tensor<1x1x1x1x64xf16>
                  %15 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%12, %13 : tensor<1x1x1x64xf16>, tensor<1x64xf16>) outs(%14 : tensor<1x1x1x1x64xf16>) {
                  ^bb0(%in: f16, %in_6: f16, %out: f16):
                    %16 = arith.addf %in, %in_6 : f16
                    linalg.yield %16 : f16
                  } -> tensor<1x1x1x1x64xf16>
                  ktdf.write_to_fifo %15, %11#5 : tensor<1x1x1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
                } {applicable_units = ["SFU"]}
                ktdf.stage depends_in(%11#8) depends_out(%11#9) {
                  ktdf.data_transfer from %11#5 size [64] to %11#2[0, 0, 0, 0, 0] size [1, 1, 1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x1x1x1x64xf16, "L1">
                } {applicable_units = ["L1SU"]}
                ktdf.stage depends_in(%11#9) depends_out(none) {
                  ktdf.data_transfer from %11#2[0, 0, 0, 0, 0] size [1, 1, 1, 1, 64] to %cast_5[%9, %c0, %c0, %10, %c0] size [1, 1, 1, 1, 64] : memref<1x1x1x1x64xf16, "L1">, memref<12x1x1x64x64xf16, strided<[4096, 4096, 4096, 64, 1], offset: ?>, "DDR">
                } {applicable_units = ["MNISU"]}
              }
            } {loop_type = #ktdf.loop_type<parallel_loop>}
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
      } {loop_type = #ktdf.loop_type<parallel_loop>}
      return
    }
  }
}
