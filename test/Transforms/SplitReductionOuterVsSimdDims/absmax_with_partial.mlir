// RUN: dataflow-scheduler-opt --split-reduction-outer-vs-simd-dims %s | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK: #[[$ATTR_4:.+]] = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
// CHECK: #[[$ATTR_5:.+]] = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>
// CHECK-LABEL:   module {
// CHECK:     func.func @sum_onstick_1core() attributes {grid = [1]} {
// CHECK:       call @local_schedule_0() : () -> ()
// CHECK:       return
// CHECK:     }
// CHECK:     func.func private @local_schedule_0()
// CHECK:   }
// CHECK:   ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

// CHECK-LABEL:   module @local_schedule_0 {
// CHECK-NEXT:     func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK-NEXT:       %[[CONSTANT_0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[CONSTANT_1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[CONSTANT_2:.*]] = arith.constant 8589934592 : index
// CHECK-NEXT:       %[[CONSTANT_3:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_0:.*]] = ktdp.construct_memory_view %[[CONSTANT_0]], sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #[[$ATTR_4]], memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
// CHECK-NEXT:       %[[CONSTRUCT_MEMORY_VIEW_1:.*]] = ktdp.construct_memory_view %[[CONSTANT_2]], sizes: [2, 64], strides: [64, 1] {coordinate_set = #[[$ATTR_5]], memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_0:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_0]] : memref<2x256x64xf16> to memref<2x256x64xf16, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_0]] to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[CAST_0:.*]] = memref.cast %[[REINTERPRET_CAST_0]] : memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[MEMORY_SPACE_CAST_1:.*]] = memref.memory_space_cast %[[CONSTRUCT_MEMORY_VIEW_1]] : memref<2x64xf16> to memref<2x64xf16, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[MEMORY_SPACE_CAST_1]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, #ktdp.memory_space<global>> to memref<2x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[CAST_1:.*]] = memref.cast %[[REINTERPRET_CAST_1]] : memref<2x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>> to memref<2x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
// CHECK-NEXT:       %[[EMPTY_0:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:       ktdf.pipeline {
// CHECK-NEXT:         %[[PRIVATE_0:.*]]:4 = ktdf.private -> (memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:           %[[ALLOC_0:.*]] = memref.alloc() : memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:           %[[ALLOC_1:.*]] = memref.alloc() : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:           %[[CREATE_TOKEN_0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           %[[CREATE_TOKEN_1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:           ktdf.private_yield %[[ALLOC_0]], %[[ALLOC_1]], %[[CREATE_TOKEN_0]], %[[CREATE_TOKEN_1]] : memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
// CHECK-NEXT:         }
// CHECK-NEXT:         ktdf.stage depends_in(none) depends_out(%[[VAL_0:.*]]#2) {
// CHECK-NEXT:           scf.for %[[VAL_1:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[SUBI_0:.*]] = arith.subi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK-NEXT:             %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_0]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[CAST_0]]{{\[}}%[[VAL_1]], 0, 0] size [1, 256, 64] to %[[VAL_0]]#0{{\[}}%[[DIVSI_0]], 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["L1LU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_2:.*]]#2) depends_out(%[[VAL_2]]#3) {
// CHECK-NEXT:           scf.for %[[VAL_3:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[CONSTANT_4:.*]] = arith.constant 4 : index
// CHECK-NEXT:             %[[CONSTANT_5:.*]] = arith.constant 0 : index
// CHECK-NEXT:             %[[CONSTANT_6:.*]] = arith.constant 1 : index
// CHECK-NEXT:             scf.for %[[VAL_4:.*]] = %[[CONSTANT_5]] to %[[CONSTANT_4]] step %[[CONSTANT_6]] {
// CHECK-NEXT:               %[[CMPI_0:.*]] = arith.cmpi eq, %[[VAL_4]], %[[CONSTANT_5]] : index
// CHECK-NEXT:               ktdf.pipeline {
// CHECK-NEXT:                 %[[PRIVATE_1:.*]]:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                   %[[FIFO_0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>
// CHECK-NEXT:                   %[[FIFO_1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                   %[[FIFO_2:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                   %[[CREATE_TOKEN_2:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                   %[[CREATE_TOKEN_3:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                   ktdf.private_yield %[[FIFO_0]], %[[FIFO_1]], %[[FIFO_2]], %[[CREATE_TOKEN_2]], %[[CREATE_TOKEN_3]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:                 }
// CHECK-NEXT:                 ktdf.stage depends_in(none) depends_out(%[[VAL_5:.*]]#3) {
// CHECK-NEXT:                   %[[CONSTANT_7:.*]] = arith.constant 0 : index
// CHECK-NEXT:                   %[[CONSTANT_8:.*]] = arith.constant 1 : index
// CHECK-NEXT:                   %[[SUBI_1:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_7]] : index
// CHECK-NEXT:                   %[[DIVSI_1:.*]] = arith.divsi %[[SUBI_1]], %[[CONSTANT_8]] : index
// CHECK-NEXT:                   ktdf.data_transfer from %[[VAL_2]]#0{{\[}}%[[DIVSI_1]], %[[CONSTANT_7]], %[[VAL_4]] * 64, %[[CONSTANT_7]]] size [1, 1, 64, 64] to %[[VAL_5]]#0 size [4096] : memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>
// CHECK-NEXT:                   scf.if %[[CMPI_0]] {
// CHECK-NEXT:                   } else {
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_2]]#1{{\[}}%[[DIVSI_1]], %[[CONSTANT_7]], %[[CONSTANT_7]]] size [1, 1, 64] to %[[VAL_5]]#1 size [64] : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                   }
// CHECK-NEXT:                 } {applicable_units = ["L1LU"]}
// CHECK-NEXT:                 ktdf.stage depends_in(%[[VAL_6:.*]]#3) depends_out(%[[VAL_6]]#4) {
// CHECK-NEXT:                   %[[READ_FROM_FIFO_0:.*]] = ktdf.read_from_fifo %[[VAL_6]]#0 : <"L1LU" -> "SFU", 4096xf16> -> tensor<1x64x64xf16>
// CHECK-NEXT:                   %[[EMPTY_1:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:                   %[[GENERIC_0:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[READ_FROM_FIFO_0]] : tensor<1x64x64xf16>) outs(%[[EMPTY_1]] : tensor<1x64xf16>) {
// CHECK-NEXT:                   ^bb0(%[[VAL_7:.*]]: f16, %[[VAL_8:.*]]: f16):
// CHECK-NEXT:                     %[[ABSF_0:.*]] = math.absf %[[VAL_7]] : f16
// CHECK-NEXT:                     %[[ABSF_1:.*]] = math.absf %[[VAL_8]] : f16
// CHECK-NEXT:                     %[[MAXNUMF_0:.*]] = arith.maxnumf %[[ABSF_0]], %[[ABSF_1]] : f16
// CHECK-NEXT:                     linalg.yield %[[MAXNUMF_0]] : f16
// CHECK-NEXT:                   } -> tensor<1x64xf16>
// CHECK-NEXT:                   %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (tensor<1x64xf16>) {
// CHECK-NEXT:                     scf.yield %[[GENERIC_0]] : tensor<1x64xf16>
// CHECK-NEXT:                   } else {
// CHECK-NEXT:                     %[[READ_FROM_FIFO_1:.*]] = ktdf.read_from_fifo %[[VAL_6]]#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
// CHECK-NEXT:                     %[[GENERIC_1:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_2]], #[[$ATTR_2]]], iterator_types = ["parallel", "parallel"]} ins(%[[READ_FROM_FIFO_1]] : tensor<1x64xf16>) outs(%[[GENERIC_0]] : tensor<1x64xf16>) {
// CHECK-NEXT:                     ^bb0(%[[VAL_9:.*]]: f16, %[[VAL_10:.*]]: f16):
// CHECK-NEXT:                       %[[ABSF_2:.*]] = math.absf %[[VAL_9]] : f16
// CHECK-NEXT:                       %[[ABSF_3:.*]] = math.absf %[[VAL_10]] : f16
// CHECK-NEXT:                       %[[MAXNUMF_1:.*]] = arith.maxnumf %[[ABSF_2]], %[[ABSF_3]] : f16
// CHECK-NEXT:                       linalg.yield %[[MAXNUMF_1]] : f16
// CHECK-NEXT:                     } -> tensor<1x64xf16>
// CHECK-NEXT:                     scf.yield %[[GENERIC_1]] : tensor<1x64xf16>
// CHECK-NEXT:                   }
// CHECK-NEXT:                   %[[EMPTY_2:.*]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:                   %[[GENERIC_2:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_3]], #[[$ATTR_1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[IF_0]] : tensor<1x64xf16>) outs(%[[EMPTY_2]] : tensor<1x64xf16>) {
// CHECK-NEXT:                   ^bb0(%[[VAL_11:.*]]: f16, %[[VAL_12:.*]]: f16):
// CHECK-NEXT:                     %[[ABSF_4:.*]] = math.absf %[[VAL_11]] : f16
// CHECK-NEXT:                     %[[ABSF_5:.*]] = math.absf %[[VAL_12]] : f16
// CHECK-NEXT:                     %[[MAXNUMF_2:.*]] = arith.maxnumf %[[ABSF_4]], %[[ABSF_5]] : f16
// CHECK-NEXT:                     linalg.yield %[[MAXNUMF_2]] : f16
// CHECK-NEXT:                   } -> tensor<1x64xf16>
// CHECK-NEXT:                   ktdf.write_to_fifo %[[GENERIC_2]], %[[VAL_6]]#2 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                 } {applicable_units = ["SFU"]}
// CHECK-NEXT:                 ktdf.stage depends_in(%[[VAL_13:.*]]#4) depends_out(none) {
// CHECK-NEXT:                   %[[CONSTANT_9:.*]] = arith.constant 0 : index
// CHECK-NEXT:                   %[[CONSTANT_10:.*]] = arith.constant 1 : index
// CHECK-NEXT:                   %[[SUBI_2:.*]] = arith.subi %[[VAL_3]], %[[CONSTANT_9]] : index
// CHECK-NEXT:                   %[[DIVSI_2:.*]] = arith.divsi %[[SUBI_2]], %[[CONSTANT_10]] : index
// CHECK-NEXT:                   ktdf.data_transfer from %[[VAL_13]]#2 size [64] to %[[VAL_2]]#1{{\[}}%[[DIVSI_2]], %[[CONSTANT_9]], %[[CONSTANT_9]]] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>
// CHECK-NEXT:                 } {applicable_units = ["L1SU"]}
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-NEXT:         ktdf.stage depends_in(%[[VAL_14:.*]]#3) depends_out(none) {
// CHECK-NEXT:           scf.for %[[VAL_15:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] {
// CHECK-NEXT:             %[[SUBI_3:.*]] = arith.subi %[[VAL_15]], %[[CONSTANT_0]] : index
// CHECK-NEXT:             %[[DIVSI_3:.*]] = arith.divsi %[[SUBI_3]], %[[CONSTANT_1]] : index
// CHECK-NEXT:             ktdf.data_transfer from %[[VAL_14]]#1{{\[}}%[[DIVSI_3]], 0, 0] size [1, 1, 64] to %[[CAST_1]]{{\[}}%[[VAL_15]], %[[CONSTANT_0]] * 64] size [1, 64] : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<2x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
// CHECK-NEXT:           } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:         } {applicable_units = ["L1SU"]}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }



// An absmax with a chunked (partial) reduction dim.  ReductionDimChunking has
// already wrapped the combine scf.if around the generic's result.  The pass
// must place G2 (SIMD reduction) *after* that scf.if so it sees the fully
// combined partial value, and must thread the scf.if result into G2 instead of
// G1's result directly.
//
// The original generic:
//   input:  (d0,d1,d2,d3) -> (d0,d1,d2)   tensor<1x64x64xf16>
//   output: (d0,d1,d2,d3) -> (d0,d3)       tensor<1x64xf16>
//   iterator_types = ["parallel","reduction","reduction","parallel"]
//   d0 = batch (1, parallel), d1 = outer reduction (64),
//   d2 = SIMD reduction (64, vector_length), d3 = output-only parallel (64)
//
// After the split:
//   G1 (outer reduction, reduces d1):
//     input:  (d0,d1,d2) -> (d0,d1,d2)     tensor<1x64x64xf16>
//     output: (d0,d1,d2) -> (d0,d2)         tensor<1x64xf16>
//     iterator_types = ["parallel","reduction","parallel"]
//     body: same absmax body as original
//
//   scf.if (combine, unchanged — feeds G2)
//
//   G2 (SIMD reduction, reduces d2 in G2 loop space):
//     input:  (d0,d1,d2) -> (d0,d1)         tensor<1x64xf16>   (from scf.if result)
//     output: (d0,d1,d2) -> (d0,d2)         tensor<1x64xf16>
//     iterator_types = ["parallel","reduction","parallel"]
//     body: abs of each side then maxnumf (combiner rebuilt)

#map  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d3)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#set  = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  module {
    func.func @sum_onstick_1core() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0          = arith.constant 0 : index
      %c1          = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %c2          = arith.constant 2 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [2, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<2x64xf16>
      %memspacecast     = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>>
      %cast     = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>
      %memspacecast_0   = memref.memory_space_cast %1 : memref<2x64xf16> to memref<2x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<2x64xf16, #ktdp.memory_space<global>> to memref<2x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>>
      %cast_2   = memref.cast %reinterpret_cast_1 : memref<2x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>> to memref<2x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
      %2 = tensor.empty() : tensor<1x64xf16>
      ktdf.pipeline {
        %3:4 = ktdf.private -> (memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
          %alloc   = memref.alloc() : memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>
          %alloc_3 = memref.alloc() : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>
          %4 = ktdf.create_token : !ktdf.token
          %5 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %4, %5 : memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%3#2) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %4 = arith.subi %arg0, %c0 : index
            %5 = arith.divsi %4, %c1 : index
            ktdf.data_transfer from %cast[%arg0, 0, 0] size [1, 256, 64] to %3#0[%5, 0, 0, 0] size [1, 1, 256, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%3#2) depends_out(%3#3) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %c4   = arith.constant 4 : index
            %c0_3 = arith.constant 0 : index
            %c1_4 = arith.constant 1 : index
            scf.for %arg1 = %c0_3 to %c4 step %c1_4 {
              %4 = arith.cmpi eq, %arg1, %c0_3 : index
              ktdf.pipeline {
                %5:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                  %6  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>
                  %7  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  %8  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                  %9  = ktdf.create_token : !ktdf.token
                  %10 = ktdf.create_token : !ktdf.token
                  ktdf.private_yield %6, %7, %8, %9, %10 : !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
                }
                ktdf.stage depends_in(none) depends_out(%5#3) {
                  %c0_5 = arith.constant 0 : index
                  %c1_6 = arith.constant 1 : index
                  %6 = arith.subi %arg0, %c0_5 : index
                  %7 = arith.divsi %6, %c1_6 : index
                  ktdf.data_transfer from %3#0[%7, %c0_5, %arg1 * 64, %c0_5] size [1, 1, 64, 64] to %5#0 size [4096] : memref<2x1x256x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>
                  scf.if %4 {
                  } else {
                    ktdf.data_transfer from %3#1[%7, %c0_5, %c0_5] size [1, 1, 64] to %5#1 size [64] : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                  }
                } {applicable_units = ["L1LU"]}
                ktdf.stage depends_in(%5#3) depends_out(%5#4) {
                  %6 = ktdf.read_from_fifo %5#0 : <"L1LU" -> "SFU", 4096xf16> -> tensor<1x64x64xf16>
                  %7 = tensor.empty() : tensor<1x64xf16>
                  %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "reduction", "parallel"]} ins(%6 : tensor<1x64x64xf16>) outs(%7 : tensor<1x64xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %10 = math.absf %in : f16
                    %11 = math.absf %out : f16
                    %12 = arith.maxnumf %10, %11 : f16
                    linalg.yield %12 : f16
                  } -> tensor<1x64xf16>
                  %9 = scf.if %4 -> (tensor<1x64xf16>) {
                    scf.yield %8 : tensor<1x64xf16>
                  } else {
                    %10 = ktdf.read_from_fifo %5#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
                    %11 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%10 : tensor<1x64xf16>) outs(%8 : tensor<1x64xf16>) {
                    ^bb0(%in: f16, %out: f16):
                      %12 = math.absf %in : f16
                      %13 = math.absf %out : f16
                      %14 = arith.maxnumf %12, %13 : f16
                      linalg.yield %14 : f16
                    } -> tensor<1x64xf16>
                    scf.yield %11 : tensor<1x64xf16>
                  }
                  ktdf.write_to_fifo %9, %5#2 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
                } {applicable_units = ["SFU"]}
                ktdf.stage depends_in(%5#4) depends_out(none) {
                  %c0_5 = arith.constant 0 : index
                  %c1_6 = arith.constant 1 : index
                  %6 = arith.subi %arg0, %c0_5 : index
                  %7 = arith.divsi %6, %c1_6 : index
                  ktdf.data_transfer from %5#2 size [64] to %3#1[%7, %c0_5, %c0_5] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, #ktdp.memory_space<ct_local>>
                } {applicable_units = ["L1SU"]}
              }
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%3#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c2 step %c1 {
            %4 = arith.subi %arg0, %c0 : index
            %5 = arith.divsi %4, %c1 : index
            ktdf.data_transfer from %3#1[%5, 0, 0] size [1, 1, 64] to %cast_2[%arg0, %c0 * 64] size [1, 64] : memref<2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<2x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1SU"]}
      }
      return
    }
  }
}
