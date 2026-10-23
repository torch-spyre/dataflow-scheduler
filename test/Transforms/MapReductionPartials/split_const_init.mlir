// RUN: dataflow-scheduler-opt --map-reduction-partials --split-input-file %s | FileCheck %s

// The inner-dim generic of a split reduction combines every lane of the
// outer-dim accumulator into lane 0, so its init is counted exactly once by
// writing it into that lane: after the outer-dim accumulator is set to the
// neutral element, and before the loop that accumulates into it.

// CHECK-LABEL:   func.func @sum_split_const_init
// CHECK:           ktdf.stage
// CHECK:           ktdf.stage
// CHECK:             %[[ACC:.*]] = memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK-NEXT:        %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:        %[[ID0:.*]] = memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK-NEXT:        linalg.fill ins(%[[ZERO]] : f16) outs(%[[ID0]] : memref<1x64xf16, "SFU_REG">)
// CHECK-NEXT:        memref.copy %[[ID0]], %[[ACC]]
// CHECK-NEXT:        %[[LANE:.*]] = memref.subview %[[ACC]][0, 0] [1, 1] [1, 1] : memref<1x64xf16, "SFU_REG"> to memref<1x1xf16, strided<[64, 1]>, "SFU_REG">
// CHECK-NEXT:        %[[ONE:.*]] = arith.constant 1.000000e+00 : f16
// CHECK-NEXT:        %[[ID1:.*]] = memref.alloc() : memref<1x1xf16, "SFU_REG">
// CHECK-NEXT:        linalg.fill ins(%[[ONE]] : f16) outs(%[[ID1]] : memref<1x1xf16, "SFU_REG">)
// CHECK-NEXT:        memref.copy %[[ID1]], %[[LANE]]
// CHECK:             scf.for
// CHECK:               linalg.generic {{.*}} outs(%[[ACC]] : memref<1x64xf16, "SFU_REG">)
// CHECK:             } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:        %[[SV:.*]] = memref.subview %[[ACC]][0, 0] [1, 1] [1, 1]
// CHECK-NEXT:        linalg.generic {{.*}} ins(%[[ACC]] : memref<1x64xf16, "SFU_REG">) outs(%[[SV]] : memref<1x1xf16, strided<[64, 1]>, "SFU_REG">)
// CHECK-NOT:         linalg.fill
// CHECK:             ktdf.write_to_fifo %[[ACC]]
// CHECK-NOT:     tensor.empty
// CHECK-NOT:     arith.constant dense

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  module {
    func.func @sum_split_const_init() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %c256 = arith.constant 256 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [256, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<256x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>>
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>
      %memspacecast_0 = memref.memory_space_cast %1 : memref<256x64xf16> to memref<256x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [256, 64], strides: [64, 1] : memref<256x64xf16, #ktdp.memory_space<global>> to memref<256x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>>
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<256x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>
          %alloc_3 = memref.alloc() : memref<256x1x64xf16, #ktdp.memory_space<ct_local>>
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %cast[0, %arg0, 0] size [2, 1, 64] to %2#0[%4, 0, 0, 0] size [1, 2, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %c0_3 = arith.constant 0 : index
            %c1_4 = arith.constant 1 : index
            %c1_5 = arith.constant 1 : index
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                %c2 = arith.constant 2 : index
                scf.for %arg1 = %c0_3 to %c2 step %c1_4 {
                  ktdf.data_transfer from %2#0[%5, %arg1, 0, 0] size [1, 1, 1, 64] to %3#0 size [64] : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                } {loop_type = #ktdf.loop_type<reduction_loop>}
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = tensor.empty() : tensor<1x64xf16>
                %c2 = arith.constant 2 : index
                %5 = scf.for %arg1 = %c0_3 to %c2 step %c1_4 iter_args(%arg2 = %4) -> (tensor<1x64xf16>) {
                  %8 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
                  %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel", "parallel"]} ins(%8 : tensor<1x1x64xf16>) outs(%arg2 : tensor<1x64xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %10 = arith.addf %in, %out : f16
                    linalg.yield %10 : f16
                  } -> tensor<1x64xf16>
                  scf.yield %9 : tensor<1x64xf16>
                } {loop_type = #ktdf.loop_type<reduction_loop>}
                %6 = arith.constant dense<1.000000e+00> : tensor<1x64xf16>
                %7 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction", "parallel"]} ins(%5 : tensor<1x64xf16>) outs(%6 : tensor<1x64xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %8 = arith.addf %in, %out : f16
                  linalg.yield %8 : f16
                } -> tensor<1x64xf16>
                ktdf.write_to_fifo %7, %3#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                %c2 = arith.constant 2 : index
                scf.for %arg1 = %c0_3 to %c2 step %c1_4 {
                  %4 = arith.subi %arg0, %c0 : index
                  %5 = arith.divsi %4, %c1 : index
                  %6 = arith.cmpi eq, %arg1, %c1_5 : index
                  scf.if %6 {
                    ktdf.data_transfer from %3#1 size [64] to %2#1[%5, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>
                  }
                } {loop_type = #ktdf.loop_type<reduction_loop>}
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %2#1[%4, 0, 0] size [1, 1, 64] to %cast_2[%arg0, %c0 * 64] size [1, 64] : memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}

// -----

// A conditional init -- the user's value on the first chunk only -- moves
// behind the outer-dim init and writes lane 0 in each branch.

// CHECK-LABEL:   func.func @sum_split_if_init
// CHECK:             %[[ACC:.*]] = memref.alloc() : memref<1x64xf16, "SFU_REG">
// CHECK:             memref.copy %{{.*}}, %[[ACC]]
// CHECK-NEXT:        %[[FIRST:.*]] = arith.cmpi eq
// CHECK-NEXT:        %[[LANE:.*]] = memref.subview %[[ACC]][0, 0] [1, 1] [1, 1]
// CHECK-NEXT:        scf.if %[[FIRST]] {
// CHECK-NEXT:          %[[ONE:.*]] = arith.constant 1.000000e+00 : f16
// CHECK-NEXT:          %[[ID1:.*]] = memref.alloc() : memref<1x1xf16, "SFU_REG">
// CHECK-NEXT:          linalg.fill ins(%[[ONE]] : f16) outs(%[[ID1]] : memref<1x1xf16, "SFU_REG">)
// CHECK-NEXT:          memref.copy %[[ID1]], %[[LANE]]
// CHECK-NEXT:        } else {
// CHECK-NEXT:          %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:          %[[ID2:.*]] = memref.alloc() : memref<1x1xf16, "SFU_REG">
// CHECK-NEXT:          linalg.fill ins(%[[ZERO]] : f16) outs(%[[ID2]] : memref<1x1xf16, "SFU_REG">)
// CHECK-NEXT:          memref.copy %[[ID2]], %[[LANE]]
// CHECK-NEXT:        }
// CHECK-NEXT:        arith.constant 2 : index
// CHECK-NEXT:        scf.for

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>
module {
  module {
    func.func @sum_split_if_init() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %c256 = arith.constant 256 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [256, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<256x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>>
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>
      %memspacecast_0 = memref.memory_space_cast %1 : memref<256x64xf16> to memref<256x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [256, 64], strides: [64, 1] : memref<256x64xf16, #ktdp.memory_space<global>> to memref<256x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>>
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<256x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>
          %alloc_3 = memref.alloc() : memref<256x1x64xf16, #ktdp.memory_space<ct_local>>
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %cast[0, %arg0, 0] size [2, 1, 64] to %2#0[%4, 0, 0, 0] size [1, 2, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %c0_3 = arith.constant 0 : index
            %c1_4 = arith.constant 1 : index
            %c1_5 = arith.constant 1 : index
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                %c2 = arith.constant 2 : index
                scf.for %arg1 = %c0_3 to %c2 step %c1_4 {
                  ktdf.data_transfer from %2#0[%5, %arg1, 0, 0] size [1, 1, 1, 64] to %3#0 size [64] : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                } {loop_type = #ktdf.loop_type<reduction_loop>}
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = tensor.empty() : tensor<1x64xf16>
                %c2 = arith.constant 2 : index
                %5 = scf.for %arg1 = %c0_3 to %c2 step %c1_4 iter_args(%arg2 = %4) -> (tensor<1x64xf16>) {
                  %8 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
                  %9 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction", "parallel", "parallel"]} ins(%8 : tensor<1x1x64xf16>) outs(%arg2 : tensor<1x64xf16>) {
                  ^bb0(%in: f16, %out: f16):
                    %10 = arith.addf %in, %out : f16
                    linalg.yield %10 : f16
                  } -> tensor<1x64xf16>
                  scf.yield %9 : tensor<1x64xf16>
                } {loop_type = #ktdf.loop_type<reduction_loop>}
                %first = arith.cmpi eq, %arg0, %c0 : index
                %6 = scf.if %first -> (tensor<1x64xf16>) {
                  %f = arith.constant dense<1.000000e+00> : tensor<1x64xf16>
                  scf.yield %f : tensor<1x64xf16>
                } else {
                  %e = tensor.empty() : tensor<1x64xf16>
                  scf.yield %e : tensor<1x64xf16>
                }
                %7 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction", "parallel"]} ins(%5 : tensor<1x64xf16>) outs(%6 : tensor<1x64xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %8 = arith.addf %in, %out : f16
                  linalg.yield %8 : f16
                } -> tensor<1x64xf16>
                ktdf.write_to_fifo %7, %3#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                %c2 = arith.constant 2 : index
                scf.for %arg1 = %c0_3 to %c2 step %c1_4 {
                  %4 = arith.subi %arg0, %c0 : index
                  %5 = arith.divsi %4, %c1 : index
                  %6 = arith.cmpi eq, %arg1, %c1_5 : index
                  scf.if %6 {
                    ktdf.data_transfer from %3#1 size [64] to %2#1[%5, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>
                  }
                } {loop_type = #ktdf.loop_type<reduction_loop>}
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %2#1[%4, 0, 0] size [1, 1, 64] to %cast_2[%arg0, %c0 * 64] size [1, 64] : memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}

