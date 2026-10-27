// CAUTION: Be careful when updating this test to make sure SSA results from ktdf.private are reusing the appropriate FileCheck matches.

// C[M][N] = A[N] + B[M][N]
// The main point of this test is to make sure buffer expansion is performed correctly for the broadcast load of A (ie L1 buffer for A expanded only in the N dim)

// RUN: dataflow-scheduler-opt --stage-coarsening --canonicalize %s | FileCheck %s


// CHECK: #[[$ATTR_0:.+]] = affine_map<()[s0, s1] -> (-s0 + s1)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<()[s0, s1] -> ((-s0 + s1) ceildiv 64)>


// CHECK:       func.func @broadcast_add_kernel(%[[VAL_0:.*]]: memref<?xf16, "DDR">, %[[VAL_1:.*]]: memref<?x?xf16, "DDR">, %[[VAL_2:.*]]: memref<?x?xf16, "DDR">, %[[VAL_3:.*]]: index, %[[VAL_4:.*]]: index) {
// CHECK-NEXT:     %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[VAL_7:.*]] = arith.constant 64 : index
// CHECK-NEXT:     %[[VAL_8:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[VAL_9:.*]] = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
// CHECK-NEXT:     %[[VAL_10:.*]] = arith.muli %[[VAL_9]], %[[VAL_7]] : index
// CHECK-NEXT:     scf.for %[[VAL_11:.*]] = %[[VAL_5]] to %[[VAL_3]] step %[[VAL_8]] {
// CHECK-NEXT:       %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_8]] : index
// CHECK-NEXT:       %[[VAL_13:.*]] = arith.minsi %[[VAL_3]], %[[VAL_12]] : index
// CHECK-NEXT:       %[[VAL_14:.*]] = affine.apply #[[$ATTR_0]](){{\[}}%[[VAL_11]], %[[VAL_13]]]
// CHECK-NEXT:       scf.for %[[VAL_15:.*]] = %[[VAL_5]] to %[[VAL_4]] step %[[VAL_10]] {
// CHECK-NEXT:         %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_10]] : index
// CHECK-NEXT:         %[[VAL_17:.*]] = arith.minsi %[[VAL_4]], %[[VAL_16]] : index
// CHECK-NEXT:         %[[VAL_18:.*]] = affine.apply #[[$ATTR_1]](){{\[}}%[[VAL_15]], %[[VAL_17]]]
// CHECK-NEXT:         ktdf.pipeline {
// CHECK-NEXT:           %[[VAL_19:.*]]:5 = ktdf.private -> (memref<?x64xf16, "L1">, memref<?x?x64xf16, "L1">, memref<?x?x64xf16, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:             %[[VAL_20:.*]] = memref.alloc(%[[VAL_18]]) : memref<?x64xf16, "L1">
// CHECK-NEXT:             %[[VAL_21:.*]] = memref.alloc(%[[VAL_14]], %[[VAL_18]]) : memref<?x?x64xf16, "L1">
// CHECK-NEXT:             %[[VAL_22:.*]] = memref.alloc(%[[VAL_14]], %[[VAL_18]]) : memref<?x?x64xf16, "L1">
// CHECK-NEXT:             %[[VAL_23:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             %[[VAL_24:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:             ktdf.private_yield %[[VAL_20]], %[[VAL_21]], %[[VAL_22]], %[[VAL_23]], %[[VAL_24]] : memref<?x64xf16, "L1">, memref<?x?x64xf16, "L1">, memref<?x?x64xf16, "L1">, !ktdf.token, !ktdf.token
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(none) depends_out(%[[VAL_19]]#3) {
// CHECK-NEXT:             scf.for %[[VAL_26:.*]] = %[[VAL_5]] to %[[VAL_14]] step %[[VAL_6]] {
// CHECK-NEXT:               scf.for %[[VAL_27:.*]] = %[[VAL_5]] to %[[VAL_18]] step %[[VAL_6]] {
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_0]]{{\[}}%[[VAL_27]] * 64 + %[[VAL_15]]] size [64] to %[[VAL_19]]#0{{\[}}%[[VAL_27]], 0] size [1, 64] : memref<?xf16, "DDR">, memref<?x64xf16, "L1">
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_1]]{{\[}}%[[VAL_26]] + %[[VAL_11]], %[[VAL_27]] * 64 + %[[VAL_15]]] size [1, 64] to %[[VAL_19]]#1{{\[}}%[[VAL_26]], %[[VAL_27]], 0] size [1, 1, 64] : memref<?x?xf16, "DDR">, memref<?x?x64xf16, "L1">
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_19]]#3) depends_out(%[[VAL_19]]#4) {
// CHECK-NEXT:             scf.for %[[VAL_32:.*]] = %[[VAL_5]] to %[[VAL_14]] step %[[VAL_6]] {
// CHECK-NEXT:               scf.for %[[VAL_33:.*]] = %[[VAL_5]] to %[[VAL_18]] step %[[VAL_6]] {
// CHECK-NEXT:                 ktdf.pipeline {
// CHECK-NEXT:                   %[[VAL_34:.*]]:5 = ktdf.private -> (memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                     %[[VAL_35:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:                     %[[VAL_36:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:                     %[[VAL_37:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:                     %[[VAL_38:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     %[[VAL_39:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                     ktdf.private_yield %[[VAL_35]], %[[VAL_36]], %[[VAL_37]], %[[VAL_38]], %[[VAL_39]] : memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">, !ktdf.token, !ktdf.token
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(none) depends_out(%[[VAL_34]]#3) {
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_19]]#0{{\[}}%[[VAL_33]], 0] size [1, 64] to %[[VAL_34]]#0[0] size [64] : memref<?x64xf16, "L1">, memref<64xf16, "SFU_REG">
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_19]]#1{{\[}}%[[VAL_32]], %[[VAL_33]], 0] size [1, 1, 64] to %[[VAL_34]]#1[0] size [64] : memref<?x?x64xf16, "L1">, memref<64xf16, "SFU_REG">
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(%[[VAL_34]]#3) depends_out(%[[VAL_34]]#4) {
// CHECK-NEXT:                     linalg.add ins(%[[VAL_34]]#0, %[[VAL_34]]#1 : memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">) outs(%[[VAL_34]]#2 : memref<64xf16, "SFU_REG">)
// CHECK-NEXT:                   }
// CHECK-NEXT:                   ktdf.stage depends_in(%[[VAL_34]]#4) depends_out(none) {
// CHECK-NEXT:                     ktdf.data_transfer from %[[VAL_34]]#2[0] size [64] to %[[VAL_19]]#2{{\[}}%[[VAL_32]], %[[VAL_33]], 0] size [1, 1, 64] : memref<64xf16, "SFU_REG">, memref<?x?x64xf16, "L1">
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:           ktdf.stage depends_in(%[[VAL_19]]#4) depends_out(none) {
// CHECK-NEXT:             scf.for %[[VAL_44:.*]] = %[[VAL_5]] to %[[VAL_14]] step %[[VAL_6]] {
// CHECK-NEXT:               scf.for %[[VAL_45:.*]] = %[[VAL_5]] to %[[VAL_18]] step %[[VAL_6]] {
// CHECK-NEXT:                 ktdf.data_transfer from %[[VAL_19]]#2{{\[}}%[[VAL_44]], %[[VAL_45]], 0] size [1, 1, 64] to %[[VAL_2]]{{\[}}%[[VAL_44]] + %[[VAL_11]], %[[VAL_45]] * 64 + %[[VAL_15]]] size [1, 64] : memref<?x?x64xf16, "L1">, memref<?x?xf16, "DDR">
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       } {loop_type = {{.*}}<parallel_loop>}
// CHECK-NEXT:     } {loop_type = {{.*}}<parallel_loop>}
// CHECK-NEXT:     return
// CHECK-NEXT:   }

#map = affine_map<()[s0, s1] -> (-s0 + s1)>
#map1 = affine_map<()[s0, s1] -> ((-s0 + s1) ceildiv 64)>
#map2 = affine_map<(d0)[s0] -> (d0 * 64 + s0)>
#map3 = affine_map<(d0)[s0] -> (d0 + s0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @broadcast_add_kernel(%arg0: memref<?xf16, "DDR">, %arg1: memref<?x?xf16, "DDR">, %arg2: memref<?x?xf16, "DDR">, %arg3: index, %arg4: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %0 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
    %1 = ktdf.tiling.reserve_size {divisibility = 1 : index, min_value = 1 : index} : index
    %2 = arith.muli %1, %c64 : index
    scf.for %m0 = %c0 to %arg3 step %0 {
      %3 = arith.addi %m0, %0 : index
      %4 = arith.minsi %arg3, %3 : index
      %5 = affine.apply #map()[%m0, %4]
      scf.for %n0 = %c0 to %arg4 step %2 {
        %6 = arith.addi %n0, %2 : index
        %7 = arith.minsi %arg4, %6 : index
        %8 = affine.apply #map1()[%n0, %7]
        scf.for %m1 = %c0 to %5 step %c1 {
          scf.for %n1 = %c0 to %8 step %c1 {
            ktdf.pipeline {
              %9:11 = ktdf.private -> (memref<64xf16, "L1">, memref<64xf16, "L1">, memref<64xf16, "L1">, memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
                %alloc = memref.alloc() : memref<64xf16, "L1"> // this should get expanded in the %n1 dim only
                %alloc_0 = memref.alloc() : memref<64xf16, "L1"> // other L1 buffers including this one should get expanded in %m1 and %n1
                %alloc_1 = memref.alloc() : memref<64xf16, "L1">
                %alloc_2 = memref.alloc() : memref<64xf16, "SFU_REG">
                %alloc_3 = memref.alloc() : memref<64xf16, "SFU_REG">
                %alloc_4 = memref.alloc() : memref<64xf16, "SFU_REG">
                %10 = ktdf.create_token : !ktdf.token
                %11 = ktdf.create_token : !ktdf.token
                %12 = ktdf.create_token : !ktdf.token
                %13 = ktdf.create_token : !ktdf.token
                %14 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %alloc, %alloc_0, %alloc_1, %alloc_2, %alloc_3, %alloc_4, %10, %11, %12, %13, %14 : memref<64xf16, "L1">, memref<64xf16, "L1">, memref<64xf16, "L1">, memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%9#6) {
                %10 = affine.apply #map2(%n1)[%n0]
                ktdf.data_transfer from %arg0[%10] size [64] to %9#0[%c0] size [64] : memref<?xf16, "DDR">, memref<64xf16, "L1">
                %11 = affine.apply #map2(%n1)[%n0]
                %12 = affine.apply #map3(%m1)[%m0]
                ktdf.data_transfer from %arg1[%12, %11] size [1, 64] to %9#1[%c0] size [64] : memref<?x?xf16, "DDR">, memref<64xf16, "L1">
              }
              ktdf.stage depends_in(%9#6) depends_out(%9#7) {
                ktdf.data_transfer from %9#0[%c0] size [64] to %9#3[%c0] size [64] : memref<64xf16, "L1">, memref<64xf16, "SFU_REG">
                ktdf.data_transfer from %9#1[%c0] size [64] to %9#4[%c0] size [64] : memref<64xf16, "L1">, memref<64xf16, "SFU_REG">
              }
              ktdf.stage depends_in(%9#7) depends_out(%9#8) {
                linalg.add ins(%9#3, %9#4 : memref<64xf16, "SFU_REG">, memref<64xf16, "SFU_REG">) outs(%9#5 : memref<64xf16, "SFU_REG">)
              }
              ktdf.stage depends_in(%9#8) depends_out(%9#9) {
                ktdf.data_transfer from %9#5[%c0] size [64] to %9#2[%c0] size [64] : memref<64xf16, "SFU_REG">, memref<64xf16, "L1">
              }
              ktdf.stage depends_in(%9#9) depends_out(%9#10) {
                %10 = affine.apply #map2(%n1)[%n0]
                %11 = affine.apply #map3(%m1)[%m0]
                ktdf.data_transfer from %9#2[%c0] size [64] to %arg2[%11, %10] size [1, 64] : memref<64xf16, "L1">, memref<?x?xf16, "DDR">
              }
            }
          }
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}

