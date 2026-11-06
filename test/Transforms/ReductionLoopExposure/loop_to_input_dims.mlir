// RUN: dataflow-scheduler-opt --reduction-loop-exposure --split-input-file --verify-diagnostics %s | FileCheck %s

// Exposure loops over each reduction dim with its loop range, and slices the
// first input along the input dim that reduction dim indexes, which differs
// from the loop dim unless the indexing map is the identity.

// Reduction loop d2 (256) indexes input dim 1: one 1x1x64 row per iteration.
// CHECK-DAG: #[[$COL_IN:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK-DAG: #[[$COL_OUT:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @column_sum(
// CHECK-SAME:      %[[SRC:[a-z0-9_]+]]: memref<2x1x256x64xf16, "L1">, %[[DST:[a-z0-9_]+]]: memref<2x1x64xf16, "L1">)
// CHECK:         scf.for %[[B:[a-z0-9_]+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK-NEXT:      %[[C0:[a-z0-9_]+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:[a-z0-9_]+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[LAST:[a-z0-9_]+]] = arith.constant 255 : index
// CHECK-NEXT:      ktdf.pipeline {
// CHECK-NEXT:        %[[S:[0-9]+]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:          ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:          ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK:             ktdf.stage depends_in(none) depends_out(%[[S]]#2) {
// CHECK-NEXT:          %[[LN:[a-z0-9_]+]] = arith.constant 256 : index
// CHECK-NEXT:          scf.for %[[LR:[a-z0-9_]+]] = %[[C0]] to %[[LN]] step %[[C1]] {
// CHECK-NEXT:            ktdf.data_transfer from %[[SRC]][%[[B]], 0, %[[LR]], 0] size [1, 1, 1, 64] to %[[S]]#0 size [64] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:          } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:        } {applicable_units = ["L1LU"]}
// CHECK-NEXT:        ktdf.stage depends_in(%[[S]]#2) depends_out(%[[S]]#3) {
// CHECK-NEXT:          %[[SEED:[a-z0-9_]+]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:          %[[CN:[a-z0-9_]+]] = arith.constant 256 : index
// CHECK-NEXT:          %{{.+}} = scf.for %[[CR:[a-z0-9_]+]] = %[[C0]] to %[[CN]] step %[[C1]] iter_args(%[[ACC:[a-z0-9_]+]] = %[[SEED]]) -> (tensor<1x64xf16>) {
// CHECK-NEXT:            %[[X:[a-z0-9_]+]] = ktdf.read_from_fifo %[[S]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
// CHECK-NEXT:            %[[R:[a-z0-9_]+]] = linalg.generic {indexing_maps = [#[[$COL_IN]], #[[$COL_OUT]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[X]] : tensor<1x1x64xf16>) outs(%[[ACC]] : tensor<1x64xf16>) {
// CHECK-NEXT:            ^bb0(%[[A:[a-z0-9_]+]]: f16, %[[O:[a-z0-9_]+]]: f16):
// CHECK-NEXT:              %[[ADD:[a-z0-9_]+]] = arith.addf %[[A]], %[[O]] : f16
// CHECK-NEXT:              linalg.yield %[[ADD]] : f16
// CHECK-NEXT:            } -> tensor<1x64xf16>
// CHECK-NEXT:            %[[ISLAST:[a-z0-9_]+]] = arith.cmpi eq, %[[CR]], %[[LAST]] : index
// CHECK-NEXT:            scf.if %[[ISLAST]] {
// CHECK-NEXT:              ktdf.write_to_fifo %[[R]], %[[S]]#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %[[R]] : tensor<1x64xf16>
// CHECK-NEXT:          } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:        } {applicable_units = ["SFU"]}
// CHECK-NEXT:        ktdf.stage depends_in(%[[S]]#3) depends_out(none) {
// CHECK-NEXT:          %[[SN:[a-z0-9_]+]] = arith.constant 256 : index
// CHECK-NEXT:          scf.for %[[SR:[a-z0-9_]+]] = %[[C0]] to %[[SN]] step %[[C1]] {
// CHECK-NEXT:            %[[SLAST:[a-z0-9_]+]] = arith.cmpi eq, %[[SR]], %[[LAST]] : index
// CHECK-NEXT:            scf.if %[[SLAST]] {
// CHECK-NEXT:              ktdf.data_transfer from %[[S]]#1 size [64] to %[[DST]][%[[B]], 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
// CHECK-NEXT:            }
// CHECK-NEXT:          } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:        } {applicable_units = ["L1SU"]}
// CHECK-NEXT:      }
// CHECK-NEXT:    }
#in  = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @column_sum(%src: memref<2x1x256x64xf16, "L1">, %dst: memref<2x1x64xf16, "L1">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %b = %c0 to %c2 step %c1 {
      ktdf.pipeline {
        %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
          %i = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
          %o = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %i, %o, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#2) {
          ktdf.data_transfer from %src[%b, 0, 0, 0] size [1, 1, 256, 64] to %p#0 size [16384] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%p#2) depends_out(%p#3) {
          %x = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 16384xf16> -> tensor<1x256x64xf16>
          %e = tensor.empty() : tensor<1x64xf16>
          %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x : tensor<1x256x64xf16>) outs(%e : tensor<1x64xf16>) {
          ^bb0(%a: f16, %acc: f16):
            %s = arith.addf %a, %acc : f16
            linalg.yield %s : f16
          } -> tensor<1x64xf16>
          ktdf.write_to_fifo %r, %p#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%p#3) depends_out(none) {
          ktdf.data_transfer from %p#1 size [64] to %dst[%b, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
        } {applicable_units = ["L1SU"]}
      }
    }
    return
  }
}

// -----

// Three loops over a rank-2 input: reduction loop d2 (256) indexes input dim 0,
// beyond which the loop dim number lies. Its size is the loop range, so it is
// exposed rather than mistaken for a dynamic one.
// CHECK-DAG: #[[$RANK_IN:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG: #[[$RANK_OUT:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-LABEL: func.func @rank_mismatch(
// CHECK-SAME:      %[[SRC:[a-z0-9_]+]]: memref<2x256x64xf16, "L1">, %[[DST:[a-z0-9_]+]]: memref<2x1x64xf16, "L1">)
// CHECK:         scf.for %[[B:[a-z0-9_]+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK-NEXT:      %[[C0:[a-z0-9_]+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:[a-z0-9_]+]] = arith.constant 1 : index
// CHECK-NEXT:      %[[LAST:[a-z0-9_]+]] = arith.constant 255 : index
// CHECK-NEXT:      ktdf.pipeline {
// CHECK-NEXT:        %[[S:[0-9]+]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:          ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:          ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK:             ktdf.stage depends_in(none) depends_out(%[[S]]#2) {
// CHECK-NEXT:          %[[LN:[a-z0-9_]+]] = arith.constant 256 : index
// CHECK-NEXT:          scf.for %[[LR:[a-z0-9_]+]] = %[[C0]] to %[[LN]] step %[[C1]] {
// CHECK-NEXT:            ktdf.data_transfer from %[[SRC]][%[[B]], %[[LR]], 0] size [1, 1, 64] to %[[S]]#0 size [64] : memref<2x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:          } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:        } {applicable_units = ["L1LU"]}
// CHECK-NEXT:        ktdf.stage depends_in(%[[S]]#2) depends_out(%[[S]]#3) {
// CHECK-NEXT:          %[[SEED:[a-z0-9_]+]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:          %[[CN:[a-z0-9_]+]] = arith.constant 256 : index
// CHECK-NEXT:          %{{.+}} = scf.for %[[CR:[a-z0-9_]+]] = %[[C0]] to %[[CN]] step %[[C1]] iter_args(%[[ACC:[a-z0-9_]+]] = %[[SEED]]) -> (tensor<1x64xf16>) {
// CHECK-NEXT:            %[[X:[a-z0-9_]+]] = ktdf.read_from_fifo %[[S]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
// CHECK-NEXT:            %[[R:[a-z0-9_]+]] = linalg.generic {indexing_maps = [#[[$RANK_IN]], #[[$RANK_OUT]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[X]] : tensor<1x64xf16>) outs(%[[ACC]] : tensor<1x64xf16>) {
// CHECK-NEXT:            ^bb0(%[[A:[a-z0-9_]+]]: f16, %[[O:[a-z0-9_]+]]: f16):
// CHECK-NEXT:              %[[ADD:[a-z0-9_]+]] = arith.addf %[[A]], %[[O]] : f16
// CHECK-NEXT:              linalg.yield %[[ADD]] : f16
// CHECK-NEXT:            } -> tensor<1x64xf16>
// CHECK-NEXT:            %[[ISLAST:[a-z0-9_]+]] = arith.cmpi eq, %[[CR]], %[[LAST]] : index
// CHECK-NEXT:            scf.if %[[ISLAST]] {
// CHECK-NEXT:              ktdf.write_to_fifo %[[R]], %[[S]]#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.yield %[[R]] : tensor<1x64xf16>
// CHECK-NEXT:          } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:        } {applicable_units = ["SFU"]}
// CHECK-NEXT:        ktdf.stage depends_in(%[[S]]#3) depends_out(none) {
// CHECK-NEXT:          %[[SN:[a-z0-9_]+]] = arith.constant 256 : index
// CHECK-NEXT:          scf.for %[[SR:[a-z0-9_]+]] = %[[C0]] to %[[SN]] step %[[C1]] {
// CHECK-NEXT:            %[[SLAST:[a-z0-9_]+]] = arith.cmpi eq, %[[SR]], %[[LAST]] : index
// CHECK-NEXT:            scf.if %[[SLAST]] {
// CHECK-NEXT:              ktdf.data_transfer from %[[S]]#1 size [64] to %[[DST]][%[[B]], 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
// CHECK-NEXT:            }
// CHECK-NEXT:          } {loop_type = #ktdf.loop_type<reduction_loop>}
// CHECK-NEXT:        } {applicable_units = ["L1SU"]}
// CHECK-NEXT:      }
// CHECK-NEXT:    }
#in  = affine_map<(d0, d1, d2) -> (d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @rank_mismatch(%src: memref<2x256x64xf16, "L1">, %dst: memref<2x1x64xf16, "L1">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %b = %c0 to %c2 step %c1 {
      ktdf.pipeline {
        %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
          %i = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
          %o = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %i, %o, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#2) {
          ktdf.data_transfer from %src[%b, 0, 0] size [1, 256, 64] to %p#0 size [16384] : memref<2x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%p#2) depends_out(%p#3) {
          %x = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 16384xf16> -> tensor<256x64xf16>
          %e = tensor.empty() : tensor<1x64xf16>
          %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x : tensor<256x64xf16>) outs(%e : tensor<1x64xf16>) {
          ^bb0(%a: f16, %acc: f16):
            %s = arith.addf %a, %acc : f16
            linalg.yield %s : f16
          } -> tensor<1x64xf16>
          ktdf.write_to_fifo %r, %p#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%p#3) depends_out(none) {
          ktdf.data_transfer from %p#1 size [64] to %dst[%b, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
        } {applicable_units = ["L1SU"]}
      }
    }
    return
  }
}

// -----

// Reduction loop d1 indexes input dim 0, whose size is 1, so there is nothing
// to expose; input dim 1 (32) is not the reduction and must not be looped over.
// CHECK-DAG: #[[$UNIT_IN:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
// CHECK-DAG: #[[$UNIT_OUT:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-LABEL: func.func @permuted(
// CHECK-SAME:      %[[SRC:[a-z0-9_]+]]: memref<2x1x32x256xf16, "L1">, %[[DST:[a-z0-9_]+]]: memref<2x32x256xf16, "L1">)
// CHECK:         scf.for %[[B:[a-z0-9_]+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK-NEXT:      ktdf.pipeline {
// CHECK-NEXT:        %[[S:[0-9]+]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 8192xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NOT:     reduction_loop
// CHECK:             ktdf.stage depends_in(none) depends_out(%[[S]]#2) {
// CHECK-NEXT:          ktdf.data_transfer from %[[SRC]][%[[B]], 0, 0, 0] size [1, 1, 32, 256] to %[[S]]#0 size [8192] : memref<2x1x32x256xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>
// CHECK-NEXT:        } {applicable_units = ["L1LU"]}
// CHECK-NEXT:        ktdf.stage depends_in(%[[S]]#2) depends_out(%[[S]]#3) {
// CHECK-NEXT:          %[[X:[a-z0-9_]+]] = ktdf.read_from_fifo %[[S]]#0 : <"L1LU" -> "SFU", 8192xf16> -> tensor<1x32x256xf16>
// CHECK-NEXT:          %[[E:[a-z0-9_]+]] = tensor.empty() : tensor<32x256xf16>
// CHECK-NEXT:          %{{.+}} = linalg.generic {indexing_maps = [#[[$UNIT_IN]], #[[$UNIT_OUT]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[X]] : tensor<1x32x256xf16>) outs(%[[E]] : tensor<32x256xf16>) {
// CHECK-NOT:     reduction_loop
// CHECK:           ktdf.data_transfer from %[[S]]#1 size [8192] to %[[DST]][%[[B]], 0, 0] size [1, 32, 256]
#in  = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
#out = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @permuted(%src: memref<2x1x32x256xf16, "L1">, %dst: memref<2x32x256xf16, "L1">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %b = %c0 to %c2 step %c1 {
      ktdf.pipeline {
        %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 8192xf16>, !ktdf.token, !ktdf.token) {
          %i = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>
          %o = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 8192xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %i, %o, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 8192xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#2) {
          ktdf.data_transfer from %src[%b, 0, 0, 0] size [1, 1, 32, 256] to %p#0 size [8192] : memref<2x1x32x256xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%p#2) depends_out(%p#3) {
          %x = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 8192xf16> -> tensor<1x32x256xf16>
          %e = tensor.empty() : tensor<32x256xf16>
          %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction", "parallel"]} ins(%x : tensor<1x32x256xf16>) outs(%e : tensor<32x256xf16>) {
          ^bb0(%a: f16, %acc: f16):
            %s = arith.addf %a, %acc : f16
            linalg.yield %s : f16
          } -> tensor<32x256xf16>
          ktdf.write_to_fifo %r, %p#1 : tensor<32x256xf16>, <"SFU" -> "L1SU", 8192xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%p#3) depends_out(none) {
          ktdf.data_transfer from %p#1 size [8192] to %dst[%b, 0, 0] size [1, 32, 256] : !ktdf.fifo.slot<"SFU" -> "L1SU", 8192xf16>, memref<2x32x256xf16, "L1">
        } {applicable_units = ["L1SU"]}
      }
    }
    return
  }
}

// -----

// The first input is not indexed by a projected permutation, so there is no
// input dim to slice along. The second input only makes the maps invertible.
#in  = affine_map<(d0, d1, d2) -> (d0, d1 + d2)>
#in2 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#out = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @not_a_permutation(%src: memref<2x1x320xf16, "L1">, %src2: memref<2x1x256x64xf16, "L1">, %dst: memref<2x1x64xf16, "L1">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %b = %c0 to %c2 step %c1 {
      ktdf.pipeline {
        %p:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 320xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
          %i = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 320xf16>
          %j = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
          %o = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %i, %j, %o, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 320xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#3) {
          ktdf.data_transfer from %src[%b, 0, 0] size [1, 1, 320] to %p#0 size [320] : memref<2x1x320xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 320xf16>
          ktdf.data_transfer from %src2[%b, 0, 0, 0] size [1, 1, 256, 64] to %p#1 size [16384] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 16384xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%p#3) depends_out(%p#4) {
          %x = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 320xf16> -> tensor<1x320xf16>
          %y = ktdf.read_from_fifo %p#1 : <"L1LU" -> "SFU", 16384xf16> -> tensor<1x256x64xf16>
          %e = tensor.empty() : tensor<1x64xf16>
          // expected-error @below {{reduction-loop-exposure: the first input is not indexed by a projected permutation, so it cannot be sliced along reduction dim 1}}
          %r = linalg.generic {indexing_maps = [#in, #in2, #out], iterator_types = ["parallel", "reduction", "parallel"]} ins(%x, %y : tensor<1x320xf16>, tensor<1x256x64xf16>) outs(%e : tensor<1x64xf16>) {
          ^bb0(%a: f16, %a2: f16, %acc: f16):
            %s = arith.addf %a, %acc : f16
            linalg.yield %s : f16
          } -> tensor<1x64xf16>
          ktdf.write_to_fifo %r, %p#2 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%p#4) depends_out(none) {
          ktdf.data_transfer from %p#2 size [64] to %dst[%b, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
        } {applicable_units = ["L1SU"]}
      }
    }
    return
  }
}
