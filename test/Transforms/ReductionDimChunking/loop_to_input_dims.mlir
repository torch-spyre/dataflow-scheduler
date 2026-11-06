// RUN: dataflow-scheduler-opt --reduction-dim-chunking="num-chunks=4" --split-input-file %s | FileCheck %s

// Chunking cuts the chunk out of the first input along the input dim each
// chunked loop dim indexes, which differs from the loop dim unless the
// indexing map is the identity (issue #179).

// Reduction loop d1 indexes input dim 2: the 256 extent is chunked by 64.
// CHECK-DAG: #[[$PERM_IN:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK-DAG: #[[$PERM_OUT:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$PERM_ID:map[0-9]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @permuted(
// CHECK-SAME:      %[[IN:[a-z0-9_]+]]: memref<2x1x32x256xf16, "L1">, %[[OUT:[a-z0-9_]+]]: memref<2x1x32xf16, "L1">)
// CHECK:         scf.for %[[B:[a-z0-9_]+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK-NEXT:      %[[NCHUNKS:[a-z0-9_]+]] = arith.constant 4 : index
// CHECK-NEXT:      %[[C0:[a-z0-9_]+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:[a-z0-9_]+]] = arith.constant 1 : index
// CHECK-NEXT:      scf.for %[[CHUNK:[a-z0-9_]+]] = %[[C0]] to %[[NCHUNKS]] step %[[C1]] {
// CHECK-NEXT:        %[[FIRST:[a-z0-9_]+]] = arith.cmpi eq, %[[CHUNK]], %[[C0]] : index
// CHECK-NEXT:        ktdf.pipeline {
// CHECK-NEXT:          %[[S:[0-9]+]]:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 2048xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:            ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 2048xf16>
// CHECK-NEXT:            ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>
// CHECK-NEXT:            ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf16>
// CHECK:               ktdf.stage depends_in(none) depends_out(%[[S]]#3) {
// CHECK-NEXT:            %[[Z:[a-z0-9_]+]] = arith.constant 0 : index
// CHECK-NEXT:            %[[ONE:[a-z0-9_]+]] = arith.constant 1 : index
// CHECK-NEXT:            %[[D:[a-z0-9_]+]] = arith.subi %[[B]], %[[Z]] : index
// CHECK-NEXT:            %[[L:[a-z0-9_]+]] = arith.divsi %[[D]], %[[ONE]] : index
// CHECK-NEXT:            ktdf.data_transfer from %[[IN]][%[[L]], %[[Z]], %[[Z]], %[[CHUNK]] * 64] size [1, 1, 32, 64] to %[[S]]#0 size [2048] : memref<2x1x32x256xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 2048xf16>
// CHECK-NEXT:            scf.if %[[FIRST]] {
// CHECK-NEXT:            } else {
// CHECK-NEXT:              ktdf.data_transfer from %[[OUT]][%[[L]], %[[Z]], %[[Z]]] size [1, 1, 32] to %[[S]]#1 size [32] : memref<2x1x32xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf16>
// CHECK-NEXT:            }
// CHECK-NEXT:          } {applicable_units = ["L1LU"]}
// CHECK-NEXT:          ktdf.stage depends_in(%[[S]]#3) depends_out(%[[S]]#4) {
// CHECK-NEXT:            %[[X:[a-z0-9_]+]] = ktdf.read_from_fifo %[[S]]#0 : <"L1LU" -> "SFU", 2048xf16> -> tensor<1x32x64xf16>
// CHECK-NEXT:            %[[E:[a-z0-9_]+]] = tensor.empty() : tensor<1x32xf16>
// CHECK-NEXT:            %[[R:[a-z0-9_]+]] = linalg.generic {indexing_maps = [#[[$PERM_IN]], #[[$PERM_OUT]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[X]] : tensor<1x32x64xf16>) outs(%[[E]] : tensor<1x32xf16>) {
// CHECK-NEXT:            ^bb0(%[[A:[a-z0-9_]+]]: f16, %[[ACC:[a-z0-9_]+]]: f16):
// CHECK-NEXT:              %[[ADD:[a-z0-9_]+]] = arith.addf %[[A]], %[[ACC]] : f16
// CHECK-NEXT:              linalg.yield %[[ADD]] : f16
// CHECK-NEXT:            } -> tensor<1x32xf16>
// CHECK-NEXT:            %[[RES:[a-z0-9_]+]] = scf.if %[[FIRST]] -> (tensor<1x32xf16>) {
// CHECK-NEXT:              scf.yield %[[R]] : tensor<1x32xf16>
// CHECK-NEXT:            } else {
// CHECK-NEXT:              %[[P:[a-z0-9_]+]] = ktdf.read_from_fifo %[[S]]#1 : <"L1LU" -> "SFU", 32xf16> -> tensor<1x32xf16>
// CHECK-NEXT:              %[[SUM:[a-z0-9_]+]] = linalg.generic {indexing_maps = [#[[$PERM_ID]], #[[$PERM_ID]]], iterator_types = ["parallel", "parallel"]} ins(%[[P]] : tensor<1x32xf16>) outs(%[[R]] : tensor<1x32xf16>) {
// CHECK-NEXT:              ^bb0(%[[PA:[a-z0-9_]+]]: f16, %[[PACC:[a-z0-9_]+]]: f16):
// CHECK-NEXT:                %[[PADD:[a-z0-9_]+]] = arith.addf %[[PA]], %[[PACC]] : f16
// CHECK-NEXT:                linalg.yield %[[PADD]] : f16
// CHECK-NEXT:              } -> tensor<1x32xf16>
// CHECK-NEXT:              scf.yield %[[SUM]] : tensor<1x32xf16>
// CHECK-NEXT:            }
// CHECK-NEXT:            ktdf.write_to_fifo %[[RES]], %[[S]]#2 : tensor<1x32xf16>, <"SFU" -> "L1SU", 32xf16>
// CHECK-NEXT:          } {applicable_units = ["SFU"]}
// CHECK-NEXT:          ktdf.stage depends_in(%[[S]]#4) depends_out(none) {
// CHECK-NEXT:            %[[SZ:[a-z0-9_]+]] = arith.constant 0 : index
// CHECK-NEXT:            %[[SONE:[a-z0-9_]+]] = arith.constant 1 : index
// CHECK-NEXT:            %[[SD:[a-z0-9_]+]] = arith.subi %[[B]], %[[SZ]] : index
// CHECK-NEXT:            %[[SL:[a-z0-9_]+]] = arith.divsi %[[SD]], %[[SONE]] : index
// CHECK-NEXT:            ktdf.data_transfer from %[[S]]#2 size [32] to %[[OUT]][%[[SL]], %[[SZ]], %[[SZ]]] size [1, 1, 32] : !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf16>, memref<2x1x32xf16, "L1">
// CHECK-NEXT:          } {applicable_units = ["L1SU"]}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
#in  = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d2)>
module {
  ktdf_arch.device @spyre_single_corelet import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @permuted(%src: memref<2x1x32x256xf16, "L1">, %dst: memref<2x1x32xf16, "L1">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %b = %c0 to %c2 step %c1 {
      ktdf.pipeline {
        %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf16>, !ktdf.token, !ktdf.token) {
          %i = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>
          %o = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %i, %o, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#2) {
          ktdf.data_transfer from %src[%b, 0, 0, 0] size [1, 1, 32, 256] to %p#0 size [8192] : memref<2x1x32x256xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 8192xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%p#2) depends_out(%p#3) {
          %x = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 8192xf16> -> tensor<1x32x256xf16>
          %e = tensor.empty() : tensor<1x32xf16>
          %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "reduction", "parallel"]} ins(%x : tensor<1x32x256xf16>) outs(%e : tensor<1x32xf16>) {
          ^bb0(%a: f16, %acc: f16):
            %s = arith.addf %a, %acc : f16
            linalg.yield %s : f16
          } -> tensor<1x32xf16>
          ktdf.write_to_fifo %r, %p#1 : tensor<1x32xf16>, <"SFU" -> "L1SU", 32xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%p#3) depends_out(none) {
          ktdf.data_transfer from %p#1 size [32] to %dst[%b, 0, 0] size [1, 1, 32] : !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf16>, memref<2x1x32xf16, "L1">
        } {applicable_units = ["L1SU"]}
      }
    }
    return
  }
}

// -----

// Three loops over a rank-2 input: reduction loop d2 indexes input dim 1.
// CHECK-DAG: #[[$RANK_IN:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[$RANK_OUT:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[$RANK_ID:map[0-9]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @rank_mismatch(
// CHECK-SAME:      %[[IN:[a-z0-9_]+]]: memref<2x1x256xf16, "L1">, %[[OUT:[a-z0-9_]+]]: memref<2x1x64xf16, "L1">)
// CHECK:         scf.for %[[B:[a-z0-9_]+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK-NEXT:      %[[NCHUNKS:[a-z0-9_]+]] = arith.constant 4 : index
// CHECK-NEXT:      %[[C0:[a-z0-9_]+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:[a-z0-9_]+]] = arith.constant 1 : index
// CHECK-NEXT:      scf.for %[[CHUNK:[a-z0-9_]+]] = %[[C0]] to %[[NCHUNKS]] step %[[C1]] {
// CHECK-NEXT:        %[[FIRST:[a-z0-9_]+]] = arith.cmpi eq, %[[CHUNK]], %[[C0]] : index
// CHECK-NEXT:        ktdf.pipeline {
// CHECK-NEXT:          %[[S:[0-9]+]]:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:            ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:            ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:            ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK:               ktdf.stage depends_in(none) depends_out(%[[S]]#3) {
// CHECK-NEXT:            %[[Z:[a-z0-9_]+]] = arith.constant 0 : index
// CHECK-NEXT:            %[[ONE:[a-z0-9_]+]] = arith.constant 1 : index
// CHECK-NEXT:            %[[D:[a-z0-9_]+]] = arith.subi %[[B]], %[[Z]] : index
// CHECK-NEXT:            %[[L:[a-z0-9_]+]] = arith.divsi %[[D]], %[[ONE]] : index
// CHECK-NEXT:            ktdf.data_transfer from %[[IN]][%[[L]], %[[Z]], %[[CHUNK]] * 64] size [1, 1, 64] to %[[S]]#0 size [64] : memref<2x1x256xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:            scf.if %[[FIRST]] {
// CHECK-NEXT:            } else {
// CHECK-NEXT:              ktdf.data_transfer from %[[OUT]][%[[L]], %[[Z]], %[[Z]]] size [1, 1, 64] to %[[S]]#1 size [64] : memref<2x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:            }
// CHECK-NEXT:          } {applicable_units = ["L1LU"]}
// CHECK-NEXT:          ktdf.stage depends_in(%[[S]]#3) depends_out(%[[S]]#4) {
// CHECK-NEXT:            %[[X:[a-z0-9_]+]] = ktdf.read_from_fifo %[[S]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
// CHECK-NEXT:            %[[E:[a-z0-9_]+]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:            %[[R:[a-z0-9_]+]] = linalg.generic {indexing_maps = [#[[$RANK_IN]], #[[$RANK_OUT]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[X]] : tensor<1x64xf16>) outs(%[[E]] : tensor<1x64xf16>) {
// CHECK-NEXT:            ^bb0(%[[A:[a-z0-9_]+]]: f16, %[[ACC:[a-z0-9_]+]]: f16):
// CHECK-NEXT:              %[[ADD:[a-z0-9_]+]] = arith.addf %[[A]], %[[ACC]] : f16
// CHECK-NEXT:              linalg.yield %[[ADD]] : f16
// CHECK-NEXT:            } -> tensor<1x64xf16>
// CHECK-NEXT:            %[[RES:[a-z0-9_]+]] = scf.if %[[FIRST]] -> (tensor<1x64xf16>) {
// CHECK-NEXT:              scf.yield %[[R]] : tensor<1x64xf16>
// CHECK-NEXT:            } else {
// CHECK-NEXT:              %[[P:[a-z0-9_]+]] = ktdf.read_from_fifo %[[S]]#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
// CHECK-NEXT:              %[[SUM:[a-z0-9_]+]] = linalg.generic {indexing_maps = [#[[$RANK_ID]], #[[$RANK_ID]]], iterator_types = ["parallel", "parallel"]} ins(%[[P]] : tensor<1x64xf16>) outs(%[[R]] : tensor<1x64xf16>) {
// CHECK-NEXT:              ^bb0(%[[PA:[a-z0-9_]+]]: f16, %[[PACC:[a-z0-9_]+]]: f16):
// CHECK-NEXT:                %[[PADD:[a-z0-9_]+]] = arith.addf %[[PA]], %[[PACC]] : f16
// CHECK-NEXT:                linalg.yield %[[PADD]] : f16
// CHECK-NEXT:              } -> tensor<1x64xf16>
// CHECK-NEXT:              scf.yield %[[SUM]] : tensor<1x64xf16>
// CHECK-NEXT:            }
// CHECK-NEXT:            ktdf.write_to_fifo %[[RES]], %[[S]]#2 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:          } {applicable_units = ["SFU"]}
// CHECK-NEXT:          ktdf.stage depends_in(%[[S]]#4) depends_out(none) {
// CHECK-NEXT:            %[[SZ:[a-z0-9_]+]] = arith.constant 0 : index
// CHECK-NEXT:            %[[SONE:[a-z0-9_]+]] = arith.constant 1 : index
// CHECK-NEXT:            %[[SD:[a-z0-9_]+]] = arith.subi %[[B]], %[[SZ]] : index
// CHECK-NEXT:            %[[SL:[a-z0-9_]+]] = arith.divsi %[[SD]], %[[SONE]] : index
// CHECK-NEXT:            ktdf.data_transfer from %[[S]]#2 size [64] to %[[OUT]][%[[SL]], %[[SZ]], %[[SZ]]] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
// CHECK-NEXT:          } {applicable_units = ["L1SU"]}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
#in  = affine_map<(d0, d1, d2) -> (d0, d2)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  ktdf_arch.device @spyre_single_corelet import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @rank_mismatch(%src: memref<2x1x256xf16, "L1">, %dst: memref<2x1x64xf16, "L1">) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %b = %c0 to %c2 step %c1 {
      ktdf.pipeline {
        %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
          %i = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
          %o = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %i, %o, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%p#2) {
          ktdf.data_transfer from %src[%b, 0, 0] size [1, 1, 256] to %p#0 size [256] : memref<2x1x256xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%p#2) depends_out(%p#3) {
          %x = ktdf.read_from_fifo %p#0 : <"L1LU" -> "SFU", 256xf16> -> tensor<1x256xf16>
          %e = tensor.empty() : tensor<1x64xf16>
          %r = linalg.generic {indexing_maps = [#in, #out], iterator_types = ["parallel", "parallel", "reduction"]} ins(%x : tensor<1x256xf16>) outs(%e : tensor<1x64xf16>) {
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

// Reduction loop d2 indexes input dim 1 of a column sum, not the innermost
// input dim 2.
// CHECK-DAG: #[[$COL_IN:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
// CHECK-DAG: #[[$COL_OUT:map[0-9]*]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG: #[[$COL_ID:map[0-9]*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func.func @column_sum(
// CHECK-SAME:      %[[IN:[a-z0-9_]+]]: memref<2x1x256x64xf16, "L1">, %[[OUT:[a-z0-9_]+]]: memref<2x1x64xf16, "L1">)
// CHECK:         scf.for %[[B:[a-z0-9_]+]] = %{{.+}} to %{{.+}} step %{{.+}} {
// CHECK-NEXT:      %[[NCHUNKS:[a-z0-9_]+]] = arith.constant 4 : index
// CHECK-NEXT:      %[[C0:[a-z0-9_]+]] = arith.constant 0 : index
// CHECK-NEXT:      %[[C1:[a-z0-9_]+]] = arith.constant 1 : index
// CHECK-NEXT:      scf.for %[[CHUNK:[a-z0-9_]+]] = %[[C0]] to %[[NCHUNKS]] step %[[C1]] {
// CHECK-NEXT:        %[[FIRST:[a-z0-9_]+]] = arith.cmpi eq, %[[CHUNK]], %[[C0]] : index
// CHECK-NEXT:        ktdf.pipeline {
// CHECK-NEXT:          %[[S:[0-9]+]]:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:            ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>
// CHECK-NEXT:            ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:            ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK:               ktdf.stage depends_in(none) depends_out(%[[S]]#3) {
// CHECK-NEXT:            %[[Z:[a-z0-9_]+]] = arith.constant 0 : index
// CHECK-NEXT:            %[[ONE:[a-z0-9_]+]] = arith.constant 1 : index
// CHECK-NEXT:            %[[D:[a-z0-9_]+]] = arith.subi %[[B]], %[[Z]] : index
// CHECK-NEXT:            %[[L:[a-z0-9_]+]] = arith.divsi %[[D]], %[[ONE]] : index
// CHECK-NEXT:            ktdf.data_transfer from %[[IN]][%[[L]], %[[Z]], %[[CHUNK]] * 64, %[[Z]]] size [1, 1, 64, 64] to %[[S]]#0 size [4096] : memref<2x1x256x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>
// CHECK-NEXT:            scf.if %[[FIRST]] {
// CHECK-NEXT:            } else {
// CHECK-NEXT:              ktdf.data_transfer from %[[OUT]][%[[L]], %[[Z]], %[[Z]]] size [1, 1, 64] to %[[S]]#1 size [64] : memref<2x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:            }
// CHECK-NEXT:          } {applicable_units = ["L1LU"]}
// CHECK-NEXT:          ktdf.stage depends_in(%[[S]]#3) depends_out(%[[S]]#4) {
// CHECK-NEXT:            %[[X:[a-z0-9_]+]] = ktdf.read_from_fifo %[[S]]#0 : <"L1LU" -> "SFU", 4096xf16> -> tensor<1x64x64xf16>
// CHECK-NEXT:            %[[E:[a-z0-9_]+]] = tensor.empty() : tensor<1x64xf16>
// CHECK-NEXT:            %[[R:[a-z0-9_]+]] = linalg.generic {indexing_maps = [#[[$COL_IN]], #[[$COL_OUT]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[X]] : tensor<1x64x64xf16>) outs(%[[E]] : tensor<1x64xf16>) {
// CHECK-NEXT:            ^bb0(%[[A:[a-z0-9_]+]]: f16, %[[ACC:[a-z0-9_]+]]: f16):
// CHECK-NEXT:              %[[ADD:[a-z0-9_]+]] = arith.addf %[[A]], %[[ACC]] : f16
// CHECK-NEXT:              linalg.yield %[[ADD]] : f16
// CHECK-NEXT:            } -> tensor<1x64xf16>
// CHECK-NEXT:            %[[RES:[a-z0-9_]+]] = scf.if %[[FIRST]] -> (tensor<1x64xf16>) {
// CHECK-NEXT:              scf.yield %[[R]] : tensor<1x64xf16>
// CHECK-NEXT:            } else {
// CHECK-NEXT:              %[[P:[a-z0-9_]+]] = ktdf.read_from_fifo %[[S]]#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
// CHECK-NEXT:              %[[SUM:[a-z0-9_]+]] = linalg.generic {indexing_maps = [#[[$COL_ID]], #[[$COL_ID]]], iterator_types = ["parallel", "parallel"]} ins(%[[P]] : tensor<1x64xf16>) outs(%[[R]] : tensor<1x64xf16>) {
// CHECK-NEXT:              ^bb0(%[[PA:[a-z0-9_]+]]: f16, %[[PACC:[a-z0-9_]+]]: f16):
// CHECK-NEXT:                %[[PADD:[a-z0-9_]+]] = arith.addf %[[PA]], %[[PACC]] : f16
// CHECK-NEXT:                linalg.yield %[[PADD]] : f16
// CHECK-NEXT:              } -> tensor<1x64xf16>
// CHECK-NEXT:              scf.yield %[[SUM]] : tensor<1x64xf16>
// CHECK-NEXT:            }
// CHECK-NEXT:            ktdf.write_to_fifo %[[RES]], %[[S]]#2 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:          } {applicable_units = ["SFU"]}
// CHECK-NEXT:          ktdf.stage depends_in(%[[S]]#4) depends_out(none) {
// CHECK-NEXT:            %[[SZ:[a-z0-9_]+]] = arith.constant 0 : index
// CHECK-NEXT:            %[[SONE:[a-z0-9_]+]] = arith.constant 1 : index
// CHECK-NEXT:            %[[SD:[a-z0-9_]+]] = arith.subi %[[B]], %[[SZ]] : index
// CHECK-NEXT:            %[[SL:[a-z0-9_]+]] = arith.divsi %[[SD]], %[[SONE]] : index
// CHECK-NEXT:            ktdf.data_transfer from %[[S]]#2 size [64] to %[[OUT]][%[[SL]], %[[SZ]], %[[SZ]]] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
// CHECK-NEXT:          } {applicable_units = ["L1SU"]}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
#in  = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#out = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  ktdf_arch.device @spyre_single_corelet import("../../Dialect/KTDFArch/sample_device.mlir")
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
