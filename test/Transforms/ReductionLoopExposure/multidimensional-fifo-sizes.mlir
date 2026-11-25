// SPDX-License-Identifier: Apache-2.0
// RUN: dataflow-scheduler-opt %s -reduction-loop-exposure | FileCheck %s

// The reduction covers six packets, not every tensor axis. Shrink the FIFO
// volume from 192 to 32 and preserve the row/lane extents in the memory access.
// CHECK-LABEL: func.func @multidimensional_fifo_sizes(
// CHECK: !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
// CHECK: %[[SIX:.*]] = arith.constant 6 : index
// CHECK: scf.for %[[PACKET:.*]] = {{.*}} to %[[SIX]]
// CHECK: ktdf.data_transfer from {{.*}}[%[[PACKET]], {{.*}}, 0] size [1, 1, 32] to {{.*}} size [32]
// CHECK-SAME: memref<6x2x32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
// CHECK: ktdf.read_from_fifo {{.*}} -> tensor<1x1x32xf32>
// CHECK: ktdf.data_transfer from {{.*}} size [32] to {{.*}} size [1, 32]

ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")

func.func @multidimensional_fifo_sizes(
    %input: memref<6x2x32xf32, "L1">, %output: memref<2x32xf32, "L1">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %row = %c0 to %c2 step %c1 {
    ktdf.pipeline {
      %private:4 = ktdf.private -> (
          !ktdf.fifo.slot<"L1LU" -> "SFU", 192xf32>,
          !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, !ktdf.token, !ktdf.token) {
        %in = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 192xf32>
        %out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>
        %loaded = ktdf.create_token : !ktdf.token
        %computed = ktdf.create_token : !ktdf.token
        ktdf.private_yield %in, %out, %loaded, %computed :
            !ktdf.fifo.slot<"L1LU" -> "SFU", 192xf32>,
            !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%private#2) {
        ktdf.data_transfer from %input[0, %row, 0] size [6, 1, 32]
            to %private#0 size [6, 1, 32] : memref<6x2x32xf32, "L1">,
            !ktdf.fifo.slot<"L1LU" -> "SFU", 192xf32>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%private#2) depends_out(%private#3) {
        %value = ktdf.read_from_fifo %private#0
            : <"L1LU" -> "SFU", 192xf32> -> tensor<6x1x32xf32>
        %init = tensor.empty() : tensor<1x32xf32>
        %sum = linalg.generic {
          indexing_maps = [affine_map<(k, r, l) -> (k, r, l)>,
                           affine_map<(k, r, l) -> (r, l)>],
          iterator_types = ["reduction", "parallel", "parallel"]
        } ins(%value : tensor<6x1x32xf32>) outs(%init : tensor<1x32xf32>) {
        ^bb0(%x: f32, %acc: f32):
          %next = arith.addf %x, %acc : f32
          linalg.yield %next : f32
        } -> tensor<1x32xf32>
        ktdf.write_to_fifo %sum, %private#1
            : tensor<1x32xf32>, <"SFU" -> "L1SU", 32xf32>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%private#3) depends_out(none) {
        ktdf.data_transfer from %private#1 size [32]
            to %output[%row, 0] size [1, 32] :
            !ktdf.fifo.slot<"SFU" -> "L1SU", 32xf32>, memref<2x32xf32, "L1">
      } {applicable_units = ["L1SU"]}
    }
  }
  return
}
