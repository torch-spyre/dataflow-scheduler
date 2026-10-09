// RUN: dataflow-scheduler-opt --split-reduction-inner-outer-dim %s | FileCheck %s

// Tests that a reduction whose accumulator is initialised by a linalg.fill
// keeps that fill on the first generic of the split, where the accumulation
// starts. The fill is rebuilt on the first generic's intermediate shape
// (tensor<2x64xf16>), which is not the output's (tensor<2xf16>); the second
// generic starts from a tensor.empty, and the original fill is erased.

// CHECK-LABEL:   func.func @max_fill_init
// CHECK:           %[[CST:.*]] = arith.constant 1.000000e+01 : f16
// CHECK:           ktdf.stage depends_in(%{{.*}}#2) depends_out(%{{.*}}#3) {
// CHECK-NEXT:        %[[READ:.*]] = ktdf.read_from_fifo
// CHECK-NEXT:        %[[EMPTY_0:.*]] = tensor.empty() : tensor<2x64xf16>
// CHECK-NEXT:        %[[FILL:.*]] = linalg.fill ins(%[[CST]] : f16) outs(%[[EMPTY_0]] : tensor<2x64xf16>) -> tensor<2x64xf16>
// CHECK-NEXT:        %[[G1:.*]] = linalg.generic {{.*}} ins(%[[READ]] : tensor<2x256x64xf16>) outs(%[[FILL]] : tensor<2x64xf16>)
// CHECK:               arith.maximumf
// CHECK:             %[[EMPTY_1:.*]] = tensor.empty() : tensor<2xf16>
// CHECK-NEXT:        %[[G2:.*]] = linalg.generic {{.*}} ins(%[[G1]] : tensor<2x64xf16>) outs(%[[EMPTY_1]] : tensor<2xf16>)
// CHECK:               arith.maximumf
// CHECK:             ktdf.write_to_fifo %[[G2]]
// CHECK-NOT:       linalg.fill

#map  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0)>
#set  = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 1 >= 0)>

module {
  module {
    func.func @max_fill_init() attributes {grid = [1]} {
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
      %cst         = arith.constant 1.000000e+01 : f16
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [2], strides: [1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<2xf16>
      %memspacecast   = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, "DDR">
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR">
      %cast           = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, "DDR"> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<2xf16> to memref<2xf16, "DDR">
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [2], strides: [1] : memref<2xf16, "DDR"> to memref<2xf16, strided<[1]>, "DDR">
      %cast_2         = memref.cast %reinterpret_cast_1 : memref<2xf16, strided<[1]>, "DDR"> to memref<2xf16, strided<[1], offset: ?>, "DDR">
      ktdf.pipeline {
        %priv:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, !ktdf.token, !ktdf.token) {
          %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>
          %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>
          %t0 = ktdf.create_token : !ktdf.token
          %t1 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %f0, %f1, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%priv#2) {
          ktdf.data_transfer from %cast[%c0, %c0, %c0] size [2, 256, 64] to %priv#0 size [32768] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, "DDR">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32768xf16>
        } {applicable_units = ["L1LU"]}
        ktdf.stage depends_in(%priv#2) depends_out(%priv#3) {
          %in   = ktdf.read_from_fifo %priv#0 : <"L1LU" -> "SFU", 32768xf16> -> tensor<2x256x64xf16>
          %empty = tensor.empty() : tensor<2xf16>
          %init = linalg.fill ins(%cst : f16) outs(%empty : tensor<2xf16>) -> tensor<2xf16>
          %res  = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction", "reduction"]} ins(%in : tensor<2x256x64xf16>) outs(%init : tensor<2xf16>) {
          ^bb0(%a: f16, %acc: f16):
            %s = arith.maximumf %a, %acc : f16
            linalg.yield %s : f16
          } -> tensor<2xf16>
          ktdf.write_to_fifo %res, %priv#1 : tensor<2xf16>, <"SFU" -> "L1SU", 2xf16>
        } {applicable_units = ["SFU"]}
        ktdf.stage depends_in(%priv#3) depends_out(none) {
          ktdf.data_transfer from %priv#1 size [2] to %cast_2[%c0] size [2] : !ktdf.fifo.slot<"SFU" -> "L1SU", 2xf16>, memref<2xf16, strided<[1], offset: ?>, "DDR">
        } {applicable_units = ["L1SU"]}
      }
      return
    }
  }
}
