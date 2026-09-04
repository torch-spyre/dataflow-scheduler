// A parallel dimension outside the last reduction is a loop around the whole
// reduction rather than lanes of the register: every turn of it writes the
// register's same position. Only the parallel dimensions inside the last
// reduction are lanes -- here 64 rather than 8 by 64.

// RUN: dataflow-scheduler-opt -materialize-registers %s | FileCheck %s

// CHECK-LABEL: func.func @parallel_outside_the_reduction(
// CHECK:       memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK-NOT:   memref.alloca() : memref<512xf16, "SFU_REG">

#in  = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#out = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
module {
  func.func @parallel_outside_the_reduction(%in: tensor<2x8x4xf16>)
      -> tensor<8x64xf16> {
    %init = tensor.empty() : tensor<8x64xf16>
    %result = linalg.generic {indexing_maps = [#in, #out],
                              iterator_types = ["reduction", "parallel",
                                                "reduction", "parallel"]}
        ins(%in : tensor<2x8x4xf16>) outs(%init : tensor<8x64xf16>) {
    ^bb0(%x: f16, %acc: f16):
      %held = memref.alloca() : memref<f16, "SFU_REG">
      memref.store %x, %held[] : memref<f16, "SFU_REG">
      %read = memref.load %held[] : memref<f16, "SFU_REG">
      %sum = arith.addf %read, %acc : f16
      linalg.yield %sum : f16
    } -> tensor<8x64xf16>
    return %result : tensor<8x64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
