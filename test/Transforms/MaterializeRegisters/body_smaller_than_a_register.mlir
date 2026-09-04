// A body working on fewer elements than a register holds still gets a whole
// one: what it works on is a position in the register. Sizing the register to
// the tile instead would leave one that is not a whole number of lanes.

// RUN: dataflow-scheduler-opt -materialize-registers %s | FileCheck %s

// CHECK-LABEL: func.func @body_smaller_than_a_register(
// CHECK:       memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK-NOT:   memref.alloca() : memref<24xf16, "SFU_REG">

#in  = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d1)>
module {
  func.func @body_smaller_than_a_register(%in: tensor<4x24xf16>)
      -> tensor<24xf16> {
    %init = tensor.empty() : tensor<24xf16>
    %result = linalg.generic {indexing_maps = [#in, #out],
                              iterator_types = ["reduction", "parallel"]}
        ins(%in : tensor<4x24xf16>) outs(%init : tensor<24xf16>) {
    ^bb0(%x: f16, %acc: f16):
      %held = memref.alloca() : memref<f16, "SFU_REG">
      memref.store %x, %held[] : memref<f16, "SFU_REG">
      %read = memref.load %held[] : memref<f16, "SFU_REG">
      %sum = arith.addf %read, %acc : f16
      linalg.yield %sum : f16
    } -> tensor<24xf16>
    return %result : tensor<24xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
