//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//===----------------------------------------------------------------------===//

// A constant in a body that arithmetic reads, rather than a template call.
//
// It goes in a register like any other, and the multiply reads a lane out of it:
// a template takes the register and reads the lanes itself, arithmetic takes one
// float. basic.mlir pins the other side of that, where the register is handed
// to the call.

// RUN: dataflow-scheduler-opt %s -materialize-registers | FileCheck %s

// CHECK:      %[[C:.*]] = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 1.250000e-01 : f16
// CHECK:      %[[REG:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT: linalg.fill ins(%[[C]] : f16) outs(%[[REG]] : memref<64xf16, "SFU_REG">)
// CHECK:      linalg.generic
// CHECK-NEXT: ^bb0(%[[X:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:   %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[V:.*]] = memref.load %[[REG]]{{\[}}%[[ZERO]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:   %[[SCALED:.*]] = arith.mulf %[[X]], %[[V]] : f16
// CHECK-NEXT:   linalg.yield %[[SCALED]] : f16

// The register is what is filled, so nothing multiplies by one.
// CHECK-NOT: arith.mulf %{{.*}}, %[[C]]

#id = affine_map<(d0) -> (d0)>
module {
  func.func @scale(%in: tensor<64xf16>) -> tensor<64xf16> {
    %init = tensor.empty() : tensor<64xf16>
    %result = linalg.generic {indexing_maps = [#id, #id],
                              iterator_types = ["parallel"]}
        ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
    ^bb0(%x: f16, %out: f16):
      %c = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 1.250000e-01 : f16
      %scaled = arith.mulf %x, %c : f16
      linalg.yield %scaled : f16
    } -> tensor<64xf16>
    return %result : tensor<64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
