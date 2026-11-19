// RUN: dataflow-scheduler-opt %s -allow-unregistered-dialect -apply-device-patterns='groups=post_scheduling' | FileCheck %s

// CHECK-LABEL:   func.func @sum_reduction() {
// CHECK-NEXT:     %[[ALLOC_0:.*]] = memref.alloc() : memref<64x64xf16, "L1">
// CHECK-NEXT:     %[[ALLOC_1:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:     "test.simd_reduction_sum"(%[[ALLOC_0]], %[[ALLOC_1]]) : (memref<64x64xf16, "L1">, memref<64xf16, "L1">) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// CHECK-LABEL:   func.func @sum_reduction_simd_lane_spans_two_axes() {
// CHECK-NEXT:     %[[ALLOC_2:.*]] = memref.alloc() : memref<64x32x2xf16, "L1">
// CHECK-NEXT:     %[[ALLOC_3:.*]] = memref.alloc() : memref<64xf16, "L1">
// CHECK-NEXT:     "test.simd_reduction_sum"(%[[ALLOC_2]], %[[ALLOC_3]]) : (memref<64x32x2xf16, "L1">, memref<64xf16, "L1">) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }


// Verify that the post_scheduling PDL pattern in the sample device rewrites a
// linalg.generic sum-reduction (inner-dim, output via subview) into a bare
// test.simd_reduction op whose second operand is the subview's *source*.

#map_in  = affine_map<(d0, d1) -> (d0, d1)>
#map_out = affine_map<(d0, d1) -> (d0)>
#map_in3  = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_out3 = affine_map<(d0, d1, d2) -> (d0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")

  func.func @sum_reduction() {
    %alloc_in  = memref.alloc() : memref<64x64xf16, "L1">
    %alloc_acc = memref.alloc() : memref<64xf16, "L1">

    // The pattern requires the output to be a memref.subview of the accumulator.
    %subview = memref.subview %alloc_acc[0][64][1]
        : memref<64xf16, "L1"> to memref<64xf16, "L1">

    linalg.generic {
      indexing_maps = [#map_in, #map_out],
      iterator_types = ["parallel", "reduction"]
    } ins(%alloc_in : memref<64x64xf16, "L1">)
      outs(%subview  : memref<64xf16, "L1">) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %out : f16
      linalg.yield %sum : f16
    }

    return
  }

  // A SIMD lane spanning two input axes: for f16 the sample device's compute unit
  // has 64 lanes, which the trailing [32, 2] axes hold together.  Both are
  // reductions, so the generic is still the inner-dim reduction.
  func.func @sum_reduction_simd_lane_spans_two_axes() {
    %alloc_in  = memref.alloc() : memref<64x32x2xf16, "L1">
    %alloc_acc = memref.alloc() : memref<64xf16, "L1">
    %subview = memref.subview %alloc_acc[0][64][1]
        : memref<64xf16, "L1"> to memref<64xf16, "L1">

    linalg.generic {
      indexing_maps = [#map_in3, #map_out3],
      iterator_types = ["parallel", "reduction", "reduction"]
    } ins(%alloc_in : memref<64x32x2xf16, "L1">)
      outs(%subview  : memref<64xf16, "L1">) {
    ^bb0(%in: f16, %out: f16):
      %sum = arith.addf %in, %out : f16
      linalg.yield %sum : f16
    }

    return
  }
}
