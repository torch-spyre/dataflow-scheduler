// RUN: dataflow-scheduler-opt -allow-unregistered-dialect -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -verify-diagnostics -split-input-file %s | FileCheck %s

// The f32 side of a conversion between f16 and f32 is a FIFO slot moved as two
// vectors, at the compute end and at the memory end alike. Every end of such a
// FIFO has to move it so: anything that would move it as one vector is
// rejected before the work is split into program units.

// A dl16tofp32 sends both f32 vectors where the write of its result stood, so
// a FIFO written in between still comes first.

// CHECK-LABEL: func.func @sends_keep_order
// CHECK:         dataflow.program_unit
// CHECK:         dataflow.program_unit
// CHECK:           dataflow.receive %{{.+}} : vector<64xf16>
// CHECK:           dataflow.send %{{.+}}, %{{.+}} : vector<64xf16>
// CHECK:           dataflow.send %{{.+}}, %{{.+}} : vector<32xf32>
// CHECK:           dataflow.send %{{.+}}, %{{.+}} : vector<32xf32>
// CHECK:         dataflow.program_unit
// CHECK:           dataflow.receive %{{.+}} : vector<64xf16>
// CHECK:           dataflow.receive %{{.+}} : vector<32xf32>
// CHECK:           dataflow.receive %{{.+}} : vector<32xf32>
#map = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @sends_keep_order() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %lu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
    %sfu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
    %su = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
    %in = builtin.unrealized_conversion_cast %c0 : index to memref<1x64xf16, "L1">
    %out = builtin.unrealized_conversion_cast %c128 : index to memref<1x64xf32, "L1">
    %out16 = builtin.unrealized_conversion_cast %c0 : index to memref<1x64xf16, "L1">
    ktdf_lowering.execute_on %lu, %sfu, %su {
      %f_in = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      %f_out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf32>
      %f_other = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      ktdf_lowering.execute_on %lu {
        ktdf.data_transfer from %in[0, 0] size [1, 64] to %f_in size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      }
      ktdf_lowering.execute_on %sfu {
        %x = ktdf.read_from_fifo %f_in : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
        %e = tensor.empty() : tensor<64xf32>
        %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%x : tensor<64xf16>) outs(%e : tensor<64xf32>) {
        ^bb0(%a: f16, %b: f32):
          %c = arith.extf %a : f16 to f32
          linalg.yield %c : f32
        } -> tensor<64xf32>
        ktdf.write_to_fifo %x, %f_other : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
        ktdf.write_to_fifo %r, %f_out : tensor<64xf32>, <"SFU" -> "L1SU", 64xf32>
      }
      ktdf_lowering.execute_on %su {
        ktdf.data_transfer from %f_other size [64] to %out16[0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64xf16, "L1">
        ktdf.data_transfer from %f_out size [64] to %out[0, 0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf32>, memref<1x64xf32, "L1">
      }
    }
    return
  }
}

// -----

// The f32 side of a conversion moves as two vectors: a FIFO from it to an
// ordinary compute op, which would receive one, is rejected.
#map = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @wide_fifo_to_compute() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %lu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
    %sfu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
    %su = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
    %in = builtin.unrealized_conversion_cast %c0 : index to memref<1x64xf16, "L1">
    ktdf_lowering.execute_on %lu, %sfu, %su {
      %f_in = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      %f_mid = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 64xf32>
      ktdf_lowering.execute_on %lu {
        ktdf.data_transfer from %in[0, 0] size [1, 64] to %f_in size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      }
      ktdf_lowering.execute_on %sfu {
        %x = ktdf.read_from_fifo %f_in : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
        %e = tensor.empty() : tensor<64xf32>
        // expected-note @below {{the conversion}}
        %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%x : tensor<64xf16>) outs(%e : tensor<64xf32>) {
        ^bb0(%a: f16, %b: f32):
          %c = arith.extf %a : f16 to f32
          linalg.yield %c : f32
        } -> tensor<64xf32>
        ktdf.write_to_fifo %r, %f_mid : tensor<64xf32>, <"SFU" -> "SFU", 64xf32>
        // expected-error @below {{the f32 side of a conversion between f16 and f32 moves as two vectors, so the other end of its FIFO must be a transfer to or from memory or another such conversion}}
        %y = ktdf.read_from_fifo %f_mid : <"SFU" -> "SFU", 64xf32> -> tensor<64xf32>
        "test.use"(%y) : (tensor<64xf32>) -> ()
      }
    }
    return
  }
}

// -----

// A fp32todl16 receives both f32 vectors of its input, so nothing else may
// read that input.
#map = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @narrowed_input_shared() attributes {grid = [1]} {
    %lu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
    %sfu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
    %su = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
    ktdf_lowering.execute_on %lu, %sfu, %su {
      %f_in = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf32>
      %f_out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      ktdf_lowering.execute_on %sfu {
        %x = ktdf.read_from_fifo %f_in : <"L1LU" -> "SFU", 64xf32> -> tensor<64xf32>
        %e = tensor.empty() : tensor<64xf16>
        // expected-error @below {{the input of a fp32todl16 conversion must be read from a FIFO and used by nothing else}}
        %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%x : tensor<64xf32>) outs(%e : tensor<64xf16>) {
        ^bb0(%a: f32, %b: f16):
          %c = arith.truncf %a : f32 to f16
          linalg.yield %c : f16
        } -> tensor<64xf16>
        "test.use"(%x) : (tensor<64xf32>) -> ()
        ktdf.write_to_fifo %r, %f_out : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
      }
    }
    return
  }
}

// -----

// A conversion on buffers is not one this lowers.
#map = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @buffer_conversion() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %lu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
    %sfu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
    %su = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
    ktdf_lowering.execute_on %lu, %sfu, %su {
      ktdf_lowering.execute_on %sfu {
        %bi = builtin.unrealized_conversion_cast %c0 : index to memref<64xf16, "L1">
        %bo = builtin.unrealized_conversion_cast %c128 : index to memref<64xf32, "L1">
        linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%bi : memref<64xf16, "L1">) outs(%bo : memref<64xf32, "L1">) {
        ^bb0(%a: f16, %b: f32):
          // expected-error @below {{a conversion between f16 and f32 must be the only operation of a lane-wise linalg.generic on tensors, converting its one input}}
          %c = arith.extf %a : f16 to f32
          linalg.yield %c : f32
        }
      }
    }
    return
  }
}

// -----

// Nor is one that does not convert each lane into the same lane.
#map = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @reversed_conversion() attributes {grid = [1]} {
    %lu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
    %sfu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
    %su = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
    ktdf_lowering.execute_on %lu, %sfu, %su {
      %f_in = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      %f_out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf32>
      ktdf_lowering.execute_on %sfu {
        %x = ktdf.read_from_fifo %f_in : <"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
        %e = tensor.empty() : tensor<64xf32>
        %r = linalg.generic {indexing_maps = [affine_map<(d0) -> (63 - d0)>, #map], iterator_types = ["parallel"]} ins(%x : tensor<64xf16>) outs(%e : tensor<64xf32>) {
        ^bb0(%a: f16, %b: f32):
          // expected-error @below {{a conversion between f16 and f32 must be the only operation of a lane-wise linalg.generic on tensors, converting its one input}}
          %c = arith.extf %a : f16 to f32
          linalg.yield %c : f32
        } -> tensor<64xf32>
        ktdf.write_to_fifo %r, %f_out : tensor<64xf32>, <"SFU" -> "L1SU", 64xf32>
      }
    }
    return
  }
}

// -----

// A splat is one vector broadcast, so it cannot fill a slot moved as two.
#map = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @splat_into_wide_slot() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %lu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
    %sfu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
    %su = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
    %in = builtin.unrealized_conversion_cast %c0 : index to memref<1x64xf32, "L1">
    ktdf_lowering.execute_on %lu, %sfu, %su {
      %f_in = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf32>
      %f_out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      ktdf_lowering.execute_on %lu {
        // expected-error @below {{the f32 side of a conversion between f16 and f32 moves as two vectors, which a splat or pad transfer does not}}
        ktdf.data_transfer from %in[0, 0] size [1, 8] to %f_in size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<1x64xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf32>
      }
      ktdf_lowering.execute_on %sfu {
        %x = ktdf.read_from_fifo %f_in : <"L1LU" -> "SFU", 64xf32> -> tensor<64xf32>
        %e = tensor.empty() : tensor<64xf16>
        // expected-note @below {{the conversion}}
        %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%x : tensor<64xf32>) outs(%e : tensor<64xf16>) {
        ^bb0(%a: f32, %b: f16):
          %c = arith.truncf %a : f32 to f16
          linalg.yield %c : f16
        } -> tensor<64xf16>
        ktdf.write_to_fifo %r, %f_out : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
      }
    }
    return
  }
}

// -----

// Neither zero-padding mode can fill the two-vector input of fp32todl16.
#map = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @zero_pad_front_into_wide_slot() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %lu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
    %sfu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
    %su = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
    %in = builtin.unrealized_conversion_cast %c0 : index to memref<1x64xf32, "L1">
    ktdf_lowering.execute_on %lu, %sfu, %su {
      %f_in = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf32>
      %f_out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      ktdf_lowering.execute_on %lu {
        // expected-error @below {{the f32 side of a conversion between f16 and f32 moves as two vectors, which a splat or pad transfer does not}}
        ktdf.data_transfer from %in[0, 0] size [1, 8] to %f_in size [64] {transfer_mode = #ktdf.transfer_mode<zero_pad_front>} : memref<1x64xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf32>
      }
      ktdf_lowering.execute_on %sfu {
        %x = ktdf.read_from_fifo %f_in : <"L1LU" -> "SFU", 64xf32> -> tensor<64xf32>
        %e = tensor.empty() : tensor<64xf16>
        // expected-note @below {{the conversion}}
        %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%x : tensor<64xf32>) outs(%e : tensor<64xf16>) {
        ^bb0(%a: f32, %b: f16):
          %c = arith.truncf %a : f32 to f16
          linalg.yield %c : f16
        } -> tensor<64xf16>
        ktdf.write_to_fifo %r, %f_out : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
      }
    }
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @zero_pad_back_into_wide_slot() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %lu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
    %sfu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
    %su = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
    %in = builtin.unrealized_conversion_cast %c0 : index to memref<1x64xf32, "L1">
    ktdf_lowering.execute_on %lu, %sfu, %su {
      %f_in = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf32>
      %f_out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
      ktdf_lowering.execute_on %lu {
        // expected-error @below {{the f32 side of a conversion between f16 and f32 moves as two vectors, which a splat or pad transfer does not}}
        ktdf.data_transfer from %in[0, 0] size [1, 8] to %f_in size [64] {transfer_mode = #ktdf.transfer_mode<zero_pad_back>} : memref<1x64xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf32>
      }
      ktdf_lowering.execute_on %sfu {
        %x = ktdf.read_from_fifo %f_in : <"L1LU" -> "SFU", 64xf32> -> tensor<64xf32>
        %e = tensor.empty() : tensor<64xf16>
        // expected-note @below {{the conversion}}
        %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%x : tensor<64xf32>) outs(%e : tensor<64xf16>) {
        ^bb0(%a: f32, %b: f16):
          %c = arith.truncf %a : f32 to f16
          linalg.yield %c : f16
        } -> tensor<64xf16>
        ktdf.write_to_fifo %r, %f_out : tensor<64xf16>, <"SFU" -> "L1SU", 64xf16>
      }
    }
    return
  }
}

// -----

// A dl16tofp32 sends both f32 vectors of its result, so nothing else may use
// that result. This is reported once.
#map = affine_map<(d0) -> (d0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @widened_result_shared() attributes {grid = [1]} {
    %c0 = arith.constant 0 : index
    %lu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1lu", type = "l1lu"} : index
    %sfu = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-sfu", type = "sfu"} : index
    %su = dataflow.get_unit {core = 0 : i32, corelet = 0 : i32, name = "C0-l1su", type = "l1su"} : index
    ktdf_lowering.execute_on %lu, %sfu, %su {
      %f_out = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf32>
      ktdf_lowering.execute_on %sfu {
        %x = builtin.unrealized_conversion_cast %c0 : index to tensor<64xf16>
        %e = tensor.empty() : tensor<64xf32>
        // expected-error @below {{the widened result of a dl16tofp32 conversion must be written to a FIFO and nothing else}}
        %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%x : tensor<64xf16>) outs(%e : tensor<64xf32>) {
        ^bb0(%a: f16, %b: f32):
          %c = arith.extf %a : f16 to f32
          linalg.yield %c : f32
        } -> tensor<64xf32>
        "test.use"(%r) : (tensor<64xf32>) -> ()
        ktdf.write_to_fifo %r, %f_out : tensor<64xf32>, <"SFU" -> "L1SU", 64xf32>
      }
    }
    return
  }
}
