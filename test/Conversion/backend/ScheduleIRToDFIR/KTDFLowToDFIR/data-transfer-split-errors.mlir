// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -verify-diagnostics -split-input-file %s

// An AGEN transfer moves one hardware vector per time step. Requests wider than
// that walk their non-unit outer dims (plus a split innermost dim, if needed)
// as AGEN time dimensions, but only when the shape allows an exact split.
// Anything else is rejected rather than silently mis-lowered.
// The vector width here is 64 lanes.

// Splitting requires the innermost source/destination size to be a multiple of
// the vector width; a 96-lane innermost dim is not a multiple of 64 and cannot
// be covered by whole 64-lane vectors.
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @innermost_size_not_multiple_of_vector_width() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    %ddr_buf = memref.alloc() : memref<2x128x96xf16, "DDR">
    %l1_buf = memref.alloc() : memref<2x1x128x96xf16, "L1">
    ktdf_lowering.execute_on %unit {
      // expected-error @below {{data transfer of 12288 elements exceeds the hardware vector width of 64; splitting requires the innermost source and destination sizes to be a multiple of the vector width, but they are 96 and 96}}
      ktdf.data_transfer from %ddr_buf[%c0, 0, 0] size [1, 128, 96]
                         to %l1_buf[%c0, 0, 0, 0] size [1, 1, 128, 96]
        : memref<2x128x96xf16, "DDR">, memref<2x1x128x96xf16, "L1">
    }
    return
  }
}

// -----

// The ordered walked dimensions must agree between the two sides, otherwise
// there is no single loop nest that walks both in lockstep.
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  func.func @outer_sizes_mismatch() attributes {grid = [2]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
    %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
    %map = uniform.def_immutable_mapping([%c0 -> %0], [%c1 -> %1]):index
    %tile_id = ktdp.get_compute_tile_id : index
    %unit = uniform.query_map(map:%map, key:%tile_id) : index
    %ddr_buf = memref.alloc() : memref<2x256x64xf16, "DDR">
    %l1_buf = memref.alloc() : memref<2x128x64xf16, "L1">
    ktdf_lowering.execute_on %unit {
      // expected-error @below {{source and destination walked dimensions must match to split a transfer of 16384 elements across multiple vectors}}
      ktdf.data_transfer from %ddr_buf[%c0, 0, 0] size [1, 256, 64]
                         to %l1_buf[0, 0, 0] size [2, 128, 64]
        : memref<2x256x64xf16, "DDR">, memref<2x128x64xf16, "L1">
    }
    return
  }
}
