// RUN: dataflow-scheduler-opt -split-input-file -pass-pipeline="builtin.module(ktdflowering-to-dfir)" -verify-diagnostics %s

// What an address over several symbols, or over a kernel's loop variable, still
// cannot be, and what a kernel cannot pass its programs. Each is reported where
// it is computed rather than left for the run to resolve to the wrong address.

// A run input as the base, and the loop variable reaching the address through
// the access tile, so as the offset of the reinterpret_cast rather than in the
// base. Not supported: the offset would have to be a number on every grid
// element.

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index) attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      scf.for %i = %c0 to %c2 step %c1 {
        func.call @sched_0(%A, %i) : (index, index) -> ()
      }
      return
    }
    func.func private @sched_0(index, index)
  }
  module @sched_0 {
    func.func private @sched_0(%base: index, %i: index) attributes {grid = [2]} {
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %6 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %7 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c4096 = arith.constant 4096 : index
      %c64000 = arith.constant 64000 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        %tilemap = uniform.def_immutable_mapping([%0 -> %c0], [%1 -> %c1]):index
        %tile = uniform.query_map(map:%tilemap, key:%arg0) : index
        %slab = arith.muli %tile, %c6 : index
        %step = arith.muli %i, %c3 : index
        %row = arith.addi %slab, %step : index
        // expected-error @below {{a start address from a run input, displaced by an offset that depends on a run input, a loop variable or anything else only known while the program runs, is not supported}}
        %off = arith.muli %row, %c4096 : index
        %v0 = ktdp.construct_memory_view %base, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc0 = memref.memory_space_cast %v0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc0 = memref.reinterpret_cast %msc0 to offset: [%off], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %r = %c0 to %c12 step %c1 {
          scf.for %s = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %rc0[%r, %c0, %s, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1 : memref<12x1x64x64xf16, "L1">
      }
      dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%6, %7) : {
      }
      return
    }
  }
}

// -----

// A loop variable of the program itself, rather than of the kernel. Nothing
// stands for it: it exists only while the program runs.

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index) attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      scf.for %i = %c0 to %c2 step %c1 {
        func.call @sched_0(%A, %i) : (index, index) -> ()
      }
      return
    }
    func.func private @sched_0(index, index)
  }
  module @sched_0 {
    func.func private @sched_0(%base: index, %i: index) attributes {grid = [2]} {
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %6 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %7 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c4096 = arith.constant 4096 : index
      %c64000 = arith.constant 64000 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        scf.for %k = %c0 to %c2 step %c1 {
        // expected-error @below {{a value computed from a block argument no symbol stands for, such as a loop variable of a program, cannot be rebuilt as a symbol expression}}
        %b = arith.addi %base, %k : index
        %v0 = ktdp.construct_memory_view %b, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc0 = memref.memory_space_cast %v0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc0 = memref.reinterpret_cast %msc0 to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %r = %c0 to %c12 step %c1 {
          scf.for %s = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %rc0[%r, %c0, %s, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1 : memref<12x1x64x64xf16, "L1">
        }
      }
      dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%6, %7) : {
      }
      return
    }
  }
}

// -----

// The kernel passes a value it computed with an operator a definition cannot be
// written with. Reported where it is computed: the program's side could only
// see an argument with nothing below it.

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index) attributes {grid = [2]} {
      %c4096 = arith.constant 4096 : index
      // expected-error @below {{@sched_0 is passed a value computed by 'arith.maxsi', which a symbol's definition cannot be written with}}
      %aligned = arith.maxsi %A, %c4096 : index
      call @sched_0(%aligned, %A) : (index, index) -> ()
      return
    }
    func.func private @sched_0(index, index)
  }
  module @sched_0 {
    func.func private @sched_0(%base: index, %i: index) attributes {grid = [2]} {
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %6 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %7 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c4096 = arith.constant 4096 : index
      %c64000 = arith.constant 64000 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        %v0 = ktdp.construct_memory_view %base, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc0 = memref.memory_space_cast %v0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc0 = memref.reinterpret_cast %msc0 to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %r = %c0 to %c12 step %c1 {
          scf.for %s = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %rc0[%r, %c0, %s, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1 : memref<12x1x64x64xf16, "L1">
      }
      dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%6, %7) : {
      }
      return
    }
  }
}

// -----

// The kernel passes a value it read from memory: not computed from its
// arguments and loop variables at all.

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index) attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %buf = memref.alloc() : memref<1xindex>
      // expected-error @below {{a value computed by 'memref.load' cannot be rebuilt as a symbol expression}}
      %loaded = memref.load %buf[%c0] : memref<1xindex>
      call @sched_0(%loaded, %A) : (index, index) -> ()
      return
    }
    func.func private @sched_0(index, index)
  }
  module @sched_0 {
    func.func private @sched_0(%base: index, %i: index) attributes {grid = [2]} {
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %6 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %7 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c4096 = arith.constant 4096 : index
      %c64000 = arith.constant 64000 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        %v0 = ktdp.construct_memory_view %base, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc0 = memref.memory_space_cast %v0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc0 = memref.reinterpret_cast %msc0 to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %r = %c0 to %c12 step %c1 {
          scf.for %s = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %rc0[%r, %c0, %s, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1 : memref<12x1x64x64xf16, "L1">
      }
      dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%6, %7) : {
      }
      return
    }
  }
}

// -----

// The kernel passes a value computed from a per-grid query: a kernel is not
// run per grid element, so the query has no one value there.

#set = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 11 >= 0, d1 >= 0, -d1 >= 0, d2 >= 0, -d2 + 63 >= 0, d3 >= 0, -d3 + 63 >= 0)>
module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")
  module {
    func.func @test(%A: index) attributes {grid = [2]} {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %map = uniform.def_immutable_mapping([%c0 -> %c64]) : index
      // expected-error @below {{@sched_0 is passed a value computed from a per-grid query, which has no single value in a kernel}}
      %q = uniform.query_map(map:%map, key:%c0) : index
      %b = arith.addi %A, %q : index
      call @sched_0(%b, %A) : (index, index) -> ()
      return
    }
    func.func private @sched_0(index, index)
  }
  module @sched_0 {
    func.func private @sched_0(%base: index, %i: index) attributes {grid = [2]} {
      %0 = dataflow.get_unit {core = 0 : i32, name = "C0-MNILU", type = "MNILU"} : index
      %1 = dataflow.get_unit {core = 1 : i32, name = "C1-MNILU", type = "MNILU"} : index
      %2 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
      %3 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
      %4 = dataflow.get_unit {core = 0 : i32, name = "C0-SFU", type = "SFU"} : index
      %5 = dataflow.get_unit {core = 1 : i32, name = "C1-SFU", type = "SFU"} : index
      %6 = dataflow.get_unit {core = 0 : i32, name = "C0-L1SU", type = "L1SU"} : index
      %7 = dataflow.get_unit {core = 1 : i32, name = "C1-L1SU", type = "L1SU"} : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c6 = arith.constant 6 : index
      %c12 = arith.constant 12 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c4096 = arith.constant 4096 : index
      %c64000 = arith.constant 64000 : index
      dataflow.program_unit iter_arg : %arg0 -> (%0, %1) : {
        %v0 = ktdp.construct_memory_view %base, sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<12x1x64x64xf16>
        %msc0 = memref.memory_space_cast %v0 : memref<12x1x64x64xf16> to memref<12x1x64x64xf16, "DDR">
        %rc0 = memref.reinterpret_cast %msc0 to offset: [%c0], sizes: [12, 1, 64, 64], strides: [4096, 4096, 64, 1] : memref<12x1x64x64xf16, "DDR"> to memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">
        %l1 = builtin.unrealized_conversion_cast %c128 : index to memref<12x1x64x64xf16, "L1">
        scf.for %r = %c0 to %c12 step %c1 {
          scf.for %s = %c0 to %c64 step %c1 {
            ktdf.data_transfer from %rc0[%r, %c0, %s, %c0] size [1, 1, 1, 64] to %l1[%c0, %c0, %c0, %c0] size [1, 1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<12x1x64x64xf16, strided<[4096, 4096, 64, 1], offset: ?>, "DDR">, memref<12x1x64x64xf16, "L1">
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {loop_type = #ktdf.loop_type<parallel_loop>}
        memref.dealloc %l1 : memref<12x1x64x64xf16, "L1">
      }
      dataflow.program_unit iter_arg : %arg0 -> (%2, %3) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%4, %5) : {
      }
      dataflow.program_unit iter_arg : %arg0 -> (%6, %7) : {
      }
      return
    }
  }
}
