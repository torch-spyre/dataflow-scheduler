// RUN: dataflow-scheduler-opt --ktir-legality-check -split-input-file -verify-diagnostics %s

// With an orchestration module present, only the program modules still to be
// scheduled are checked. A marked one was scheduled by an earlier run, and is
// skipped whatever loops that run left in it: here an scf.for with iter_args,
// which the rules reject in new KTIR. The new kernel beside it holds a load and
// is legal.

module {
  module {
    func.func @kernel() {
      call @scheduled() : () -> ()
      call @new() : () -> ()
      return
    }
    func.func private @scheduled()
    func.func private @new()
  }
  module @scheduled attributes {dataflow_scheduler.scheduled} {
    func.func @scheduled() {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c1024 = arith.constant 1024 : index
      %v = ktdp.construct_memory_view %c1024, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
      } : memref<1x64xf16>
      %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c0) -> (index) {
        %n = arith.addi %acc, %c1 : index
        scf.yield %n : index
      }
      return
    }
  }
  module @new {
    func.func @new() {
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %v = ktdp.construct_memory_view %c1024, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
      } : memref<1x64xf16>
      %t = ktdp.construct_access_tile %v[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
      %x = ktdp.load %t : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
      return
    }
  }
}

// -----

// An unmarked module is checked in full, a function in it that holds no
// transfer op included.

module {
  module {
    func.func @kernel() {
      call @new() : () -> ()
      return
    }
    func.func private @new()
  }
  module @new {
    func.func @new() {
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %v = ktdp.construct_memory_view %c1024, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
      } : memref<1x64xf16>
      %t = ktdp.construct_access_tile %v[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
      %x = ktdp.load %t : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
      return
    }
    func.func @helper() {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // expected-error @+1 {{V1 does not support scf.while loops}}
      scf.while : () -> () {
        %false = arith.constant false
        scf.condition(%false)
      } do {
        scf.yield
      }
      return
    }
  }
}

// -----

// Only the marker decides whether a module is skipped, not what it holds. A
// marked module is skipped even with a transfer op and an scf.while in it.

module {
  module {
    func.func @kernel() {
      call @prog() : () -> ()
      return
    }
    func.func private @prog()
  }
  module @prog attributes {dataflow_scheduler.scheduled} {
    func.func @prog() {
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %v = ktdp.construct_memory_view %c1024, sizes: [1, 64], strides: [64, 1] {
        coordinate_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>,
        memory_space = #ktdp.memory_space<global>
      } : memref<1x64xf16>
      %t = ktdp.construct_access_tile %v[%c0, %c0] {
        access_tile_set = affine_set<(d0, d1) : (d0 == 0, d1 >= 0, -d1 + 63 >= 0)>,
        access_tile_order = affine_map<(d0, d1) -> (d0, d1)>
      } : memref<1x64xf16> -> !ktdp.access_tile<1x64xindex>
      %x = ktdp.load %t : !ktdp.access_tile<1x64xindex> -> tensor<1x64xf16>
      scf.while : () -> () {
        %false = arith.constant false
        scf.condition(%false)
      } do {
        scf.yield
      }
      return
    }
  }
}

// -----

// An unmarked module is checked as new KTIR even when it holds no transfer op.

module {
  module {
    func.func @kernel() {
      call @prog() : () -> ()
      return
    }
    func.func private @prog()
  }
  module @prog {
    func.func @prog() {
      // expected-error @+1 {{V1 does not support scf.while loops}}
      scf.while : () -> () {
        %false = arith.constant false
        scf.condition(%false)
      } do {
        scf.yield
      }
      return
    }
  }
}

// -----

// Without an orchestration module, every function is checked, as on a first run,
// the marker notwithstanding.

module attributes {dataflow_scheduler.scheduled} {
  func.func @kernel() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    // expected-error @+1 {{V1 does not support scf.for with loop-carried arguments (iter_args)}}
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %c0) -> (index) {
      %n = arith.addi %acc, %c1 : index
      scf.yield %n : index
    }
    return
  }
}
