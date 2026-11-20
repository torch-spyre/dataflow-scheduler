// A constant the kernel writes, such as the 10.0 in x + 10.0, carries no
// ktdf_arch.maps_to: no device pattern placed it. The pass maps it to the memory
// local to the compute the generic runs on, and from there it gets a register
// the way a mapped constant does.

// RUN: dataflow-scheduler-opt %s -split-input-file -verify-diagnostics \
// RUN:   -materialize-registers | FileCheck %s
// RUN: dataflow-scheduler-opt %s -split-input-file -verify-diagnostics \
// RUN:   -materialize-registers -hoist-constant-storage \
// RUN:   | FileCheck %s --check-prefix=HOIST

// Arithmetic reads the constant: it goes in a register in SFU_REG, the memory
// local to the SFU, and the add reads a lane out of it.

// CHECK-LABEL: func.func @constant_read_by_arithmetic(
// CHECK:        %[[C:.*]] = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 1.000000e+01 : f16
// CHECK-NEXT:   %[[REG:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:   linalg.fill ins(%[[C]] : f16) outs(%[[REG]] : memref<64xf16, "SFU_REG">)
// CHECK-NEXT:   linalg.generic
// CHECK-NEXT:   ^bb0(%[[X:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:     %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[V:.*]] = memref.load %[[REG]][%[[ZERO]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:     %[[SUM:.*]] = arith.addf %[[X]], %[[V]] : f16
// CHECK-NEXT:     linalg.yield %[[SUM]] : f16

#id = affine_map<(d0) -> (d0)>
module {
  func.func @constant_read_by_arithmetic(%in: tensor<64xf16>) -> tensor<64xf16> {
    %init = tensor.empty() : tensor<64xf16>
    %c = arith.constant 1.000000e+01 : f16
    %result = linalg.generic {indexing_maps = [#id, #id],
                              iterator_types = ["parallel"]}
        ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
    ^bb0(%x: f16, %out: f16):
      %sum = arith.addf %x, %c : f16
      linalg.yield %sum : f16
    } -> tensor<64xf16>
    return %result : tensor<64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}

// -----

// A template call reads the constant: it is handed the register itself.

// CHECK-LABEL: func.func @constant_read_by_a_template(
// CHECK:        %[[C:.*]] = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 2.000000e+00 : f16
// CHECK-NEXT:   %[[REG:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:   linalg.fill ins(%[[C]] : f16) outs(%[[REG]] : memref<64xf16, "SFU_REG">)
// CHECK-NEXT:   %[[OUT:.*]] = memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:   %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:   linalg.generic
// CHECK-NEXT:   ^bb0(%{{.*}}: f16, %{{.*}}: f16):
// CHECK-NEXT:     ktdf.opaque "fake_template"
// CHECK-NEXT:       ins(%[[REG]]: memref<64xf16, "SFU_REG">)
// CHECK-NEXT:       outs(%[[OUT]]: memref<64xf16, "SFU_REG">)
// CHECK-NEXT:     %[[V:.*]] = memref.load %[[OUT]][%[[ZERO]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:     linalg.yield %[[V]] : f16

#id = affine_map<(d0) -> (d0)>
module {
  func.func @constant_read_by_a_template(%in: tensor<64xf16>) -> tensor<64xf16> {
    %init = tensor.empty() : tensor<64xf16>
    %c = arith.constant 2.000000e+00 : f16
    %result = linalg.generic {indexing_maps = [#id, #id],
                              iterator_types = ["parallel"]}
        ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
    ^bb0(%x: f16, %out: f16):
      %reg = memref.alloca() : memref<f16, "SFU_REG">
      ktdf.opaque "fake_template" ins(%c : f16) outs(%reg : memref<f16, "SFU_REG">)
      %v = memref.load %reg[] : memref<f16, "SFU_REG">
      linalg.yield %v : f16
    } -> tensor<64xf16>
    return %result : tensor<64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}

// -----

// An index addresses memory rather than being computed on, so an index constant
// stays a scalar.

// CHECK-LABEL: func.func @index_constant_left_alone(
// CHECK-SAME:      %{{.*}}: tensor<64xf16>, %[[TABLE:.*]]: memref<65xf16, "L1">)
// CHECK-NEXT:   %{{.*}} = tensor.empty() : tensor<64xf16>
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   linalg.generic
// CHECK-NEXT:   ^bb0(%{{.*}}: f16, %{{.*}}: f16):
// CHECK-NEXT:     %[[I:.*]] = linalg.index 0 : index
// CHECK-NEXT:     %[[NEXT:.*]] = arith.addi %[[I]], %[[C1]] : index
// CHECK-NEXT:     %[[V:.*]] = memref.load %[[TABLE]][%[[NEXT]]] : memref<65xf16, "L1">
// CHECK-NEXT:     linalg.yield %[[V]] : f16
// CHECK-NOT:    memref.alloc

#id = affine_map<(d0) -> (d0)>
module {
  func.func @index_constant_left_alone(%in: tensor<64xf16>, %table: memref<65xf16, "L1">) -> tensor<64xf16> {
    %init = tensor.empty() : tensor<64xf16>
    %c1 = arith.constant 1 : index
    %result = linalg.generic {indexing_maps = [#id, #id],
                              iterator_types = ["parallel"]}
        ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
    ^bb0(%x: f16, %out: f16):
      %i = linalg.index 0 : index
      %next = arith.addi %i, %c1 : index
      %v = memref.load %table[%next] : memref<65xf16, "L1">
      linalg.yield %v : f16
    } -> tensor<64xf16>
    return %result : tensor<64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}

// -----

// One constant read by arithmetic and stored into a register of the body. The
// add gets a register of its own and reads a lane of it. The store keeps the
// scalar, which fills the alloca it was stored into.
//
// Only the add's register is filled once, outside the loop. The alloca stays in
// the stage, so its fill runs on every iteration.

// CHECK-LABEL: func.func @constant_read_and_stored(
// CHECK:          %[[C:.*]] = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 2.000000e+00 : f16
// CHECK:          ktdf.stage
// CHECK-NEXT:       tensor.empty
// CHECK-NEXT:       %[[C_REG:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:       linalg.fill ins(%[[C]] : f16) outs(%[[C_REG]] : memref<64xf16, "SFU_REG">)
// CHECK-NEXT:       %[[LATCH:.*]] = memref.alloca() : memref<64xf16, "SFU_REG">
// CHECK-NEXT:       %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:       linalg.fill ins(%[[C]] : f16) outs(%[[LATCH]] : memref<64xf16, "SFU_REG">)
// CHECK-NEXT:       linalg.generic
// CHECK-NEXT:       ^bb0(%[[X:.*]]: f16, %{{.*}}: f16):
// CHECK-NEXT:         %[[LATCHED:.*]] = memref.load %[[LATCH]][%[[ZERO]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:         %[[LANE:.*]] = arith.constant 0 : index
// CHECK-NEXT:         %[[V:.*]] = memref.load %[[C_REG]][%[[LANE]]] : memref<64xf16, "SFU_REG">
// CHECK-NEXT:         %[[SUM:.*]] = arith.addf %[[X]], %[[V]] : f16
// CHECK-NEXT:         %[[MAX:.*]] = arith.maximumf %[[SUM]], %[[LATCHED]] : f16
// CHECK-NEXT:         linalg.yield %[[MAX]] : f16

// HOIST-LABEL: func.func @constant_read_and_stored(
// HOIST:         %[[C:.*]] = arith.constant {ktdf_arch.maps_to = "SFU_REG"} 2.000000e+00 : f16
// HOIST:         %[[C_REG:.*]] = memref.alloc() : memref<64xf16, "SFU_REG">
// HOIST-NEXT:    ktdf.pipeline {
// HOIST-NEXT:      ktdf.stage
// HOIST-NEXT:        linalg.fill ins(%[[C]] : f16) outs(%[[C_REG]] : memref<64xf16, "SFU_REG">)
// HOIST:         scf.for
// HOIST-NEXT:      ktdf.pipeline {
// HOIST:             ktdf.stage
// HOIST-NEXT:          %[[LATCH:.*]] = memref.alloca() : memref<64xf16, "SFU_REG">
// HOIST-NEXT:          linalg.fill ins(%[[C]] : f16) outs(%[[LATCH]] : memref<64xf16, "SFU_REG">)
// HOIST-NEXT:          linalg.generic

#id = affine_map<(d0) -> (d0)>
module {
  func.func @constant_read_and_stored(%in: tensor<64xf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c = arith.constant 2.000000e+00 : f16
    scf.for %i = %c0 to %c4 step %c1 {
      ktdf.pipeline {
        %token = ktdf.private -> (!ktdf.token) {
          %t = ktdf.create_token : !ktdf.token
          ktdf.private_yield %t : !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%token) {
          %init = tensor.empty() : tensor<64xf16>
          %result = linalg.generic {indexing_maps = [#id, #id],
                                    iterator_types = ["parallel"]}
              ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>) {
          ^bb0(%x: f16, %out: f16):
            %latch = memref.alloca() : memref<f16, "SFU_REG">
            memref.store %c, %latch[] : memref<f16, "SFU_REG">
            %latched = memref.load %latch[] : memref<f16, "SFU_REG">
            %sum = arith.addf %x, %c : f16
            %max = arith.maximumf %sum, %latched : f16
            linalg.yield %max : f16
          } -> tensor<64xf16>
        } {applicable_units = ["SFU"]}
      }
    }
    return
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}

// -----

// The MPE array has no local memory of its own, so there is nowhere to put the
// register.

#id = affine_map<(d0) -> (d0)>
module {
  func.func @no_local_memory(%in: tensor<64xf16>) -> tensor<64xf16> {
    %init = tensor.empty() : tensor<64xf16>
    %c = arith.constant 1.000000e+01 : f16
    // expected-error@+1 {{no unambiguous local memory of "MPE_ARRAY" to hold a register for a constant}}
    %result = linalg.generic {indexing_maps = [#id, #id],
                              iterator_types = ["parallel"]}
        ins(%in : tensor<64xf16>) outs(%init : tensor<64xf16>)
        attrs = {ktdf_arch.maps_to = "MPE_ARRAY"} {
    ^bb0(%x: f16, %out: f16):
      %sum = arith.addf %x, %c : f16
      linalg.yield %sum : f16
    } -> tensor<64xf16>
    return %result : tensor<64xf16>
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
