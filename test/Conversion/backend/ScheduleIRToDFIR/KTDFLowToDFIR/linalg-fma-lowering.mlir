// RUN: dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s | FileCheck %s

// math.fma, and an arith.mulf whose only reader is an arith.addf, lower to one
// vectorchain.multiply_and_accumulate over the lanes. A product that goes
// through a register, or that something else also reads, keeps its multiply.

// CHECK: #[[$ID:.+]] = affine_map<(d0) -> (d0)>

// A product added straight away is one multiply-accumulate.
// CHECK-LABEL: func.func @mul_add(
// CHECK:         dataflow.program_unit
// CHECK:         dataflow.program_unit
// CHECK:         %[[IN:.*]] = dataflow.receive
// CHECK-NOT:     vectorchain.binary
// CHECK:         %[[MAC:.*]] = vectorchain.multiply_and_accumulate %[[IN]], %[[IN]], %[[IN]] {reduction_map = #[[$ID]]} : vector<64xf16>, vector<64xf16>, vector<64xf16>, vector<64xf16>
// CHECK-NOT:     vectorchain.binary
// CHECK:         dataflow.send %{{.*}}, %[[MAC]] : vector<64xf16>

// CHECK-LABEL: func.func @fma(
// CHECK:         dataflow.program_unit
// CHECK:         dataflow.program_unit
// CHECK:         %[[IN:.*]] = dataflow.receive
// CHECK:         %[[MAC:.*]] = vectorchain.multiply_and_accumulate %[[IN]], %[[IN]], %[[IN]] {reduction_map = #[[$ID]]}
// CHECK:         dataflow.send %{{.*}}, %[[MAC]] : vector<64xf16>

// A product stored to a register and read back is added on its own: the
// multiply and the add stay two ops.
// CHECK-LABEL: func.func @mul_through_register(
// CHECK-NOT:     vectorchain.multiply_and_accumulate
// CHECK:         vectorchain.binary {{.*}}binary_operator mul
// CHECK:         agen.vector_store
// CHECK:         agen.vector_load
// CHECK:         vectorchain.binary {{.*}}binary_operator add
// CHECK-NOT:     vectorchain.multiply_and_accumulate
// CHECK:         return

// A product with a second reader keeps its multiply, so neither add fuses.
// CHECK-LABEL: func.func @mul_read_twice(
// CHECK-NOT:     vectorchain.multiply_and_accumulate
// CHECK:         %[[PROD:.*]] = vectorchain.binary {{.*}}binary_operator mul
// CHECK:         %[[SUM:.*]] = vectorchain.binary %[[PROD]], %{{.*}}binary_operator add
// CHECK:         vectorchain.binary %[[SUM]], %[[PROD]] {{.*}}binary_operator add
// CHECK-NOT:     vectorchain.multiply_and_accumulate
// CHECK:         return

// Of two products added together, the first fuses and the second is the addend.
// CHECK-LABEL: func.func @sum_of_products(
// CHECK:         dataflow.program_unit
// CHECK:         dataflow.program_unit
// CHECK:         %[[IN:.*]] = dataflow.receive
// CHECK:         %[[RHS:.*]] = vectorchain.binary %[[IN]], %[[IN]] {{.*}}binary_operator mul
// CHECK:         vectorchain.multiply_and_accumulate %[[IN]], %[[IN]], %[[RHS]]
// CHECK-NOT:     vectorchain.binary
// CHECK:         return

#id = affine_map<(d0) -> (d0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @mul_add() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x64xf16, "L1">
    %in_fifo  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    %out_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %alloc_l1[%c0, %c0] size [1, 64] to %in_fifo size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf_lowering.execute_on %u_sfu {
      %input  = ktdf.read_from_fifo %in_fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %init   = tensor.empty() : tensor<64xf16>
      %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel"]
      } ins(%input : tensor<64xf16>) outs(%init : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %prod = arith.mulf %in, %in : f16
        %sum = arith.addf %in, %prod : f16
        linalg.yield %sum : f16
      } -> tensor<64xf16>
      ktdf.write_to_fifo %result, %out_fifo : tensor<64xf16>, <"SFU" -> "L1LU", 64xf16>
    }
    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %out_fifo size [64] to %alloc_l1[%c0, %c0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>, memref<1x64xf16, "L1">
    }
    return
  }

  func.func @fma() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x64xf16, "L1">
    %in_fifo  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    %out_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %alloc_l1[%c0, %c0] size [1, 64] to %in_fifo size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf_lowering.execute_on %u_sfu {
      %input  = ktdf.read_from_fifo %in_fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %init   = tensor.empty() : tensor<64xf16>
      %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel"]
      } ins(%input : tensor<64xf16>) outs(%init : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %fma = math.fma %in, %in, %in : f16
        linalg.yield %fma : f16
      } -> tensor<64xf16>
      ktdf.write_to_fifo %result, %out_fifo : tensor<64xf16>, <"SFU" -> "L1LU", 64xf16>
    }
    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %out_fifo size [64] to %alloc_l1[%c0, %c0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>, memref<1x64xf16, "L1">
    }
    return
  }

  func.func @mul_through_register() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x64xf16, "L1">
    %in_fifo  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    %out_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %alloc_l1[%c0, %c0] size [1, 64] to %in_fifo size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf_lowering.execute_on %u_sfu {
      %reg = memref.alloc() : memref<64xf16, "SFU_REG">
      %input  = ktdf.read_from_fifo %in_fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %init   = tensor.empty() : tensor<64xf16>
      %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel"]
      } ins(%input : tensor<64xf16>) outs(%init : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %prod = arith.mulf %in, %in : f16
        memref.store %prod, %reg[%c0] : memref<64xf16, "SFU_REG">
        %held = memref.load %reg[%c0] : memref<64xf16, "SFU_REG">
        %sum = arith.addf %held, %in : f16
        linalg.yield %sum : f16
      } -> tensor<64xf16>
      ktdf.write_to_fifo %result, %out_fifo : tensor<64xf16>, <"SFU" -> "L1LU", 64xf16>
    }
    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %out_fifo size [64] to %alloc_l1[%c0, %c0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>, memref<1x64xf16, "L1">
    }
    return
  }

  func.func @mul_read_twice() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x64xf16, "L1">
    %in_fifo  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    %out_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %alloc_l1[%c0, %c0] size [1, 64] to %in_fifo size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf_lowering.execute_on %u_sfu {
      %input  = ktdf.read_from_fifo %in_fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %init   = tensor.empty() : tensor<64xf16>
      %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel"]
      } ins(%input : tensor<64xf16>) outs(%init : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %prod = arith.mulf %in, %in : f16
        %sum = arith.addf %prod, %in : f16
        %again = arith.addf %sum, %prod : f16
        linalg.yield %again : f16
      } -> tensor<64xf16>
      ktdf.write_to_fifo %result, %out_fifo : tensor<64xf16>, <"SFU" -> "L1LU", 64xf16>
    }
    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %out_fifo size [64] to %alloc_l1[%c0, %c0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>, memref<1x64xf16, "L1">
    }
    return
  }

  func.func @sum_of_products() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x64xf16, "L1">
    %in_fifo  = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    %out_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>

    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %alloc_l1[%c0, %c0] size [1, 64] to %in_fifo size [64] : memref<1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    }
    ktdf_lowering.execute_on %u_sfu {
      %input  = ktdf.read_from_fifo %in_fifo : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16> -> tensor<64xf16>
      %init   = tensor.empty() : tensor<64xf16>
      %result = linalg.generic {
        indexing_maps = [#id, #id],
        iterator_types = ["parallel"]
      } ins(%input : tensor<64xf16>) outs(%init : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %lhs = arith.mulf %in, %in : f16
        %rhs = arith.mulf %in, %in : f16
        %sum = arith.addf %lhs, %rhs : f16
        linalg.yield %sum : f16
      } -> tensor<64xf16>
      ktdf.write_to_fifo %result, %out_fifo : tensor<64xf16>, <"SFU" -> "L1LU", 64xf16>
    }
    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %out_fifo size [64] to %alloc_l1[%c0, %c0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>, memref<1x64xf16, "L1">
    }
    return
  }
}
