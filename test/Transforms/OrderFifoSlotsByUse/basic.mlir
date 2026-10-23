// The slots one stage writes and another reads between the same two ends share
// one FIFO, which delivers in the order it was filled, so they are written and
// read in the order the consumer uses them.

// RUN: dataflow-scheduler-opt -order-fifo-slots-by-use %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>

// The consumer copies %b into a register before the generic takes %a, as a
// splat operand is, so %b goes first on both sides.
// CHECK-LABEL:   func.func @copied_before_use(
// CHECK:           ktdf.stage
// CHECK-NEXT:        ktdf.data_transfer from %[[B_MEM:.*]][0] size [32] to %[[P:.*]]#1
// CHECK-NEXT:        ktdf.data_transfer from %[[A_MEM:.*]][0] size [32] to %[[P]]#0
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        %[[B:.*]] = ktdf.read_from_fifo %[[P]]#1
// CHECK-NEXT:        %[[A:.*]] = ktdf.read_from_fifo %[[P]]#0
// CHECK-NEXT:        memref.copy %[[B]], %[[P]]#4
// CHECK-NEXT:        linalg.generic {{.*}} ins(%[[A]], %[[P]]#4 :
func.func @copied_before_use(%a_mem: memref<32xf32, "L1">, %b_mem: memref<32xf32, "L1">) {
  ktdf.pipeline {
    %p:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, memref<32xf32>, memref<32xf32, "SFU_REG">) {
      %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      %t = ktdf.create_token : !ktdf.token
      %out = memref.alloc() : memref<32xf32>
      %reg = memref.alloc() : memref<32xf32, "SFU_REG">
      ktdf.private_yield %slots#0, %slots#1, %t, %out, %reg : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, memref<32xf32>, memref<32xf32, "SFU_REG">
    }
    ktdf.stage depends_in(none) depends_out(%p#2) {
      ktdf.data_transfer from %a_mem[0] size [32] to %p#0 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      ktdf.data_transfer from %b_mem[0] size [32] to %p#1 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(%p#2) depends_out(none) {
      %a = ktdf.read_from_fifo %p#0 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      %b = ktdf.read_from_fifo %p#1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      memref.copy %b, %p#4 : memref<32xf32> to memref<32xf32, "SFU_REG">
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%a, %p#4 : memref<32xf32>, memref<32xf32, "SFU_REG">) outs(%p#3 : memref<32xf32>) {
      ^bb0(%in: f32, %in_b: f32, %o: f32):
        %r = arith.addf %in, %in_b : f32
        linalg.yield %r : f32
      }
    } {applicable_units = ["SFU"]}
  }
  return
}

// The generic takes %a and %b in that order, but its body uses %b first.
// CHECK-LABEL:   func.func @used_first_in_body(
// CHECK:           ktdf.stage
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P:.*]]#1
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P]]#0
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#1
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#0
func.func @used_first_in_body(%a_mem: memref<32xf32, "L1">, %b_mem: memref<32xf32, "L1">) {
  ktdf.pipeline {
    %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, memref<32xf32>) {
      %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      %t = ktdf.create_token : !ktdf.token
      %out = memref.alloc() : memref<32xf32>
      ktdf.private_yield %slots#0, %slots#1, %t, %out : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, memref<32xf32>
    }
    ktdf.stage depends_in(none) depends_out(%p#2) {
      ktdf.data_transfer from %a_mem[0] size [32] to %p#0 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      ktdf.data_transfer from %b_mem[0] size [32] to %p#1 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(%p#2) depends_out(none) {
      %a = ktdf.read_from_fifo %p#0 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      %b = ktdf.read_from_fifo %p#1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%a, %b : memref<32xf32>, memref<32xf32>) outs(%p#3 : memref<32xf32>) {
      ^bb0(%in: f32, %in_b: f32, %o: f32):
        %sq = arith.mulf %in_b, %in_b : f32
        %r = arith.addf %in, %sq : f32
        linalg.yield %r : f32
      }
    } {applicable_units = ["SFU"]}
  }
  return
}

// Already written and read in the order they are used: left alone.
// CHECK-LABEL:   func.func @in_order(
// CHECK:           ktdf.stage
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P:.*]]#0
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P]]#1
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#0
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#1
func.func @in_order(%a_mem: memref<32xf32, "L1">, %b_mem: memref<32xf32, "L1">) {
  ktdf.pipeline {
    %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, memref<32xf32>) {
      %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      %t = ktdf.create_token : !ktdf.token
      %out = memref.alloc() : memref<32xf32>
      ktdf.private_yield %slots#0, %slots#1, %t, %out : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, memref<32xf32>
    }
    ktdf.stage depends_in(none) depends_out(%p#2) {
      ktdf.data_transfer from %a_mem[0] size [32] to %p#0 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      ktdf.data_transfer from %b_mem[0] size [32] to %p#1 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(%p#2) depends_out(none) {
      %a = ktdf.read_from_fifo %p#0 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      %b = ktdf.read_from_fifo %p#1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%a, %b : memref<32xf32>, memref<32xf32>) outs(%p#3 : memref<32xf32>) {
      ^bb0(%in: f32, %in_b: f32, %o: f32):
        %r = arith.addf %in, %in_b : f32
        linalg.yield %r : f32
      }
    } {applicable_units = ["SFU"]}
  }
  return
}

// A store between the writes keeps them where they are, and with them the
// reads.
// CHECK-LABEL:   func.func @memory_op_between_writes(
// CHECK:           ktdf.stage
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P:.*]]#0
// CHECK-NEXT:        memref.store
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P]]#1
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#0
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#1
func.func @memory_op_between_writes(%a_mem: memref<32xf32, "L1">, %b_mem: memref<32xf32, "L1">, %v: f32) {
  %c0 = arith.constant 0 : index
  ktdf.pipeline {
    %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, memref<32xf32>) {
      %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      %t = ktdf.create_token : !ktdf.token
      %out = memref.alloc() : memref<32xf32>
      ktdf.private_yield %slots#0, %slots#1, %t, %out : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, memref<32xf32>
    }
    ktdf.stage depends_in(none) depends_out(%p#2) {
      ktdf.data_transfer from %a_mem[0] size [32] to %p#0 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      memref.store %v, %b_mem[%c0] : memref<32xf32, "L1">
      ktdf.data_transfer from %b_mem[0] size [32] to %p#1 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(%p#2) depends_out(none) {
      %a = ktdf.read_from_fifo %p#0 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      %b = ktdf.read_from_fifo %p#1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%a, %b : memref<32xf32>, memref<32xf32>) outs(%p#3 : memref<32xf32>) {
      ^bb0(%in: f32, %in_b: f32, %o: f32):
        %sq = arith.mulf %in_b, %in_b : f32
        %r = arith.addf %in, %sq : f32
        linalg.yield %r : f32
      }
    } {applicable_units = ["SFU"]}
  }
  return
}
