// A FIFO is reordered only when the order it delivers in is known: one block
// writes all of its slots, a block of another stage reads all of them, and no
// other op of the function takes a slot between the same ends. In every case
// below the consumer uses %b first, so the slots would otherwise be swapped.

// RUN: dataflow-scheduler-opt -order-fifo-slots-by-use %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>

// The slots are written by two stages not ordered with each other, so which
// value enters the FIFO first is up to which stage runs first.
// CHECK-LABEL:   func.func @dag_branches(
// CHECK:           ktdf.stage
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P:.*]]#0
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P]]#1
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#0
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#1
func.func @dag_branches(%a_mem: memref<32xf32, "L1">, %b_mem: memref<32xf32, "L1">) {
  ktdf.pipeline {
    %p:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, !ktdf.token, memref<32xf32>) {
      %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      %out = memref.alloc() : memref<32xf32>
      ktdf.private_yield %slots#0, %slots#1, %t0, %t1, %out : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, !ktdf.token, memref<32xf32>
    }
    ktdf.stage depends_in(none) depends_out(%p#2) {
      ktdf.data_transfer from %a_mem[0] size [32] to %p#0 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(none) depends_out(%p#3) {
      ktdf.data_transfer from %b_mem[0] size [32] to %p#1 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(%p#2, %p#3) depends_out(none) {
      %a = ktdf.read_from_fifo %p#0 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      %b = ktdf.read_from_fifo %p#1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%a, %b : memref<32xf32>, memref<32xf32>) outs(%p#4 : memref<32xf32>) {
      ^bb0(%in: f32, %in_b: f32, %o: f32):
        %sq = arith.mulf %in_b, %in_b : f32
        %r = arith.addf %in, %sq : f32
        linalg.yield %r : f32
      }
    } {applicable_units = ["SFU"]}
  }
  return
}

// A third slot between the same ends, written by another stage, is read
// between the two. Moving the read of %b above it would make the consumer take
// the values out of the order the FIFO delivers them in.
// CHECK-LABEL:   func.func @other_read_between(
// CHECK:           ktdf.stage
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P:.*]]#0
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P]]#1
// CHECK-NEXT:      }
// CHECK:           ktdf.stage depends_in(%{{.*}}) depends_out(none)
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#0
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#2
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#1
func.func @other_read_between(%a_mem: memref<32xf32, "L1">, %b_mem: memref<32xf32, "L1">, %x_mem: memref<32xf32, "L1">, %x_out: memref<32xf32, "L1">) {
  ktdf.pipeline {
    %p:6 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, !ktdf.token, memref<32xf32>) {
      %slots:3 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      %t0 = ktdf.create_token : !ktdf.token
      %t1 = ktdf.create_token : !ktdf.token
      %out = memref.alloc() : memref<32xf32>
      ktdf.private_yield %slots#0, %slots#1, %slots#2, %t0, %t1, %out : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, !ktdf.token, memref<32xf32>
    }
    ktdf.stage depends_in(none) depends_out(%p#3) {
      ktdf.data_transfer from %a_mem[0] size [32] to %p#0 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      ktdf.data_transfer from %b_mem[0] size [32] to %p#1 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(%p#3) depends_out(%p#4) {
      ktdf.data_transfer from %x_mem[0] size [32] to %p#2 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(%p#4) depends_out(none) {
      %a = ktdf.read_from_fifo %p#0 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      %x = ktdf.read_from_fifo %p#2 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      %b = ktdf.read_from_fifo %p#1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      memref.copy %x, %x_out : memref<32xf32> to memref<32xf32, "L1">
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%a, %b : memref<32xf32>, memref<32xf32>) outs(%p#5 : memref<32xf32>) {
      ^bb0(%in: f32, %in_b: f32, %o: f32):
        %sq = arith.mulf %in_b, %in_b : f32
        %r = arith.addf %in, %sq : f32
        linalg.yield %r : f32
      }
    } {applicable_units = ["SFU"]}
  }
  return
}

// A nested pipeline sends between the same ends, so its values share the FIFO
// with those of the outer one.
// CHECK-LABEL:   func.func @nested_pipeline_same_ends(
// CHECK:           ktdf.stage
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P:.*]]#0
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P]]#1
// CHECK-NEXT:        scf.for
// CHECK:           ktdf.stage depends_in(%[[P]]#2) depends_out(none)
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#0
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#1
func.func @nested_pipeline_same_ends(%a_mem: memref<32xf32, "L1">, %b_mem: memref<32xf32, "L1">, %x_mem: memref<32xf32, "L1">, %x_out: memref<32xf32, "L1">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
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
      scf.for %i = %c0 to %c2 step %c1 {
        ktdf.pipeline {
          %q:2 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token) {
            %inner = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
            %u = ktdf.create_token : !ktdf.token
            ktdf.private_yield %inner, %u : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token
          }
          ktdf.stage depends_in(none) depends_out(%q#1) {
            ktdf.data_transfer from %x_mem[0] size [32] to %q#0 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
          } {applicable_units = ["L1LU"]}
          ktdf.stage depends_in(%q#1) depends_out(none) {
            %x = ktdf.read_from_fifo %q#0 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
            memref.copy %x, %x_out : memref<32xf32> to memref<32xf32, "L1">
          } {applicable_units = ["SFU"]}
        }
      }
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

// A slot between the same ends is read twice, so it is not one of the slots the
// pass orders, yet it goes through the same FIFO.
// CHECK-LABEL:   func.func @rejected_slot_same_ends(
// CHECK:           ktdf.stage
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P:.*]]#0
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P]]#1
// CHECK-NEXT:        ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P]]#2
// CHECK-NEXT:      }
// CHECK-NEXT:      ktdf.stage
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#0
// CHECK-NEXT:        ktdf.read_from_fifo %[[P]]#1
func.func @rejected_slot_same_ends(%a_mem: memref<32xf32, "L1">, %b_mem: memref<32xf32, "L1">, %x_mem: memref<32xf32, "L1">, %x_out: memref<32xf32, "L1">) {
  ktdf.pipeline {
    %p:5 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, memref<32xf32>) {
      %slots:3 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      %t = ktdf.create_token : !ktdf.token
      %out = memref.alloc() : memref<32xf32>
      ktdf.private_yield %slots#0, %slots#1, %slots#2, %t, %out : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, memref<32xf32>
    }
    ktdf.stage depends_in(none) depends_out(%p#3) {
      ktdf.data_transfer from %a_mem[0] size [32] to %p#0 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      ktdf.data_transfer from %b_mem[0] size [32] to %p#1 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      ktdf.data_transfer from %x_mem[0] size [32] to %p#2 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(%p#3) depends_out(none) {
      %a = ktdf.read_from_fifo %p#0 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      %b = ktdf.read_from_fifo %p#1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      %x0 = ktdf.read_from_fifo %p#2 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      %x1 = ktdf.read_from_fifo %p#2 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
      memref.copy %x0, %x_out : memref<32xf32> to memref<32xf32, "L1">
      memref.copy %x1, %x_out : memref<32xf32> to memref<32xf32, "L1">
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%a, %b : memref<32xf32>, memref<32xf32>) outs(%p#4 : memref<32xf32>) {
      ^bb0(%in: f32, %in_b: f32, %o: f32):
        %sq = arith.mulf %in_b, %in_b : f32
        %r = arith.addf %in, %sq : f32
        linalg.yield %r : f32
      }
    } {applicable_units = ["SFU"]}
  }
  return
}

// One stage both writes and reads the slots.
// CHECK-LABEL:   func.func @same_stage(
// CHECK:           ktdf.stage
// CHECK-NEXT:        ktdf.write_to_fifo %{{.*}}, %[[P:.*]]#0
// CHECK-NEXT:        ktdf.write_to_fifo %{{.*}}, %[[P]]#1
// CHECK-NEXT:        scf.for
// CHECK-NEXT:          ktdf.read_from_fifo %[[P]]#0
// CHECK-NEXT:          ktdf.read_from_fifo %[[P]]#1
func.func @same_stage(%a_mem: memref<32xf32>, %b_mem: memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  ktdf.pipeline {
    %p:3 = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"SFU" -> "SFU", 32xf32>, memref<32xf32>) {
      %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"SFU" -> "SFU", 32xf32>
      %out = memref.alloc() : memref<32xf32>
      ktdf.private_yield %slots#0, %slots#1, %out : !ktdf.fifo.slot<"SFU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"SFU" -> "SFU", 32xf32>, memref<32xf32>
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.write_to_fifo %a_mem, %p#0 : memref<32xf32>, !ktdf.fifo.slot<"SFU" -> "SFU", 32xf32>
      ktdf.write_to_fifo %b_mem, %p#1 : memref<32xf32>, !ktdf.fifo.slot<"SFU" -> "SFU", 32xf32>
      scf.for %i = %c0 to %c1 step %c1 {
        %a = ktdf.read_from_fifo %p#0 : !ktdf.fifo.slot<"SFU" -> "SFU", 32xf32> -> memref<32xf32>
        %b = ktdf.read_from_fifo %p#1 : !ktdf.fifo.slot<"SFU" -> "SFU", 32xf32> -> memref<32xf32>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%a, %b : memref<32xf32>, memref<32xf32>) outs(%p#2 : memref<32xf32>) {
        ^bb0(%in: f32, %in_b: f32, %o: f32):
          %sq = arith.mulf %in_b, %in_b : f32
          %r = arith.addf %in, %sq : f32
          linalg.yield %r : f32
        }
      }
    } {applicable_units = ["SFU"]}
  }
  return
}

// Writes and reads in loops of their stages run once per iteration, on both
// ends alike, so the FIFO is still reordered.
// CHECK-LABEL:   func.func @in_loops(
// CHECK:           ktdf.stage
// CHECK-NEXT:        scf.for
// CHECK-NEXT:          ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P:.*]]#1
// CHECK-NEXT:          ktdf.data_transfer from %{{.*}}[0] size [32] to %[[P]]#0
// CHECK:           ktdf.stage
// CHECK-NEXT:        scf.for
// CHECK-NEXT:          ktdf.read_from_fifo %[[P]]#1
// CHECK-NEXT:          ktdf.read_from_fifo %[[P]]#0
func.func @in_loops(%a_mem: memref<32xf32, "L1">, %b_mem: memref<32xf32, "L1">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  ktdf.pipeline {
    %p:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, memref<32xf32>) {
      %slots:2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      %t = ktdf.create_token : !ktdf.token
      %out = memref.alloc() : memref<32xf32>
      ktdf.private_yield %slots#0, %slots#1, %t, %out : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>, !ktdf.token, memref<32xf32>
    }
    ktdf.stage depends_in(none) depends_out(%p#2) {
      scf.for %i = %c0 to %c4 step %c1 {
        ktdf.data_transfer from %a_mem[0] size [32] to %p#0 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
        ktdf.data_transfer from %b_mem[0] size [32] to %p#1 size [32] : memref<32xf32, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32>
      }
    } {applicable_units = ["L1LU"]}
    ktdf.stage depends_in(%p#2) depends_out(none) {
      scf.for %i = %c0 to %c4 step %c1 {
        %a = ktdf.read_from_fifo %p#0 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
        %b = ktdf.read_from_fifo %p#1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 32xf32> -> memref<32xf32>
        linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%a, %b : memref<32xf32>, memref<32xf32>) outs(%p#3 : memref<32xf32>) {
        ^bb0(%in: f32, %in_b: f32, %o: f32):
          %sq = arith.mulf %in_b, %in_b : f32
          %r = arith.addf %in, %sq : f32
          linalg.yield %r : f32
        }
      }
    } {applicable_units = ["SFU"]}
  }
  return
}
