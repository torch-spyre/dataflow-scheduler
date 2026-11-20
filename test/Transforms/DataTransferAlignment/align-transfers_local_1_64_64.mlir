// RUN: dataflow-scheduler-opt --data-transfer-alignment %s | FileCheck %s

// Transfer chains anchored at ct_local (L1) memrefs instead of global ones.
// Both the strided input view and the row-major output view live in per-core
// scratchpad memory, so there is no staging buffer to widen and no block
// transfer to widen: the L1 view itself is the "staging buffer". The pass must
// still do the stride swap in the element-addressable L1LU / L1SU units.

// CHECK-LABEL: func.func @local_schedule_0() attributes {grid = [1]} {
// CHECK-NEXT:    %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:    %[[C64:.*]] = arith.constant 64 : index
// CHECK-NEXT:    %[[C8192:.*]] = arith.constant 8192 : index
// CHECK-NEXT:    %[[C0:.*]] = arith.constant 0 : index

// Strided ct_local input view rewritten to memory order: strides [4096, 1, 64]
// become row-major [4096, 64, 1] and the offset becomes dynamic.
// CHECK-NEXT:    %[[IN_VIEW:.*]] = ktdp.construct_memory_view %[[C0]], sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>, #ktdp.memory_space<ct_local>>

// Row-major ct_local output view is untouched.
// CHECK-NEXT:    %[[OUT_VIEW:.*]] = ktdp.construct_memory_view %[[C8192]], sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<1x64x64xf16, #ktdp.memory_space<ct_local>>

// The new input layout is propagated through the downstream memory_space_cast.
// CHECK-NEXT:    %[[IN_MSC:.*]] = memref.memory_space_cast %[[IN_VIEW]] : memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>, #ktdp.memory_space<ct_local>> to memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>, "L1">
// CHECK-NEXT:    %[[OUT_MSC:.*]] = memref.memory_space_cast %[[OUT_VIEW]] : memref<1x64x64xf16, #ktdp.memory_space<ct_local>> to memref<1x64x64xf16, "L1">

// CHECK-NEXT:    ktdf.pipeline {
// CHECK-NEXT:      ktdf.stage depends_in(none) depends_out(none) {

// The loop walking the contiguous dim shrinks 64 -> 1 (factor = 64 / 1), and
// the nested pipeline is wrapped in two element loops over [0, 64).
// CHECK-NEXT:        %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT:        scf.for %[[I:.*]] = %[[C0]] to %[[ONE]] step %[[C1]] {
// CHECK-NEXT:          %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NEXT:          %[[STEP:.*]] = arith.constant 1 : index
// CHECK-NEXT:          %[[BOUND:.*]] = arith.constant 64 : index
// CHECK-NEXT:          scf.for %[[ROW:.*]] = %[[ZERO]] to %[[BOUND]] step %[[STEP]] {
// CHECK-NEXT:            scf.for %[[COL:.*]] = %[[ZERO]] to %[[BOUND]] step %[[STEP]] {
// CHECK-NEXT:              ktdf.pipeline {
// CHECK-NEXT:                %[[FIFO:.*]]:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:                  %[[F0:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                  %[[F1:.*]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                  %[[T0:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                  %[[T1:.*]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:                  ktdf.private_yield %[[F0]], %[[F1]], %[[T0]], %[[T1]] : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
// CHECK-NEXT:                }

// Splat load of the strided input swaps the stride: indices are [.., col, row].
// The read comes straight from the rewritten L1 view, with no staging buffer.
// CHECK-NEXT:                ktdf.stage depends_in(none) depends_out(%[[FIFO]]#2) {
// CHECK-NEXT:                  %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:                  ktdf.data_transfer from %[[IN_MSC]][%[[C0]], %[[COL]], %[[ROW]]] size [1, 1, 1] to %[[FIFO]]#0 size [1, 1, 64] {{.*}}transfer_mode = #ktdf.transfer_mode<splat>{{.*}} : memref<1x64x64xf16, strided<[4096, 64, 1], offset: ?>, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK-NEXT:                } {applicable_units = ["L1LU"]}

// The compute stage is unchanged.
// CHECK-NEXT:                ktdf.stage depends_in(%[[FIFO]]#2) depends_out(%[[FIFO]]#3) {
// CHECK-NEXT:                  %[[READ:.*]] = ktdf.read_from_fifo %[[FIFO]]#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
// CHECK-NEXT:                  ktdf.write_to_fifo %[[READ]], %[[FIFO]]#1 : tensor<1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
// CHECK-NEXT:                } {applicable_units = ["SFU"]}

// Extract store keeps natural order: indices are [.., row, col].
// CHECK-NEXT:                ktdf.stage depends_in(%[[FIFO]]#3) depends_out(none) {
// CHECK-NEXT:                  %{{.*}} = arith.constant 0 : index
// CHECK-NEXT:                  ktdf.data_transfer from %[[FIFO]]#1 size [1, 1, 64] to %[[OUT_MSC]][%[[C0]], %[[ROW]], %[[COL]]] size [1, 1, 1] {{.*}}transfer_mode = #ktdf.transfer_mode<extract>{{.*}} : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64x64xf16, "L1">
// CHECK-NEXT:                } {applicable_units = ["L1SU"]}
// CHECK-NEXT:              }
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        } {loop_type = #ktdf.loop_type<parallel_loop>}
// CHECK-NEXT:      } {applicable_units = ["L1LU", "SFU", "L1SU"]}
// CHECK-NEXT:    }
// CHECK-NEXT:    return

#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 >= 0, d1 >= 0, -d1 + 63 >= 0, d2 >= 0, -d2 + 63 >= 0)>
module {
  module {
    func.func @ktir_fused_clone_transpose_local_0() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<#ktdp.memory_space<global> = "DDR", #ktdp.memory_space<ct_local> = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c8192 = arith.constant 8192 : index
      %c0 = arith.constant 0 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [1, 64, 64], strides: [4096, 1, 64] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<1x64x64xf16, strided<[4096, 1, 64]>, #ktdp.memory_space<ct_local>>
      %1 = ktdp.construct_memory_view %c8192, sizes: [1, 64, 64], strides: [4096, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<ct_local>} : memref<1x64x64xf16, #ktdp.memory_space<ct_local>>
      %memspacecast = memref.memory_space_cast %0 : memref<1x64x64xf16, strided<[4096, 1, 64]>, #ktdp.memory_space<ct_local>> to memref<1x64x64xf16, strided<[4096, 1, 64]>, "L1">
      %memspacecast_0 = memref.memory_space_cast %1 : memref<1x64x64xf16, #ktdp.memory_space<ct_local>> to memref<1x64x64xf16, "L1">
      ktdf.pipeline {
        ktdf.stage depends_in(none) depends_out(none) {
          scf.for %arg0 = %c0 to %c64 step %c1 {
            ktdf.pipeline {
              %2:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %3 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %5 = ktdf.create_token : !ktdf.token
                %6 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %3, %4, %5, %6 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%2#2) {
                ktdf.data_transfer from %memspacecast[%c0, %arg0, %c0 * 64] size [1, 1, 64] to %2#0 size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : memref<1x64x64xf16, strided<[4096, 1, 64]>, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%2#2) depends_out(%2#3) {
                %3 = ktdf.read_from_fifo %2#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x1x64xf16>
                ktdf.write_to_fifo %3, %2#1 : tensor<1x1x64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%2#3) depends_out(none) {
                ktdf.data_transfer from %2#1 size [1, 1, 64] to %memspacecast_0[%c0, %arg0, %c0 * 64] size [1, 1, 64] {dataflow_scheduler.throttle = 64 : i64} : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<1x64x64xf16, "L1">
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
      }
      return
    }
  }
}
