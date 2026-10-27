// RUN: dataflow-scheduler-opt --reduction-dim-chunking="num-chunks=4,1" --allow-unregistered-dialect %s | FileCheck %s

// Chunking of a top-k reduction with multiple outputs: the values output is
// indexed by the chunked dim and partitioned by the chunks, while the other
// outputs accumulate over them.

#val = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#row = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#topk = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
ktdf_arch.device @spyre_single_corelet import("../../Dialect/KTDFArch/sample_device.mlir")

// The values are loaded and stored in chunks, the accumulators are read back
// from where they are stored on all but the first chunk.
// CHECK-LABEL: func.func @topk(
// CHECK-SAME:      %[[LOGITS:[^:]+]]: memref<2x1x64x64xf16, "L1">, %[[VALS:[^:]+]]: memref<2x1x64x64xf16, "L1">,
// CHECK-SAME:      %[[IDX:[^:]+]]: memref<2x1x64xf16, "L1">, %[[CNT:[^:]+]]: memref<2x1x64xf16, "L1">,
// CHECK-SAME:      %[[TKV:[^:]+]]: memref<2x1x4x64xf16, "L1">, %[[TKI:[^:]+]]: memref<2x1x4x64xf16, "L1">)
// CHECK:         scf.for %[[B:.+]] = {{.+}} {
// CHECK:           scf.for %[[CHUNK:.+]] = {{.+}} {
// CHECK:             %[[FIRST:.+]] = arith.cmpi eq, %[[CHUNK]]
// CHECK:             ktdf.pipeline {
// CHECK:               %[[S:.+]]:16 = ktdf.private
// CHECK-SAME:              (!ktdf.fifo.slot<"L1LU" -> "SFU", 1024xf16>,
// CHECK-SAME:               !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
// CHECK-SAME:               !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>,
// CHECK-SAME:               !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>,
// CHECK-SAME:               !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>,
// CHECK-SAME:               !ktdf.fifo.slot<"SFU" -> "L1SU", 1024xf16>,
// CHECK-SAME:               !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>,
// CHECK-SAME:               !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>,
// CHECK-SAME:               !ktdf.token, !ktdf.token)
// CHECK:               ktdf.stage {{.+}} {
// CHECK-NEXT:            ktdf.data_transfer from %[[LOGITS]][%[[B]], 0, %[[CHUNK]] * 16, 0] size [1, 1, 16, 64] to %[[S]]#0 size [1024]
// CHECK-NEXT:            ktdf.data_transfer from %[[IDX]][%[[B]], 0, 0] size [1, 1, 64] to %[[S]]#5 size [64]
// CHECK-NEXT:            ktdf.data_transfer from %[[CNT]][%[[B]], 0, 0] size [1, 1, 64] to %[[S]]#6 size [64]
// CHECK-NEXT:            ktdf.data_transfer from %[[TKV]][%[[B]], 0, 0, 0] size [1, 1, 4, 64] to %[[S]]#7 size [256]
// CHECK-NEXT:            ktdf.data_transfer from %[[TKI]][%[[B]], 0, 0, 0] size [1, 1, 4, 64] to %[[S]]#8 size [256]
// CHECK-NEXT:            scf.if %[[FIRST]] {
// CHECK-NEXT:            } else {
// CHECK-NEXT:              ktdf.data_transfer from %[[IDX]][%[[B]], 0, 0] size [1, 1, 64] to %[[S]]#1 size [64]
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.if %[[FIRST]] {
// CHECK-NEXT:            } else {
// CHECK-NEXT:              ktdf.data_transfer from %[[CNT]][%[[B]], 0, 0] size [1, 1, 64] to %[[S]]#2 size [64]
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.if %[[FIRST]] {
// CHECK-NEXT:            } else {
// CHECK-NEXT:              ktdf.data_transfer from %[[TKV]][%[[B]], 0, 0, 0] size [1, 1, 4, 64] to %[[S]]#3 size [256]
// CHECK-NEXT:            }
// CHECK-NEXT:            scf.if %[[FIRST]] {
// CHECK-NEXT:            } else {
// CHECK-NEXT:              ktdf.data_transfer from %[[TKI]][%[[B]], 0, 0, 0] size [1, 1, 4, 64] to %[[S]]#4 size [256]
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK:               ktdf.stage {{.+}} {
// CHECK-NEXT:            %[[V:.+]] = ktdf.read_from_fifo %[[S]]#0 {{.+}} -> tensor<1x16x64xf16>
// CHECK-NEXT:            %[[X0:.+]] = ktdf.read_from_fifo %[[S]]#5
// CHECK-NEXT:            %[[N0:.+]] = ktdf.read_from_fifo %[[S]]#6
// CHECK-NEXT:            %[[KV0:.+]] = ktdf.read_from_fifo %[[S]]#7
// CHECK-NEXT:            %[[KI0:.+]] = ktdf.read_from_fifo %[[S]]#8
// CHECK-NEXT:            %[[X:.+]] = scf.if %[[FIRST]] -> (tensor<1x64xf16>) {
// CHECK-NEXT:              scf.yield %[[X0]]
// CHECK-NEXT:            } else {
// CHECK-NEXT:              %[[XP:.+]] = ktdf.read_from_fifo %[[S]]#1
// CHECK-NEXT:              scf.yield %[[XP]]
// CHECK-NEXT:            }
// CHECK-NEXT:            %[[N:.+]] = scf.if %[[FIRST]] -> (tensor<1x64xf16>) {
// CHECK-NEXT:              scf.yield %[[N0]]
// CHECK-NEXT:            } else {
// CHECK-NEXT:              %[[NP:.+]] = ktdf.read_from_fifo %[[S]]#2
// CHECK-NEXT:              scf.yield %[[NP]]
// CHECK-NEXT:            }
// CHECK-NEXT:            %[[KV:.+]] = scf.if %[[FIRST]] -> (tensor<1x4x64xf16>) {
// CHECK-NEXT:              scf.yield %[[KV0]]
// CHECK-NEXT:            } else {
// CHECK-NEXT:              %[[KVP:.+]] = ktdf.read_from_fifo %[[S]]#3
// CHECK-NEXT:              scf.yield %[[KVP]]
// CHECK-NEXT:            }
// CHECK-NEXT:            %[[KI:.+]] = scf.if %[[FIRST]] -> (tensor<1x4x64xf16>) {
// CHECK-NEXT:              scf.yield %[[KI0]]
// CHECK-NEXT:            } else {
// CHECK-NEXT:              %[[KIP:.+]] = ktdf.read_from_fifo %[[S]]#4
// CHECK-NEXT:              scf.yield %[[KIP]]
// CHECK-NEXT:            }
// CHECK-NEXT:            %[[R:.+]]:5 = linalg.generic
// CHECK-SAME:              outs(%[[V]], %[[X]], %[[N]], %[[KV]], %[[KI]] : tensor<1x16x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x4x64xf16>, tensor<1x4x64xf16>)
// CHECK:                 } -> (tensor<1x16x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x4x64xf16>, tensor<1x4x64xf16>)
// CHECK-NEXT:            ktdf.write_to_fifo %[[R]]#0, %[[S]]#9 : tensor<1x16x64xf16>
// CHECK-NEXT:            ktdf.write_to_fifo %[[R]]#1, %[[S]]#10 : tensor<1x64xf16>
// CHECK-NEXT:            ktdf.write_to_fifo %[[R]]#2, %[[S]]#11 : tensor<1x64xf16>
// CHECK-NEXT:            ktdf.write_to_fifo %[[R]]#3, %[[S]]#12 : tensor<1x4x64xf16>
// CHECK-NEXT:            ktdf.write_to_fifo %[[R]]#4, %[[S]]#13 : tensor<1x4x64xf16>
// CHECK-NEXT:          }
// CHECK:               ktdf.stage {{.+}} {
// CHECK-NEXT:            ktdf.data_transfer from %[[S]]#9 size [1024] to %[[VALS]][%[[B]], 0, %[[CHUNK]] * 16, 0] size [1, 1, 16, 64]
// CHECK-NEXT:            ktdf.data_transfer from %[[S]]#10 size [64] to %[[IDX]][%[[B]], 0, 0] size [1, 1, 64]
// CHECK-NEXT:            ktdf.data_transfer from %[[S]]#11 size [64] to %[[CNT]][%[[B]], 0, 0] size [1, 1, 64]
// CHECK-NEXT:            ktdf.data_transfer from %[[S]]#12 size [256] to %[[TKV]][%[[B]], 0, 0, 0] size [1, 1, 4, 64]
// CHECK-NEXT:            ktdf.data_transfer from %[[S]]#13 size [256] to %[[TKI]][%[[B]], 0, 0, 0] size [1, 1, 4, 64]
// CHECK-NEXT:          }
func.func @topk(%logits: memref<2x1x64x64xf16, "L1">, %vals: memref<2x1x64x64xf16, "L1">,
                %idx: memref<2x1x64xf16, "L1">, %cnt: memref<2x1x64xf16, "L1">,
                %tkv: memref<2x1x4x64xf16, "L1">, %tki: memref<2x1x4x64xf16, "L1">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %b = %c0 to %c2 step %c1 {
    ktdf.pipeline {
      %s:12 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>,
                                !ktdf.fifo.slot<"SFU" -> "L1SU", 4096xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>,
                                !ktdf.token, !ktdf.token) {
        %i0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>
        %i1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        %i2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        %i3 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
        %i4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
        %o0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 4096xf16>
        %o1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
        %o2 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
        %o3 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>
        %o4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>
        %t0 = ktdf.create_token : !ktdf.token
        %t1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %i0, %i1, %i2, %i3, %i4, %o0, %o1, %o2, %o3, %o4, %t0, %t1 : !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>, !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 4096xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%s#10) {
        ktdf.data_transfer from %logits[%b, 0, 0, 0] size [1, 1, 64, 64] to %s#0 size [4096] : memref<2x1x64x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 4096xf16>
        ktdf.data_transfer from %idx[%b, 0, 0] size [1, 1, 64] to %s#1 size [64] : memref<2x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        ktdf.data_transfer from %cnt[%b, 0, 0] size [1, 1, 64] to %s#2 size [64] : memref<2x1x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
        ktdf.data_transfer from %tkv[%b, 0, 0, 0] size [1, 1, 4, 64] to %s#3 size [256] : memref<2x1x4x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
        ktdf.data_transfer from %tki[%b, 0, 0, 0] size [1, 1, 4, 64] to %s#4 size [256] : memref<2x1x4x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 256xf16>
      } {applicable_units = ["L1LU"]}
      ktdf.stage depends_in(%s#10) depends_out(%s#11) {
        %v = ktdf.read_from_fifo %s#0 : <"L1LU" -> "SFU", 4096xf16> -> tensor<1x64x64xf16>
        %x = ktdf.read_from_fifo %s#1 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
        %n = ktdf.read_from_fifo %s#2 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
        %kv = ktdf.read_from_fifo %s#3 : <"L1LU" -> "SFU", 256xf16> -> tensor<1x4x64xf16>
        %ki = ktdf.read_from_fifo %s#4 : <"L1LU" -> "SFU", 256xf16> -> tensor<1x4x64xf16>
        %r:5 = linalg.generic {indexing_maps = [#val, #row, #row, #topk, #topk],
                               iterator_types = ["parallel", "reduction", "parallel", "reduction"]}
            outs(%v, %x, %n, %kv, %ki : tensor<1x64x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x4x64xf16>, tensor<1x4x64xf16>) {
        ^bb0(%curr_val: f16, %curr_idx: f16, %counter: f16, %top_k_val: f16, %top_k_idx: f16):
          %one = arith.constant 1.0 : f16
          %k = linalg.index 3 : index
          %is_first = arith.cmpi eq, %k, %c0 : index
          %counter_inc = arith.addf %counter, %one : f16
          %counter_next = arith.select %is_first, %counter_inc, %counter : f16
          %curr_idx_selected = arith.select %is_first, %counter, %curr_idx : f16
          %res:4 = "spyreop_next.compare_and_swap_max"(%curr_val, %curr_idx_selected, %top_k_val, %top_k_idx) : (f16, f16, f16, f16) -> (f16, f16, f16, f16)
          linalg.yield %res#0, %res#1, %counter_next, %res#2, %res#3 : f16, f16, f16, f16, f16
        } -> (tensor<1x64x64xf16>, tensor<1x64xf16>, tensor<1x64xf16>, tensor<1x4x64xf16>, tensor<1x4x64xf16>)
        ktdf.write_to_fifo %r#0, %s#5 : tensor<1x64x64xf16>, <"SFU" -> "L1SU", 4096xf16>
        ktdf.write_to_fifo %r#1, %s#6 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
        ktdf.write_to_fifo %r#2, %s#7 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
        ktdf.write_to_fifo %r#3, %s#8 : tensor<1x4x64xf16>, <"SFU" -> "L1SU", 256xf16>
        ktdf.write_to_fifo %r#4, %s#9 : tensor<1x4x64xf16>, <"SFU" -> "L1SU", 256xf16>
      } {applicable_units = ["SFU"]}
      ktdf.stage depends_in(%s#11) depends_out(none) {
        ktdf.data_transfer from %s#5 size [4096] to %vals[%b, 0, 0, 0] size [1, 1, 64, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 4096xf16>, memref<2x1x64x64xf16, "L1">
        ktdf.data_transfer from %s#6 size [64] to %idx[%b, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
        ktdf.data_transfer from %s#7 size [64] to %cnt[%b, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<2x1x64xf16, "L1">
        ktdf.data_transfer from %s#8 size [256] to %tkv[%b, 0, 0, 0] size [1, 1, 4, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, memref<2x1x4x64xf16, "L1">
        ktdf.data_transfer from %s#9 size [256] to %tki[%b, 0, 0, 0] size [1, 1, 4, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 256xf16>, memref<2x1x4x64xf16, "L1">
      } {applicable_units = ["L1SU"]}
    }
  } {loop_type = #ktdf.loop_type<parallel_loop>}
  return
}
