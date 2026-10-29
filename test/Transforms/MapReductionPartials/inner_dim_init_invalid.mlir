// RUN: dataflow-scheduler-opt --map-reduction-partials --verify-diagnostics %s

// An inner-dim reduction with no outer-dim loop reduces a copy of the fifo
// data, so there is nowhere to seed its init ahead of the reduction: an init
// other than tensor.empty is rejected rather than dropped.

#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d2)>
#set = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 255 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set1 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 255 >= 0, d1 >= 0, -d1 + 63 >= 0)>

module {
  module {
    func.func @sum_1core() attributes {grid = [1]} {
      call @local_schedule_0() : () -> ()
      return
    }
    func.func private @local_schedule_0()
  }
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
  module @local_schedule_0 {
    func.func @local_schedule_0() attributes {grid = [1]} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8589934592 = arith.constant 8589934592 : index
      %c256 = arith.constant 256 : index
      %0 = ktdp.construct_memory_view %c0, sizes: [2, 256, 64], strides: [16384, 64, 1] {coordinate_set = #set, memory_space = #ktdp.memory_space<global>} : memref<2x256x64xf16>
      %1 = ktdp.construct_memory_view %c8589934592, sizes: [256, 64], strides: [64, 1] {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>} : memref<256x64xf16>
      %memspacecast = memref.memory_space_cast %0 : memref<2x256x64xf16> to memref<2x256x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast = memref.reinterpret_cast %memspacecast to offset: [0], sizes: [2, 256, 64], strides: [16384, 64, 1] : memref<2x256x64xf16, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>>
      %cast = memref.cast %reinterpret_cast : memref<2x256x64xf16, strided<[16384, 64, 1]>, #ktdp.memory_space<global>> to memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>
      %memspacecast_0 = memref.memory_space_cast %1 : memref<256x64xf16> to memref<256x64xf16, #ktdp.memory_space<global>>
      %reinterpret_cast_1 = memref.reinterpret_cast %memspacecast_0 to offset: [0], sizes: [256, 64], strides: [64, 1] : memref<256x64xf16, #ktdp.memory_space<global>> to memref<256x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>>
      %cast_2 = memref.cast %reinterpret_cast_1 : memref<256x64xf16, strided<[64, 1]>, #ktdp.memory_space<global>> to memref<256x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
      ktdf.pipeline {
        %2:4 = ktdf.private -> (memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token) {
          %alloc = memref.alloc() : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>
          %alloc_3 = memref.alloc() : memref<256x1x64xf16, #ktdp.memory_space<ct_local>>
          %3 = ktdf.create_token : !ktdf.token
          %4 = ktdf.create_token : !ktdf.token
          ktdf.private_yield %alloc, %alloc_3, %3, %4 : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.token, !ktdf.token
        }
        ktdf.stage depends_in(none) depends_out(%2#2) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %cast[0, %arg0, 0] size [2, 1, 64] to %2#0[%4, 0, 0, 0] size [1, 2, 1, 64] : memref<2x256x64xf16, strided<[16384, 64, 1], offset: ?>, #ktdp.memory_space<global>>, memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNILU"]}
        ktdf.stage depends_in(%2#2) depends_out(%2#3) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %c0_3 = arith.constant 0 : index
            %c1_4 = arith.constant 1 : index
            %c1_5 = arith.constant 1 : index
            ktdf.pipeline {
              %3:4 = ktdf.private -> (!ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token) {
                %4 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                %5 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>
                %6 = ktdf.create_token : !ktdf.token
                %7 = ktdf.create_token : !ktdf.token
                ktdf.private_yield %4, %5, %6, %7 : !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, !ktdf.token, !ktdf.token
              }
              ktdf.stage depends_in(none) depends_out(%3#2) {
                %4 = arith.subi %arg0, %c0 : index
                %5 = arith.divsi %4, %c1 : index
                %c2 = arith.constant 2 : index
                scf.for %arg1 = %c0_3 to %c2 step %c1_4 {
                  ktdf.data_transfer from %2#0[%5, %arg1, 0, 0] size [1, 1, 1, 64] to %3#0 size [64] : memref<256x2x1x64xf16, #ktdp.memory_space<ct_local>>, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
                } {loop_type = #ktdf.loop_type<reduction_loop>}
              } {applicable_units = ["L1LU"]}
              ktdf.stage depends_in(%3#2) depends_out(%3#3) {
                %4 = ktdf.read_from_fifo %3#0 : <"L1LU" -> "SFU", 64xf16> -> tensor<1x64xf16>
                %5 = arith.constant dense<1.000000e+00> : tensor<1x64xf16>
                // expected-error @+1 {{rewriteInnerDimGeneric: an init other than tensor.empty is not supported without an outer reduction loop}}
                %6 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction", "parallel"]} ins(%4 : tensor<1x64xf16>) outs(%5 : tensor<1x64xf16>) {
                ^bb0(%in: f16, %out: f16):
                  %7 = arith.addf %in, %out : f16
                  linalg.yield %7 : f16
                } -> tensor<1x64xf16>
                ktdf.write_to_fifo %6, %3#1 : tensor<1x64xf16>, <"SFU" -> "L1SU", 64xf16>
              } {applicable_units = ["SFU"]}
              ktdf.stage depends_in(%3#3) depends_out(none) {
                %c2 = arith.constant 2 : index
                scf.for %arg1 = %c0_3 to %c2 step %c1_4 {
                  %4 = arith.subi %arg0, %c0 : index
                  %5 = arith.divsi %4, %c1 : index
                  %6 = arith.cmpi eq, %arg1, %c1_5 : index
                  scf.if %6 {
                    ktdf.data_transfer from %3#1 size [64] to %2#1[%5, 0, 0] size [1, 1, 64] : !ktdf.fifo.slot<"SFU" -> "L1SU", 64xf16>, memref<256x1x64xf16, #ktdp.memory_space<ct_local>>
                  }
                } {loop_type = #ktdf.loop_type<reduction_loop>}
              } {applicable_units = ["L1SU"]}
            }
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["L1LU", "SFU", "L1SU"]}
        ktdf.stage depends_in(%2#3) depends_out(none) {
          scf.for %arg0 = %c0 to %c256 step %c1 {
            %3 = arith.subi %arg0, %c0 : index
            %4 = arith.divsi %3, %c1 : index
            ktdf.data_transfer from %2#1[%4, 0, 0] size [1, 1, 64] to %cast_2[%arg0, %c0 * 64] size [1, 64] : memref<256x1x64xf16, #ktdp.memory_space<ct_local>>, memref<256x64xf16, strided<[64, 1], offset: ?>, #ktdp.memory_space<global>>
          } {loop_type = #ktdf.loop_type<parallel_loop>}
        } {applicable_units = ["MNISU"]}
      }
      return
    }
  }
}
