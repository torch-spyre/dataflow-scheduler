// RUN: dataflow-scheduler-opt --path-expansion %s | FileCheck %s
// NOTE: FIFO memory-space names ("DDR", "SFU", "L1") must match the kind
// strings in sample_device.mlir, which the routing graph keys on.

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @gather_pipeline() {
// CHECK:         ktdf.pipeline {
// CHECK-NEXT:      %[[PRV:.+]]:8 = ktdf.private -> (memref<2x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>, !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>, memref<2x64xf16, "L1">, !ktdf.token, !ktdf.token, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:        %[[ALLOC0:.+]] = memref.alloc() : memref<2x64xf16, "L1">
// CHECK-NEXT:        %[[FIFO0:.+]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
// CHECK-NEXT:        %[[FIFO1:.+]] = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>
// CHECK-NEXT:        %[[ALLOC1:.+]] = memref.alloc() : memref<2x64xf16, "L1">
// CHECK-NEXT:        %[[TOK0:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:        %[[TOK1:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:        %[[TOK2:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:        %[[TOK3:.+]] = ktdf.create_token : !ktdf.token
// CHECK-NEXT:        ktdf.private_yield %[[ALLOC0]], %[[FIFO0]], %[[FIFO1]], %[[ALLOC1]], %[[TOK0]], %[[TOK1]], %[[TOK2]], %[[TOK3]]
// CHECK:           }
// CHECK-NEXT:      ktdf.stage depends_in(none) depends_out(%[[PRV]]#4) {
// CHECK-NEXT:        %[[EQ:.+]] = arith.cmpi eq
// CHECK-NEXT:        scf.if %[[EQ]] {
// CHECK-NEXT:          ktdf.data_transfer from %{{.*}}{{\[}}%{{.*}}{{\]}} size [32] to %{{.*}}{{\[}}%{{.*}}{{\]}} size [32]
// CHECK:             }
// CHECK-NEXT:        ktdf.ind_data_transfer
// CHECK-NEXT:          ind_src = %{{.*}}{{\[}}%{{.*}}{{\]}}
// CHECK-NEXT:          dir_src = %{{.*}}{{\[}}%{{.*}}, %{{.*}}, %{{.*}}{{\]}} size [1, 2, 64]
// CHECK-NEXT:          ind_dst = none
// CHECK-NEXT:          dir_dst = %[[PRV]]#0{{\[}}%{{.*}}, %{{.*}}{{\]}} size [2, 64]
// CHECK-NEXT:          : memref<32xindex, {{.*}}, "IAB">, memref<64x2x64xf16, {{.*}}, "DDR">, none, memref<2x64xf16, "L1">
// CHECK:           } {applicable_units = ["MNILU"]}
// CHECK-NEXT:      ktdf.stage depends_in(%[[PRV]]#4) depends_out(%[[PRV]]#5) {
// CHECK-NEXT:        ktdf.data_transfer from %[[PRV]]#0[0, 0] size [2, 64] to %[[PRV]]#1 size [128]
// CHECK-SAME:          : memref<2x64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 128xf16>
// CHECK:           } {applicable_units = ["L1LU"]}
// CHECK-NEXT:      ktdf.stage depends_in(%[[PRV]]#5) depends_out(%[[PRV]]#6) {
// CHECK-NEXT:        %[[READ:.+]] = ktdf.read_from_fifo %[[PRV]]#1 : <"L1LU" -> "SFU", 128xf16> -> tensor<2x64xf16>
// CHECK:             linalg.generic {indexing_maps = [#[[$MAP]], #[[$MAP]]],
// CHECK:             ktdf.write_to_fifo %{{.*}}, %[[PRV]]#2 : tensor<2x64xf16>, <"SFU" -> "L1SU", 128xf16>
// CHECK:           } {applicable_units = ["SFU"]}
// CHECK-NEXT:      ktdf.stage depends_in(%[[PRV]]#6) depends_out(%[[PRV]]#7) {
// CHECK-NEXT:        ktdf.data_transfer from %[[PRV]]#2 size [128] to %[[PRV]]#3[0, 0] size [2, 64]
// CHECK-SAME:          : !ktdf.fifo.slot<"SFU" -> "L1SU", 128xf16>, memref<2x64xf16, "L1">
// CHECK:           } {applicable_units = ["L1SU"]}
// CHECK-NEXT:      ktdf.stage depends_in(%[[PRV]]#7) depends_out(none) {
// CHECK-NEXT:        ktdf.data_transfer from %[[PRV]]#3[0, 0] size [2, 64] to %{{.*}}[0, 0] size [2, 64]
// CHECK-SAME:          : memref<2x64xf16, "L1">, memref<2x64xf16, {{.*}}, "DDR">
// CHECK:           } {applicable_units = ["MNISU"]}

#set1 = affine_set<(d0, d1, d2) : (d0 >= 0, -d0 + 63 >= 0, d1 >= 0, -d1 + 1 >= 0, d2 >= 0, -d2 + 63 >= 0)>
#set2 = affine_set<(d0, d1) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0)>
#set3 = affine_set<(d1) : (d1 >= 0, -d1 + 31 >= 0)>
#set4 = affine_set<(d2, d3) : (d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#set5 = affine_set<(d0, d1, d2, d3) : (d0 >= 0, -d0 + 1 >= 0, d1 >= 0, -d1 + 31 >= 0, d2 >= 0, -d2 + 1 >= 0, d3 >= 0, -d3 + 63 >= 0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @gather_pipeline() {
    %c0     = arith.constant 0 : index
    %c1     = arith.constant 1 : index
    %c2     = arith.constant 2 : index
    %c32    = arith.constant 32 : index
    %c1000  = arith.constant 1000 : index
    %c10000 = arith.constant 10000 : index

    %desc_1_mv = ktdp.construct_memory_view %c1000,
        sizes: [64, 2, 64], strides: [64, 4096, 1]
        {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
        : memref<64x2x64xf16>
    %desc_1_msc = memref.memory_space_cast %desc_1_mv
        : memref<64x2x64xf16> to memref<64x2x64xf16, "DDR">
    %desc_1_rc = memref.reinterpret_cast %desc_1_msc
        to offset: [0], sizes: [64, 2, 64], strides: [64, 4096, 1]
        : memref<64x2x64xf16, "DDR">
          to memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">

    %addr_buf_base = arith.constant 999 : index
    %addr_buf_mv = ktdp.construct_memory_view %addr_buf_base, sizes: [2, 32], strides: [32, 1]
        {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>}
        : memref<2x32xindex, #ktdp.memory_space<global>>
    %addr_buf_msc = memref.memory_space_cast %addr_buf_mv
        : memref<2x32xindex, #ktdp.memory_space<global>>
          to memref<2x32xindex, "DDR">

    %desc_2_mv = ktdp.construct_memory_view %c10000,
        sizes: [2, 32, 2, 64], strides: [4096, 128, 64, 1]
        {coordinate_set = #set5, memory_space = #ktdp.memory_space<global>}
        : memref<2x32x2x64xf16>
    %desc_2_msc = memref.memory_space_cast %desc_2_mv
        : memref<2x32x2x64xf16> to memref<2x32x2x64xf16, "DDR">

    scf.for %i1 = %c0 to %c2 step %c1 {
      scf.for %i2 = %c0 to %c32 step %c1 {
        %addr_off = arith.muli %i1, %c32 : index
        %addr_buf_rc = memref.reinterpret_cast %addr_buf_msc
            to offset: [%addr_off], sizes: [32], strides: [1]
            : memref<2x32xindex, "DDR">
              to memref<32xindex, strided<[1], offset: ?>, "DDR">

        %iab_mv = memref.alloc() : memref<32xindex, "IAB">
        %iab_rc = memref.reinterpret_cast %iab_mv
            to offset: [0], sizes: [32], strides: [1]
            : memref<32xindex, "IAB">
              to memref<32xindex, strided<[1], offset: ?>, "IAB">

        %c4096 = arith.constant 4096 : index
        %c128  = arith.constant 128  : index
        %off_i1   = arith.muli %i1, %c4096 : index
        %off_i2   = arith.muli %i2, %c128  : index
        %desc2_off = arith.addi %off_i1, %off_i2 : index
        %desc_2_rc = memref.reinterpret_cast %desc_2_msc
            to offset: [%desc2_off], sizes: [2, 64], strides: [64, 1]
            : memref<2x32x2x64xf16, "DDR">
              to memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">

        ktdf.pipeline {
          %prv:5 = ktdf.private -> (
              !ktdf.fifo.slot<"DDR" -> "SFU", 128xf16>,
              !ktdf.fifo.slot<"SFU" -> "DDR", 128xf16>,
              !ktdf.token,
              !ktdf.token,
              !ktdf.token
          ) {
            %fifo_ld = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "SFU", 128xf16>
            %fifo_st = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "DDR", 128xf16>
            %tok0 = ktdf.create_token : !ktdf.token
            %tok1 = ktdf.create_token : !ktdf.token
            %tok2 = ktdf.create_token : !ktdf.token
            ktdf.private_yield %fifo_ld, %fifo_st, %tok0, %tok1, %tok2
                : !ktdf.fifo.slot<"DDR" -> "SFU", 128xf16>,
                  !ktdf.fifo.slot<"SFU" -> "DDR", 128xf16>,
                  !ktdf.token, !ktdf.token, !ktdf.token
          }

          ktdf.stage depends_in(none) depends_out(%prv#2) {
            %eq0 = arith.cmpi eq, %i2, %c0 : index
            scf.if %eq0 {
              ktdf.data_transfer
                  from %addr_buf_rc[%c0] size [32]
                  to   %iab_rc[%c0]      size [32]
                  : memref<32xindex, strided<[1], offset: ?>, "DDR">,
                    memref<32xindex, strided<[1], offset: ?>, "IAB">
            }
            ktdf.ind_data_transfer
                ind_src = %iab_rc[%i2]
                dir_src = %desc_1_rc[%c0, %c0, %c0] size [1, 2, 64]
                ind_dst = none
                dir_dst = %prv#0                     size [2, 64]
                : memref<32xindex, strided<[1], offset: ?>, "IAB">,
                  memref<64x2x64xf16, strided<[64, 4096, 1], offset: ?>, "DDR">,
                  none,
                  !ktdf.fifo.slot<"DDR" -> "SFU", 128xf16>
          }

          ktdf.stage depends_in(%prv#2) depends_out(%prv#3) {
            %tmp1_0 = ktdf.read_from_fifo %prv#0
                : !ktdf.fifo.slot<"DDR" -> "SFU", 128xf16> -> tensor<2x64xf16>
            %12 = tensor.empty() : tensor<2x64xf16>
            %13 = linalg.generic {
                indexing_maps = [#map1, #map1],
                iterator_types = ["parallel", "parallel"]
            } ins(%tmp1_0 : tensor<2x64xf16>) outs(%12 : tensor<2x64xf16>) {
            ^bb0(%in: f16, %out: f16):
              %cf10 = arith.constant 10.000000e+00 : f16
              %14 = arith.addf %in, %cf10 : f16
              linalg.yield %14 : f16
            } -> tensor<2x64xf16>
            ktdf.write_to_fifo %13, %prv#1
                : tensor<2x64xf16>, !ktdf.fifo.slot<"SFU" -> "DDR", 128xf16>
          } {applicable_units = ["SFU"]}

          ktdf.stage depends_in(%prv#3) depends_out(%prv#4) {
            ktdf.data_transfer
                from %prv#1              size [2, 64]
                to   %desc_2_rc[%c0, %c0] size [2, 64]
                : !ktdf.fifo.slot<"SFU" -> "DDR", 128xf16>,
                  memref<2x64xf16, strided<[64, 1], offset: ?>, "DDR">
          }
        }
      } {loop_type = #ktdf.loop_type<parallel_loop>}
    } {loop_type = #ktdf.loop_type<parallel_loop>}
    return
  }
}
