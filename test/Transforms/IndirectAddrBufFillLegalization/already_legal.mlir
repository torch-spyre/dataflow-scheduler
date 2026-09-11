// RUN: dataflow-scheduler-opt --indirect-addr-buf-fill-legalization %s -split-input-file | FileCheck %s

// The store unit already fills the indirect address buffer out of "L1", which
// it has a datapath from, so the pipeline is left exactly as it is: no second
// staging buffer, no extra transfer.

// CHECK-LABEL: func.func @legal_fill() {
// CHECK:         %[[PRV:.+]]:3 = ktdf.private -> (memref<32xindex, "L1">, !ktdf.token, !ktdf.token) {
// CHECK-NEXT:      %[[IDX_BUF:.+]] = memref.alloc() : memref<32xindex, "L1">
// CHECK-NOT:       memref.alloc() : memref<32xindex, "L1">
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[PRV]]#1) {
// CHECK-NEXT:      ktdf.data_transfer from %{{.*}}{{\[}}%{{.*}}] size [32] to %[[PRV]]#0[0] size [32] : memref<32xindex, strided<[1], offset: ?>, "DDR">, memref<32xindex, "L1">
// CHECK-NEXT:    } {applicable_units = ["MNILU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[PRV]]#1) depends_out(%[[PRV]]#2) {
// CHECK-NEXT:      ktdf.data_transfer from %[[PRV]]#0[0] size [32] to %{{.*}}{{\[}}%{{.*}}] size [32] : memref<32xindex, "L1">, memref<32xindex, strided<[1], offset: ?>, "IAB">
// CHECK-NEXT:    } {applicable_units = ["MNISU"]}

#set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @legal_fill() {
    %c0 = arith.constant 0 : index
    %c999 = arith.constant 999 : index

    %addr_buf_mv = ktdp.construct_memory_view %c999, sizes: [32], strides: [1]
        {coordinate_set = #set, memory_space = #ktdp.memory_space<global>}
        : memref<32xindex, #ktdp.memory_space<global>>
    %addr_buf_msc = memref.memory_space_cast %addr_buf_mv
        : memref<32xindex, #ktdp.memory_space<global>> to memref<32xindex, "DDR">
    %addr_buf_rc = memref.reinterpret_cast %addr_buf_msc
        to offset: [0], sizes: [32], strides: [1]
        : memref<32xindex, "DDR">
          to memref<32xindex, strided<[1], offset: ?>, "DDR">

    %iab = memref.alloc() : memref<32xindex, "IAB">
    %iab_rc = memref.reinterpret_cast %iab
        to offset: [0], sizes: [32], strides: [1]
        : memref<32xindex, "IAB">
          to memref<32xindex, strided<[1], offset: ?>, "IAB">

    ktdf.pipeline {
      %prv:3 = ktdf.private -> (
          memref<32xindex, "L1">,
          !ktdf.token,
          !ktdf.token
      ) {
        %l1_idx = memref.alloc() : memref<32xindex, "L1">
        %tok0 = ktdf.create_token : !ktdf.token
        %tok1 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %l1_idx, %tok0, %tok1
            : memref<32xindex, "L1">, !ktdf.token, !ktdf.token
      }

      ktdf.stage depends_in(none) depends_out(%prv#1) {
        ktdf.data_transfer
            from %addr_buf_rc[%c0] size [32]
            to   %prv#0[0]         size [32]
            : memref<32xindex, strided<[1], offset: ?>, "DDR">,
              memref<32xindex, "L1">
      } {applicable_units = ["MNILU"]}

      ktdf.stage depends_in(%prv#1) depends_out(%prv#2) {
        ktdf.data_transfer
            from %prv#0[0]    size [32]
            to   %iab_rc[%c0] size [32]
            : memref<32xindex, "L1">,
              memref<32xindex, strided<[1], offset: ?>, "IAB">
      } {applicable_units = ["MNISU"]}
    }
    return
  }
}

// -----

// The gather side of the same rule: the load unit fills the buffer out of
// "DDR", which it does read, so nothing is staged there either.

// CHECK-LABEL: func.func @gather_side_fill() {
// CHECK:         %[[PRV:.+]] = ktdf.private -> (!ktdf.token) {
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[PRV]]) {
// CHECK-NEXT:      ktdf.data_transfer from %{{.*}}{{\[}}%{{.*}}] size [32] to %{{.*}}{{\[}}%{{.*}}] size [32] : memref<32xindex, strided<[1], offset: ?>, "DDR">, memref<32xindex, strided<[1], offset: ?>, "IAB">
// CHECK-NEXT:    } {applicable_units = ["MNILU"]}
// CHECK-NOT:     memref<32xindex, "L1">

#set2 = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @gather_side_fill() {
    %c0 = arith.constant 0 : index
    %c999 = arith.constant 999 : index

    %addr_buf_mv = ktdp.construct_memory_view %c999, sizes: [32], strides: [1]
        {coordinate_set = #set2, memory_space = #ktdp.memory_space<global>}
        : memref<32xindex, #ktdp.memory_space<global>>
    %addr_buf_msc = memref.memory_space_cast %addr_buf_mv
        : memref<32xindex, #ktdp.memory_space<global>> to memref<32xindex, "DDR">
    %addr_buf_rc = memref.reinterpret_cast %addr_buf_msc
        to offset: [0], sizes: [32], strides: [1]
        : memref<32xindex, "DDR">
          to memref<32xindex, strided<[1], offset: ?>, "DDR">

    %iab = memref.alloc() : memref<32xindex, "IAB">
    %iab_rc = memref.reinterpret_cast %iab
        to offset: [0], sizes: [32], strides: [1]
        : memref<32xindex, "IAB">
          to memref<32xindex, strided<[1], offset: ?>, "IAB">

    ktdf.pipeline {
      %prv = ktdf.private -> (!ktdf.token) {
        %tok0 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %tok0 : !ktdf.token
      }

      ktdf.stage depends_in(none) depends_out(%prv) {
        ktdf.data_transfer
            from %addr_buf_rc[%c0] size [32]
            to   %iab_rc[%c0]      size [32]
            : memref<32xindex, strided<[1], offset: ?>, "DDR">,
              memref<32xindex, strided<[1], offset: ?>, "IAB">
      } {applicable_units = ["MNILU"]}
    }
    return
  }
}
