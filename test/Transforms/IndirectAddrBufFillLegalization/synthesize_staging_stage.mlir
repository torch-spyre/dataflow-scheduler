// RUN: dataflow-scheduler-opt --indirect-addr-buf-fill-legalization %s | FileCheck %s

// The store unit (MNISU) fills the indirect address buffer out of "DDR", which
// it has no datapath from, and no earlier stage can bring the index data into
// "L1" (e.g. every other input already lives in "L1"). A stage on the one unit
// that can -- MNILU -- is synthesized at the start of the pipeline, and the
// fill stage is made to wait for it.

// CHECK-LABEL: func.func @no_staging_stage() {
// CHECK:         %[[PRV:.+]]:3 = ktdf.private -> (!ktdf.token, !ktdf.token, memref<32xindex, "L1">) {
// CHECK:           %[[BUF:.+]] = memref.alloc() : memref<32xindex, "L1">
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[PRV]]#1) {
// CHECK-NEXT:      ktdf.data_transfer from %{{.*}}{{\[}}%{{.*}}] size [32] to %[[PRV]]#2[0] size [32] : memref<32xindex, strided<[1], offset: ?>, "DDR">, memref<32xindex, "L1">
// CHECK-NEXT:    } {applicable_units = ["MNILU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[PRV]]#1) depends_out(%[[PRV]]#0) {
// CHECK-NEXT:      ktdf.data_transfer from %[[PRV]]#2[0] size [32] to %{{.*}}{{\[}}%{{.*}}] size [32] : memref<32xindex, "L1">, memref<32xindex, strided<[1], offset: ?>, "IAB">
// CHECK-NEXT:    } {applicable_units = ["MNISU"]}

#set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @no_staging_stage() {
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
      } {applicable_units = ["MNISU"]}
    }
    return
  }
}
