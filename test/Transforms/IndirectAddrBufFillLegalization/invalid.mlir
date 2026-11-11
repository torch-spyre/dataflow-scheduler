// RUN: dataflow-scheduler-opt --indirect-addr-buf-fill-legalization %s -verify-diagnostics -split-input-file

// A stage that is not mapped to a single unit gives no answer to which unit
// performs the fill.

#set1 = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../Dialect/KTDFArch/sample_device.mlir")
  func.func @unmapped_stage() {
    %c0 = arith.constant 0 : index
    %c999 = arith.constant 999 : index

    %addr_buf_mv = ktdp.construct_memory_view %c999, sizes: [32], strides: [1]
        {coordinate_set = #set1, memory_space = #ktdp.memory_space<global>}
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

      // expected-error @below {{stage filling an indirect address buffer must be mapped to exactly one unit}}
      ktdf.stage depends_in(none) depends_out(%prv) {
        ktdf.data_transfer
            from %addr_buf_rc[%c0] size [32]
            to   %iab_rc[%c0]      size [32]
            : memref<32xindex, strided<[1], offset: ?>, "DDR">,
              memref<32xindex, strided<[1], offset: ?>, "IAB">
      }
    }
    return
  }
}
