// RUN: dataflow-scheduler-opt --access-granularity-legalization --verify-diagnostics --split-input-file %s

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<"DDR" = "DDR", "L1" = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

// store: fails when destination size (16xf16 = 32 bytes) cannot satisfy the
// store unit's minimum fitting access granularity (64 bytes = 32xf16) without
// write-widening.

module {
  func.func @store_under_granularity(%dst: memref<16xf16, "L1">) {
    ktdf.pipeline {
      %p:2 = ktdf.private -> (!ktdf.fifo.slot<"SFU" -> "L1SU", 16xf16>, !ktdf.token) {
        %slot = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1SU", 16xf16>
        %t = ktdf.create_token : !ktdf.token
        ktdf.private_yield %slot, %t : !ktdf.fifo.slot<"SFU" -> "L1SU", 16xf16>, !ktdf.token
      }
      ktdf.stage depends_in(%p#1) depends_out(none) {
        // expected-error @below {{store transfer of 16 elements cannot satisfy the hardware store access granularity of 32 elements without write-widening}}
        ktdf.data_transfer from %p#0 size [16] to %dst[0] size [16]
            : !ktdf.fifo.slot<"SFU" -> "L1SU", 16xf16>, memref<16xf16, "L1">
      } {applicable_units = ["L1SU"]}
    }
    return
  }
}
