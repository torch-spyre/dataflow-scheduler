// RUN: dataflow-scheduler-opt --splat-legalization --split-input-file --verify-diagnostics %s | FileCheck %s

// Tests moving a splat whose load unit cannot splat down its path.

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<"DDR" = "DDR", "L1" = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

// The splat moves to the first transfer into a FIFO whose load unit can splat,
// and the hops before it carry only the source.
// CHECK-LABEL: func.func @move_to_splat_capable_unit
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:4 = ktdf.private
// CHECK-DAG:         ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "L1", 1xf16>
// CHECK-DAG:         memref.alloc() : memref<1xf16, "L1">
// CHECK-DAG:         ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
// CHECK:           ktdf.stage {{.+}} {
// CHECK-NEXT:        ktdf.data_transfer from %{{.+}}[0] size [1] to %[[P]]#0 size [1] :
// CHECK:           ktdf.stage {{.+}} {
// CHECK-NEXT:        ktdf.data_transfer from %[[P]]#0 size [1] to %[[P]]#1[0] size [1] :
// CHECK:           ktdf.stage {{.+}} {
// CHECK-NEXT:        ktdf.data_transfer from %[[P]]#1[0] size [1] to %[[P]]#2 size [64]
// CHECK-SAME:          transfer_mode = #ktdf.transfer_mode<splat>
func.func @move_to_splat_capable_unit(%ddr: memref<1xf16, "DDR">) {
  ktdf.pipeline {
    %p:4 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.token) {
      %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
      %l1 = memref.alloc() : memref<64xf16, "L1">
      %f1 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      %t = ktdf.create_token : !ktdf.token
      ktdf.private_yield %f0, %l1, %f1, %t : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %ddr[0] size [1] to %p#0 size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<1xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
    } {applicable_units = ["MNILU"]}
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %p#0 size [64] to %p#1[0] size [64] : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, memref<64xf16, "L1">
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %p#1[0] size [64] to %p#2 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    } {applicable_units = ["L1LU"]}
  }
  return
}

// -----

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<"DDR" = "DDR", "L1" = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

// A splat into a memory, as path expansion leaves it, moves to the transfer
// into a FIFO after it.
// CHECK-LABEL: func.func @move_out_of_memory
// CHECK:         ktdf.pipeline {
// CHECK:           %[[P:.+]]:3 = ktdf.private
// CHECK:             memref.alloc() : memref<1xf16, "L1">
// CHECK:           ktdf.stage {{.+}} {
// CHECK-NEXT:        ktdf.data_transfer from %{{.+}}[0] size [1] to %[[P]]#0[0] size [1] :
// CHECK:           ktdf.stage {{.+}} {
// CHECK-NEXT:        ktdf.data_transfer from %[[P]]#0[0] size [1] to %[[P]]#1 size [64]
// CHECK-SAME:          transfer_mode = #ktdf.transfer_mode<splat>
func.func @move_out_of_memory(%ddr: memref<1xf16, "DDR">) {
  ktdf.pipeline {
    %p:3 = ktdf.private -> (memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.token) {
      %l1 = memref.alloc() : memref<64xf16, "L1">
      %f = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
      %t = ktdf.create_token : !ktdf.token
      ktdf.private_yield %l1, %f, %t : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %ddr[0] size [1] to %p#0[0] size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<1xf16, "DDR">, memref<64xf16, "L1">
    } {applicable_units = ["MNILU"]}
    ktdf.stage depends_in(none) depends_out(none) {
      ktdf.data_transfer from %p#0[0] size [64] to %p#1 size [64] : memref<64xf16, "L1">, !ktdf.fifo.slot<"L1LU" -> "SFU", 64xf16>
    } {applicable_units = ["L1LU"]}
  }
  return
}

// -----

ktdf_arch.device @sample_device attributes {mem_space_mapping = #ktdf_arch.map<"DDR" = "DDR", "L1" = "L1">} import("../../Dialect/KTDFArch/sample_device.mlir")

// A splat that no load unit on its path can perform is rejected.
func.func @no_splat_capable_unit(%ddr: memref<1xf16, "DDR">) {
  ktdf.pipeline {
    %p:2 = ktdf.private -> (!ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.token) {
      %f0 = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
      %t = ktdf.create_token : !ktdf.token
      ktdf.private_yield %f0, %t : !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>, !ktdf.token
    }
    ktdf.stage depends_in(none) depends_out(none) {
      // expected-error @below {{splat data_transfer requires the load unit to declare ktdf_arch.feature.simd = { splat, ... }}}
      ktdf.data_transfer from %ddr[0] size [1] to %p#0 size [64] {transfer_mode = #ktdf.transfer_mode<splat>} : memref<1xf16, "DDR">, !ktdf.fifo.slot<"DDR" -> "L1", 64xf16>
    } {applicable_units = ["MNILU"]}
  }
  return
}
