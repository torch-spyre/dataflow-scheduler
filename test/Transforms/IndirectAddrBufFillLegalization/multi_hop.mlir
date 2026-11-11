// RUN: dataflow-scheduler-opt --indirect-addr-buf-fill-legalization %s | FileCheck %s

// Verify multi-hop IAB fill staging:
// Source is in DDR, fill unit is MNISU (which reads from L1).
// The path in the device is: DDR -> MNILU -> L2 -> L2LU -> L1.
// Two intermediate private buffers (in L2 and L1) and staging stages are created.

// CHECK-LABEL: func.func @multi_hop_fill() {
// CHECK:         %[[PRV:.+]]:5 = ktdf.private -> (!ktdf.token, !ktdf.token, !ktdf.token, memref<32xindex, "L2">, memref<32xindex, "L1">) {
// CHECK-DAG:       memref.alloc() : memref<32xindex, "L2">
// CHECK-DAG:       memref.alloc() : memref<32xindex, "L1">
// CHECK:         }
// CHECK:         ktdf.stage depends_in(none) depends_out(%[[PRV]]#1) {
// CHECK-NEXT:      ktdf.data_transfer from %{{.*}}{{\[}}%{{.*}}] size [32] to %[[PRV]]#3[0] size [32] : memref<32xindex, strided<[1], offset: ?>, "DDR">, memref<32xindex, "L2">
// CHECK-NEXT:    } {applicable_units = ["MNILU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[PRV]]#1) depends_out(%[[PRV]]#2) {
// CHECK-NEXT:      ktdf.data_transfer from %[[PRV]]#3[0] size [32] to %[[PRV]]#4[0] size [32] : memref<32xindex, "L2">, memref<32xindex, "L1">
// CHECK-NEXT:    } {applicable_units = ["L2LU"]}
// CHECK-NEXT:    ktdf.stage depends_in(%[[PRV]]#2) depends_out(%[[PRV]]#0) {
// CHECK-NEXT:      ktdf.data_transfer from %[[PRV]]#4[0] size [32] to %{{.*}}{{\[}}%{{.*}}] size [32] : memref<32xindex, "L1">, memref<32xindex, strided<[1], offset: ?>, "IAB">
// CHECK-NEXT:    } {applicable_units = ["MNISU"]}

#set = affine_set<(d0) : (d0 >= 0, -d0 + 31 >= 0)>

#DDR = { kind = "DDR" }
#L2  = { kind = "L2" }
#L1  = { kind = "L1" }
#IAB = {
  kind = "IAB",
  ktdf_arch.features = {
    ktdf_arch.feature.indirect_address_buffer = { num_entries = 32, entry_type = si32 }
  }
}

#MNILU = { kind = "MNILU" }
#L2LU  = { kind = "L2LU" }
#MNISU = { kind = "MNISU" }

module {
  ktdf_arch.device @multihop_device {
    %ddr = memory #DDR
    %l2 = memory #L2
    %l1 = memory #L1
    %mnilu = exec_unit #MNILU
    %l2lu = exec_unit #L2LU
    %mnisu = group { kind = "MNISU_Block" } share() {
      %iab = memory #IAB
      %mnisu_unit = exec_unit #MNISU
      yield %mnisu_unit
    } -> exec_unit

    datapath %ddr to %mnilu : memory, exec_unit
    datapath %mnilu to %l2  : exec_unit, memory
    datapath %l2 to %l2lu   : memory, exec_unit
    datapath %l2lu to %l1   : exec_unit, memory
    datapath %l1 to %mnisu  : memory, exec_unit
    datapath %mnisu to %ddr : exec_unit, memory
  }

  func.func @multi_hop_fill() {
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
