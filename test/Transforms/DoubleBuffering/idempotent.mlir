// Running the pass twice produces the same output as running once. The
// second run sees the pre-existing modulo and skips.

// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %s -double-buffering > %t.once.mlir
// RUN: dataflow-scheduler-opt -allow-unregistered-dialect %t.once.mlir -double-buffering > %t.twice.mlir
// RUN: diff %t.once.mlir %t.twice.mlir

// Inlined device spec (taken from sample_device.mlir). This is needed to make this idempotency test 
// possible without hardcoding the path to the device file, since the import handles paths that are 
// relative to the test files and the %t resolves to a temp dir.
#DDR = {
  kind = "DDR",
  size = 1073741824 //1GB
}
#DDR_MNILU = {
  ktdf_arch.transfer_granularity = array<i64: 64> //64B
}
#MNISU_DDR = {
  ktdf_arch.transfer_granularity = array<i64: 64> //64B
}

#L1 = {
  kind = "L1",
  size = 1048576 //1MB
}
#MNILU = {
  kind = "MNILU",
  load_store,
  dataflow_scheduler.double_buffer_last
}
#MNILU_L1 = {
  ktdf_arch.transfer_granularity = array<i64: 64> //64B
}
#MNISU = {
  kind = "MNISU",
  load_store,
  dataflow_scheduler.double_buffer_last
}
#L1_MNISU = {
  ktdf_arch.transfer_granularity = array<i64: 64> //64B
}

#SFU = {
  kind = "SFU",
  ktdf_arch.features = {
    ktdf_arch.feature.compute,
    ktdf_arch.feature.simd = { lanes = #ktdf_arch.map<f16 = 64> }
  }
}

#L1LU = {
  kind = "L1LU",
  load_store,
  ktdf_arch.features = { ktdf_arch.feature.simd = { splat, zero_pad } }
}
#L1LU_CORE_FIFO = {
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}
#L1LU_SUBCORE_FIFO = {
  ktdf_arch.transfer_granularity = array<i64: 64, 2>,
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}

#L1SU = {
  kind = "L1SU",
  load_store
}
#L1SU_CORE_FIFO = {
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}
#L1SU_SUBCORE_FIFO = {
  ktdf_arch.transfer_granularity = array<i64: 64, 2>,
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}

ktdf_arch.device @sample_device {
  %ddr = memory #DDR

  // First core.
  group { kind = "core" } share(%ddr) {
    // Private scratchpad memory.
    %l1 = memory #L1

    // DDR <-> L1 DMA units.
    %mnilu = exec_unit #MNILU
    %mnisu = exec_unit #MNISU

    // DMA datapaths: 256B/cy read, 128B/cy write.
    datapath #DDR_MNILU %ddr to %mnilu : memory, exec_unit
    datapath #MNILU_L1 %mnilu to %l1 : exec_unit, memory
    datapath #L1_MNISU %l1 to %mnisu : memory, exec_unit
    datapath #MNISU_DDR %mnisu to %ddr : exec_unit, memory

    // First sub-core.
    group  share(%l1) {
      // Compute units.
      %sfu = exec_unit #SFU

      // L1 Load/Store units.
      %l1lu = exec_unit #L1LU
      %l1su = exec_unit #L1SU

      // FIFO datapaths.
      datapath #L1LU_CORE_FIFO %l1 to %l1lu : memory, exec_unit
      datapath #L1LU_SUBCORE_FIFO %l1lu to %sfu : exec_unit, exec_unit
      datapath #L1SU_SUBCORE_FIFO %sfu to %l1su : exec_unit, exec_unit
      datapath #L1SU_CORE_FIFO %l1su to %l1 : exec_unit, memory
    }

    // Second sub-core.
    group  share(%l1) {
      // Compute units.
      %sfu = exec_unit #SFU

      // L1 Load/Store units.
      %l1lu = exec_unit #L1LU
      %l1su = exec_unit #L1SU

      // FIFO datapaths.
      datapath #L1LU_CORE_FIFO %l1 to %l1lu : memory, exec_unit
      datapath #L1LU_SUBCORE_FIFO %l1lu to %sfu : exec_unit, exec_unit
      datapath #L1SU_SUBCORE_FIFO %sfu to %l1su : exec_unit, exec_unit
      datapath #L1SU_CORE_FIFO %l1su to %l1 : exec_unit, memory
    }
  }
}

func.func @idempotent(%src: memref<64xf16, "DDR">,
                      %dst: memref<64xf16, "DDR">) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    ktdf.pipeline {
      %l1, %t1, %t2 = ktdf.private -> (
          memref<64xf16, "L1">,
          !ktdf.token, !ktdf.token) {
        %a = memref.alloc() : memref<64xf16, "L1">
        %tk1 = ktdf.create_token : !ktdf.token
        %tk2 = ktdf.create_token : !ktdf.token
        ktdf.private_yield %a, %tk1, %tk2
          : memref<64xf16, "L1">, !ktdf.token, !ktdf.token
      }
      ktdf.stage depends_in(none) depends_out(%t1) {
        ktdf.data_transfer from %src[%c0] size [64] to %l1[%c0] size [64]
          : memref<64xf16, "DDR">,
            memref<64xf16, "L1">
      }
      ktdf.stage depends_in(%t1) depends_out(%t2) {
        ktdf.data_transfer from %l1[%c0] size [64] to %dst[%c0] size [64]
          : memref<64xf16, "L1">,
            memref<64xf16, "DDR">
      }
    }
  }
  return
}
