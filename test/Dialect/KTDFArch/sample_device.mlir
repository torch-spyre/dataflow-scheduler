// RUN: dataflow-scheduler-opt %s

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
  ktdf_arch.features = { 
    ktdf_arch.feature.load = {
      word_size = #ktdf_arch.map<"DDR" = 64, "L1" = 64>,
      access_granularity = #ktdf_arch.map<
        "DDR" = [{size_in_words = 1, align_in_words = 1}],
        "L1" = [{size_in_words = 1, align_in_words = 1}]
      >
    }
  },
  dataflow_scheduler.double_buffer_last
}
#MNILU_L1 = {
  ktdf_arch.transfer_granularity = array<i64: 64> //64B
}
#MNISU = {
  kind = "MNISU",
  ktdf_arch.features = { 
    ktdf_arch.feature.store = {
      word_size = #ktdf_arch.map<"DDR" = 64, "L1" = 64>,
      access_granularity = #ktdf_arch.map<
        "DDR" = [{size_in_words = 1, align_in_words = 1}],
        "L1" = [{size_in_words = 1, align_in_words = 1}]
      >
    }
  },
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

#SFU_REG = {
  kind = "SFU_REG",
  size = 2048 //2KB
}

#L1LU = {
  kind = "L1LU",
  ktdf_arch.features = { 
    ktdf_arch.feature.load = {
      access_granularity = #ktdf_arch.map<
        "L1" = [
          {size_in_words = 64, align_in_words = 64}, 
          {size_in_words = 2, align_in_words = 2}
        ]
      >
    },
    ktdf_arch.feature.simd = { splat, zero_pad } 
  }
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
  ktdf_arch.features = {
    ktdf_arch.feature.store = {
      access_granularity = #ktdf_arch.map<
        "L1" = [
          {size_in_words = 64, align_in_words = 64}, 
          {size_in_words = 2, align_in_words = 2}
        ]
      >
    }
  }
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
      %sfu = group { kind = "SFU_Block" } share() {
         %sfp_reg = memory #SFU_REG
         %sfp_unit = exec_unit #SFU
         yield %sfp_unit
      } -> exec_unit

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
      %sfu = group { kind = "SFU_Block" } share() {
         %sfp_reg = memory #SFU_REG
         %sfp_unit = exec_unit #SFU
         yield %sfp_unit
      } -> exec_unit

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
