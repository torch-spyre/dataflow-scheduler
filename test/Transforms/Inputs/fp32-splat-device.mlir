//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

// Generic load, compute, and store units with a compute-local register file.
#DDR = {kind = "DDR", size = 17179869184}
#L1 = {kind = "L1", size = 2097152}
#SFU_REG = {kind = "SFU_REG", size = 2048}
#L1LU = {
  kind = "L1LU",
  ktdf_arch.features = {
    ktdf_arch.feature.load = {
      word_size = #ktdf_arch.map<"L1" = 1>,
      access_granularity = #ktdf_arch.map<"L1" = [
        {size_in_words = 128, align_in_words = 2},
        {size_in_words = 16, align_in_words = 2},
        {size_in_words = 2, align_in_words = 2}
      ]>
    },
    ktdf_arch.feature.simd = {splat, zero_pad}
  }
}
#SFU = {
  kind = "SFU",
  ktdf_arch.features = {
    ktdf_arch.feature.compute,
    ktdf_arch.feature.simd = {
      sub_simd_lanes = #ktdf_arch.map<f32 = 4>,
      lanes = #ktdf_arch.map<f16 = 64, f32 = 32, i32 = 32>
    }
  }
}
#L1SU = {
  kind = "L1SU",
  ktdf_arch.features = {
    ktdf_arch.feature.store = {
      word_size = #ktdf_arch.map<"L1" = 1>,
      access_granularity = #ktdf_arch.map<"L1" = [
        {size_in_words = 128, align_in_words = 128},
        {size_in_words = 16, align_in_words = 16},
        {size_in_words = 2, align_in_words = 2}
      ]>
    }
  }
}
#FIFO = {
  ktdf_arch.transfer_granularity = array<i64: 64, 2>,
  ktdf_arch.features = {
    ktdf_arch.feature.queue = {depth = 1024, ordered}
  }
}
ktdf_arch.device @sample_device {
  %ddr = memory #DDR
  group {kind = "core"} share(%ddr) {
    %l1 = memory #L1
    group share(%l1) {
      %compute = group {kind = "SFU_Block"} share() {
        %reg = memory #SFU_REG
        %unit = exec_unit #SFU
        yield %unit
      } -> exec_unit
      %load = exec_unit #L1LU
      %store = exec_unit #L1SU
      datapath #FIFO %l1 to %load : memory, exec_unit
      datapath #FIFO %load to %compute : exec_unit, exec_unit
      datapath #FIFO %compute to %store : exec_unit, exec_unit
      datapath #FIFO %store to %l1 : exec_unit, memory
    }
  }
}
