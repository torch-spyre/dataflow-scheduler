// Minimal device definition for IndirectComputeGroupSplit tests.
// Provides:
//   %ddr  — global DDR memory (root of the MemoryTree)
//   %iab  — indirect address buffer (IAB) memory with entry_type = si32
//   %sfu  — compute execution unit (required so ResourceKinds::getComputeKind
//           does not deref a null result)

ktdf_arch.device @iab_device {
  %ddr = memory { kind = "DDR", size = 1073741824 }
  %iab = memory {
    kind = "IAB",
    ktdf_arch.features = {
      ktdf_arch.feature.indirect_address_buffer = { num_entries = 64, entry_type = si32 }
    },
    size = 256
  }
  %sfu = exec_unit {
    kind = "SFU",
    ktdf_arch.features = { ktdf_arch.feature.compute }
  }
}
