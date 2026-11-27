// A device whose memories are named by ktdp memory spaces, as Spyre's are:
// a global memory and, per core, a core-local one of 2 MiB. Only the memory
// tree matters to the tests that use it, so there are no execution units.

#HBM = {
  kind = #ktdp.memory_space<global>,
  size = 17179869184
}

#CORE_LOCAL = {
  kind = #ktdp.memory_space<ct_local>,
  size = 2097152
}

ktdf_arch.device @ct_local_device {
  %hbm = memory #HBM
  group { kind = "core" } share(%hbm) {
    %core_local = memory #CORE_LOCAL
  }
}
