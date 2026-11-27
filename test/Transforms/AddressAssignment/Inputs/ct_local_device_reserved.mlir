// As ct_local_device.mlir, with the core-local memory reserving more than the
// front end's default share (1625344 bytes), so that its own `reserved` is the
// larger of the two.

#HBM = {
  kind = #ktdp.memory_space<global>,
  size = 17179869184
}

#CORE_LOCAL = {
  kind = #ktdp.memory_space<ct_local>,
  size = 2097152,
  reserved = 1835008
}

ktdf_arch.device @ct_local_device {
  %hbm = memory #HBM
  group { kind = "core" } share(%hbm) {
    %core_local = memory #CORE_LOCAL
  }
}
