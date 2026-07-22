// Import file that declares more than one ktdf_arch.device, so a device name
// cannot be inferred unambiguously.
ktdf_arch.device @first attributes {version = 1, overridable = 0} {
  exec_unit
}
ktdf_arch.device @second attributes {version = 1, overridable = 0} {
  exec_unit
}
