// Checks that the pass instantiates the ktdf_arch::DeviceManager analysis and
// drives it to import the declared device. When the requested device cannot be
// found in the imported file, the DeviceManager emits its own diagnostic --
// observing that diagnostic proves the manager was created and invoked.
//
// The injected declaration carries an unknown location, so the import failure
// is reported at an unknown source location.
//
// RUN: dataflow-scheduler-opt %s \
// RUN:   "-ensure-device-declaration=device-filename=%S/../../Dialect/KTDFArch/sample_device.mlir device-name=nonexistent" \
// RUN:   -verify-diagnostics

// expected-error@unknown {{'ktdf_arch.device' op unable to import device graph: no device named 'nonexistent'}}
module {
}
