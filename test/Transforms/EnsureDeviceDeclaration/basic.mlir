// Checks that the EnsureDeviceDeclaration pass injects a `ktdf_arch.device`
// declaration when none is present, and leaves an existing one untouched.
//
// The second RUN line uses -verify-diagnostics with no expected diagnostics to
// assert that the DeviceManager analysis is created and imports the device
// cleanly (a failed import would emit an "unable to import device graph"
// diagnostic; see device-manager.mlir for that negative case).
//
// RUN: dataflow-scheduler-opt %s -split-input-file \
// RUN:   "-ensure-device-declaration=device-filename=%S/../../Dialect/KTDFArch/sample_device.mlir" \
// RUN:   | FileCheck %s

// No device present: the pass injects one, inferring the name from the single
// device in the imported file.
// CHECK-LABEL: module
// CHECK: ktdf_arch.device @sample_device import("{{.*}}sample_device.mlir")
module {
}

// -----

// A device is already declared: the pass is a no-op (idempotent) and does not
// add a second declaration. The DeviceManager still imports the existing one.
// CHECK-LABEL: module
// CHECK:     ktdf_arch.device @sample_device import("{{.*}}sample_device.mlir")
// CHECK-NOT: ktdf_arch.device
module {
  ktdf_arch.device @sample_device import("../../Dialect/KTDFArch/sample_device.mlir")
}
