// Checks that the `device-name` option controls the name of the injected
// `ktdf_arch.device` declaration.
//
// RUN: dataflow-scheduler-opt %s \
// RUN:   "-ensure-device-declaration=device-filename=%S/../../Dialect/KTDFArch/sample_device.mlir device-name=sample_device" \
// RUN:   | FileCheck %s

// CHECK-LABEL: module
// CHECK: ktdf_arch.device @sample_device import("{{.*}}sample_device.mlir")
module {
}
