// Checks that the `device-name` option controls the name of the injected
// `ktdf_arch.device` declaration.
//
// RUN: dataflow-scheduler-opt %s \
// RUN:   "-ensure-device-declaration=device-filename=%S/Inputs/multiple-devices.mlir device-name=second" \
// RUN:   | FileCheck %s

// CHECK-LABEL: module
// CHECK: ktdf_arch.device @second import("{{.*}}multiple-devices.mlir")
module {
}
