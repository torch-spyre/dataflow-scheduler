// Checks that the pass fails when no device is declared and no device file is
// provided to fall back on.
//
// RUN: dataflow-scheduler-opt %s -ensure-device-declaration -verify-diagnostics

// expected-error @below {{no 'ktdf_arch.device' is present in the module and no device file name is provided to ensure-device-declaration pass}}
module {
}
