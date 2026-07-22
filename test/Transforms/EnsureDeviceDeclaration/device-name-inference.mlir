// Checks the failure modes of device-name inference (getDeviceName), which runs
// when a device file is given but no device-name option.
//
// These diagnostics are emitted from a throwaway MLIRContext created while
// reading the import file (and case PARSE produces nested diagnostics that
// point at that other file), so they cannot be intercepted with
// -verify-diagnostics. We match them on stderr with FileCheck instead. `not` is
// used because each case makes the pass fail.

// The import file does not exist.
// RUN: not dataflow-scheduler-opt %s \
// RUN:   "-ensure-device-declaration=device-filename=%S/Inputs/does-not-exist.mlir" \
// RUN:   2>&1 | FileCheck %s --check-prefix=NOFILE
// NOFILE: error: unable to import device: cannot open input file '{{.*}}does-not-exist.mlir'

// The import file does not parse as valid MLIR. The parse error is a nested
// diagnostic reported against the import file, not this input.
// RUN: not dataflow-scheduler-opt %s \
// RUN:   "-ensure-device-declaration=device-filename=%S/Inputs/not-mlir.mlir" \
// RUN:   2>&1 | FileCheck %s --check-prefix=PARSE
// PARSE: {{.*}}not-mlir.mlir:{{[0-9]+}}:{{[0-9]+}}: error:

// The import file parses but contains no device.
// RUN: not dataflow-scheduler-opt %s \
// RUN:   "-ensure-device-declaration=device-filename=%S/Inputs/no-device.mlir" \
// RUN:   2>&1 | FileCheck %s --check-prefix=NODEVICE
// NODEVICE: error: no device in import file

// The import file contains more than one device.
// RUN: not dataflow-scheduler-opt %s \
// RUN:   "-ensure-device-declaration=device-filename=%S/Inputs/multiple-devices.mlir" \
// RUN:   2>&1 | FileCheck %s --check-prefix=MULTI
// MULTI: error: multiple devices in import file

module {
}
