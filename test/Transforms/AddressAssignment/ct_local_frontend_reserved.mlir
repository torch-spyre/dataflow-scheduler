// RUN: env -u DXP_LX_FRAC_AVAIL dataflow-scheduler-opt %s -address-assignment -split-input-file | FileCheck %s --check-prefix=DEFAULT
// RUN: env DXP_LX_FRAC_AVAIL=0.5 dataflow-scheduler-opt %s -address-assignment -split-input-file | FileCheck %s --check-prefix=HALF
// RUN: env DXP_LX_FRAC_AVAIL=1 dataflow-scheduler-opt %s -address-assignment -split-input-file | FileCheck %s --check-prefix=NONE

// The front end owns the start of every core-local (ct_local) memory, and
// allocations start above it. Its share is what torch-spyre's scratchpad
// planner computes from DXP_LX_FRAC_AVAIL (0.2 when unset):
// (size - 64 KiB of program and debug data) * (1 - DXP_LX_FRAC_AVAIL),
// truncated and rounded up to 128 bytes. For a 2 MiB memory that is
//   0.2: 2031616 * 0.8 = 1625292.8 -> 1625292 -> 1625344 bytes = 812672 f16
//   0.5: 2031616 * 0.5 = 1015808 bytes                          = 507904 f16
//   1:   0 bytes
// Addresses count elements of the allocation's element type. A global memory
// is not split, so its allocations start at 0 whatever the variable says.

module {
  ktdf_arch.device @ct_local_device attributes {} import("Inputs/ct_local_device.mlir")
  func.func @frontend_share() {
    // DEFAULT-LABEL: @frontend_share
    // DEFAULT: arith.constant 812672 : index
    // DEFAULT: arith.constant 812800 : index
    // DEFAULT: arith.constant 0 : index
    // HALF-LABEL: @frontend_share
    // HALF: arith.constant 507904 : index
    // HALF: arith.constant 508032 : index
    // NONE-LABEL: @frontend_share
    // NONE: arith.constant 0 : index
    // NONE: arith.constant 128 : index
    %0 = memref.alloc() : memref<128xf16, #ktdp.memory_space<ct_local>>
    %1 = memref.alloc() : memref<64xf16, #ktdp.memory_space<ct_local>>
    %2 = memref.alloc() : memref<512xf16, #ktdp.memory_space<global>>
    return
  }
}

// -----

// A memory's own `reserved` still applies; the larger of the two wins. Here
// the device reserves 1835008 bytes (917504 f16), more than the front end's
// default share, so allocations start there.

module {
  ktdf_arch.device @ct_local_device attributes {} import("Inputs/ct_local_device_reserved.mlir")
  func.func @spec_reserved_wins() {
    // DEFAULT-LABEL: @spec_reserved_wins
    // DEFAULT: arith.constant 917504 : index
    // NONE-LABEL: @spec_reserved_wins
    // NONE: arith.constant 917504 : index
    %0 = memref.alloc() : memref<128xf16, #ktdp.memory_space<ct_local>>
    return
  }
}
