// RUN: dataflow-scheduler-opt %s -apply-device-patterns='groups=pre_scheduling' | FileCheck %s

// ktdf.inherits_mapping looks at the op's own mapping or, failing that, the one
// of the nearest enclosing op that has one. Negated, it keeps the pattern below
// out of compute mapped to the array: the add there stays an add, and every
// other add becomes a sub.

#id = affine_map<(d0) -> (d0)>

#DDR = {
  kind = "DDR",
  size = 1048576
}
#SFU = {
  kind = "SFU"
}
#MPE_ARRAY = {
  kind = "MPE_ARRAY"
}

module {
  ktdf_arch.device @tiny_device {
    %ddr = memory #DDR
    %sfu = exec_unit #SFU
    %mpe_array = exec_unit #MPE_ARRAY

    patterns ["pre_scheduling"] {
      pdl.pattern : benefit(1) {
        %lhs = operand
        %rhs = operand
        %f16 = type : f16
        %add = operation "arith.addf" (%lhs, %rhs : !pdl.value, !pdl.value) -> (%f16 : !pdl.type)
        %array = attribute = "MPE_ARRAY"
        apply_native_constraint "ktdf.inherits_mapping"(%add, %array : !pdl.operation, !pdl.attribute) {isNegated = true}

        rewrite %add {
          %sub = operation "arith.subf" (%lhs, %rhs : !pdl.value, !pdl.value) -> (%f16 : !pdl.type)
          replace %add with %sub
        }
      }
    }
  }

  // CHECK-LABEL: func.func @mapped_to_array
  // CHECK:         arith.addf
  // CHECK-NOT:     arith.subf
  func.func @mapped_to_array(%a: tensor<64xf16>, %init: tensor<64xf16>) -> tensor<64xf16> {
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%a : tensor<64xf16>) outs(%init : tensor<64xf16>)
        attrs = {ktdf_arch.maps_to = "MPE_ARRAY"} {
    ^bb0(%x: f16, %out: f16):
      %s = arith.addf %x, %out : f16
      linalg.yield %s : f16
    } -> tensor<64xf16>
    return %r : tensor<64xf16>
  }

  // A list of units names the array too.
  // CHECK-LABEL: func.func @mapped_to_list_with_array
  // CHECK:         arith.addf
  // CHECK-NOT:     arith.subf
  func.func @mapped_to_list_with_array(%a: tensor<64xf16>, %init: tensor<64xf16>) -> tensor<64xf16> {
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%a : tensor<64xf16>) outs(%init : tensor<64xf16>)
        attrs = {ktdf_arch.maps_to = ["SFU", "MPE_ARRAY"]} {
    ^bb0(%x: f16, %out: f16):
      %s = arith.addf %x, %out : f16
      linalg.yield %s : f16
    } -> tensor<64xf16>
    return %r : tensor<64xf16>
  }

  // CHECK-LABEL: func.func @mapped_elsewhere
  // CHECK:         arith.subf
  // CHECK-NOT:     arith.addf
  func.func @mapped_elsewhere(%a: tensor<64xf16>, %init: tensor<64xf16>) -> tensor<64xf16> {
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%a : tensor<64xf16>) outs(%init : tensor<64xf16>)
        attrs = {ktdf_arch.maps_to = "SFU"} {
    ^bb0(%x: f16, %out: f16):
      %s = arith.addf %x, %out : f16
      linalg.yield %s : f16
    } -> tensor<64xf16>
    return %r : tensor<64xf16>
  }

  // CHECK-LABEL: func.func @unmapped
  // CHECK:         arith.subf
  // CHECK-NOT:     arith.addf
  func.func @unmapped(%a: tensor<64xf16>, %init: tensor<64xf16>) -> tensor<64xf16> {
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel"]}
        ins(%a : tensor<64xf16>) outs(%init : tensor<64xf16>) {
    ^bb0(%x: f16, %out: f16):
      %s = arith.addf %x, %out : f16
      linalg.yield %s : f16
    } -> tensor<64xf16>
    return %r : tensor<64xf16>
  }
}
