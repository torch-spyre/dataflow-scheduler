// RUN: dataflow-scheduler-opt %s -apply-device-patterns='groups=pre_mapping' | FileCheck %s

// ktdf.insert_via routes one operand of one op through a ktdf.via, placed just
// before the op. Other readers of the value keep reading it directly. The
// pattern below sends the first input of a contraction through the array's
// memory, and maps the contraction so it does not match again.

#mk = affine_map<(m, n, k) -> (m, k)>
#kn = affine_map<(m, n, k) -> (k, n)>
#mn = affine_map<(m, n, k) -> (m, n)>

#DDR = {
  kind = "DDR",
  size = 1048576
}
#ARRAY_MEM = {
  kind = "ARRAY_MEM",
  size = 8192
}
#MPE_ARRAY = {
  kind = "MPE_ARRAY"
}

module {
  ktdf_arch.device @tiny_device {
    %ddr = memory #DDR
    %array_mem = memory #ARRAY_MEM
    %mpe_array = exec_unit #MPE_ARRAY

    patterns ["pre_mapping"] {
      pdl.pattern : benefit(1) {
        %lhs = operand
        %rhs = operand
        %init = operand
        %results = types
        %contraction = operation "linalg.generic"
          (%lhs, %rhs, %init : !pdl.value, !pdl.value, !pdl.value)
          -> (%results : !pdl.range<type>)
        apply_native_constraint "linalg.is_contraction"(%contraction : !pdl.operation)
        apply_native_constraint "ktdf_arch.mapped_to"(%contraction : !pdl.operation) : !pdl.attribute {isNegated = true}

        rewrite {
          %hops = attribute = ["ARRAY_MEM"]
          apply_native_rewrite "ktdf.insert_via"(%contraction, %lhs, %hops : !pdl.operation, !pdl.value, !pdl.attribute)
          %array = attribute = "MPE_ARRAY"
          apply_native_rewrite "ktdf_arch.map_to"(%contraction, %array : !pdl.operation, !pdl.attribute)
        }
      }
    }
  }

  // CHECK-LABEL: func.func @matmul
  // CHECK-SAME:    (%[[A:[^:]*]]: tensor<64x64xf16>, %[[B:[^:]*]]: tensor<64x64xf16>
  // CHECK:         %[[A_MEM:.*]] = ktdf.via["ARRAY_MEM"] %[[A]] : tensor<64x64xf16>
  // CHECK-NEXT:    %[[C:.*]] = linalg.generic
  // CHECK-SAME:      ins(%[[A_MEM]], %[[B]] :
  // CHECK-SAME:      ktdf_arch.maps_to = "MPE_ARRAY"
  // CHECK:         return %[[C]], %[[A]]
  func.func @matmul(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>,
                    %init: tensor<64x64xf16>)
      -> (tensor<64x64xf16>, tensor<64x64xf16>) {
    %c = linalg.generic {
        indexing_maps = [#mk, #kn, #mn],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
        outs(%init : tensor<64x64xf16>) {
    ^bb0(%x: f16, %y: f16, %acc: f16):
      %mul = arith.mulf %x, %y : f16
      %sum = arith.addf %mul, %acc : f16
      linalg.yield %sum : f16
    } -> tensor<64x64xf16>
    return %c, %a : tensor<64x64xf16>, tensor<64x64xf16>
  }
}
