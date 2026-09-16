// RUN: not dataflow-scheduler-opt -pass-pipeline="builtin.module(ktdflowering-to-dfir)" %s 2>&1 | FileCheck %s

// A tensor-semantics linalg.generic with a reduction dimension, handed straight to
// this pass.
//
// ReductionLoopExposure and MapReductionPartials rewrite these before the lowering
// runs, so nothing below it lowers a reduction. Reaching it means the pipeline is not
// the one this pass expects, and the elementwise path would take the reduction for an
// elementwise op and quietly give a wrong answer. So it is an error and not an assert:
// an assert is gone in a release build, and what follows it is the silent path.

// The generic is left where it is once the lowering refuses it, so what cannot be
// lowered after it reports too. Only the first message is the one under test.
// CHECK: error: tensor-semantics linalg.generic with a reduction dimension reached the DataflowIR lowering

#in  = affine_map<(d0, d1) -> (d0, d1)>
#out = affine_map<(d0, d1) -> (d1)>

module {
  ktdf_arch.device @sample_device attributes {} import("../../../../Dialect/KTDFArch/sample_device.mlir")

  func.func @reduction_reaches_lowering() attributes {grid = [2]} {
    %l1lu0 = dataflow.get_unit {core = 0 : i32, name = "C0-L1LU", type = "L1LU"} : index
    %l1lu1 = dataflow.get_unit {core = 1 : i32, name = "C1-L1LU", type = "L1LU"} : index
    %sfu0  = dataflow.get_unit {core = 0 : i32, name = "C0-SFU",  type = "SFU"}  : index
    %sfu1  = dataflow.get_unit {core = 1 : i32, name = "C1-SFU",  type = "SFU"}  : index
    %tile_id = ktdp.get_compute_tile_id : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %map_l1lu = uniform.def_immutable_mapping([%c0 -> %l1lu0], [%c1 -> %l1lu1]) : index
    %u_l1lu   = uniform.query_map(map:%map_l1lu, key:%tile_id) : index
    %map_sfu  = uniform.def_immutable_mapping([%c0 -> %sfu0],  [%c1 -> %sfu1])  : index
    %u_sfu    = uniform.query_map(map:%map_sfu,  key:%tile_id) : index

    %alloc_l1 = memref.alloc() : memref<1x64xf16, "L1">
    %out_fifo = ktdf.fifo.allocate() -> !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>

    ktdf_lowering.execute_on %u_sfu {
      %red_in = tensor.empty() : tensor<2x64xf16>
      %init   = tensor.empty() : tensor<64xf16>
      %result = linalg.generic {
        indexing_maps = [#in, #out],
        iterator_types = ["reduction", "parallel"]
      } ins(%red_in : tensor<2x64xf16>) outs(%init : tensor<64xf16>) {
      ^bb0(%in: f16, %out: f16):
        %v = arith.addf %in, %out : f16
        linalg.yield %v : f16
      } -> tensor<64xf16>
      ktdf.write_to_fifo %result, %out_fifo : tensor<64xf16>, <"SFU" -> "L1LU", 64xf16>
    }
    ktdf_lowering.execute_on %u_l1lu {
      ktdf.data_transfer from %out_fifo size [64] to %alloc_l1[%c0, %c0] size [1, 64] : !ktdf.fifo.slot<"SFU" -> "L1LU", 64xf16>, memref<1x64xf16, "L1">
    }
    return
  }
}
