// RUN: dataflow-scheduler-opt --ktir-legality-check --verify-diagnostics %s
#map = affine_map<(d0) -> (d0)>
// A cast is legal only if an address computation is what reads it.
func.func @cast_for_anything_else(%idx: tensor<4xi32>, %o: tensor<4xi32>,
                                  %n: index) -> tensor<4xi32> {
  %r = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]}
       ins(%idx : tensor<4xi32>) outs(%o : tensor<4xi32>) {
  ^bb0(%in: i32, %out: i32):
    // expected-error @+1 {{V1 only supports add/mul/sub compute ops; found unsupported compute op}}
    %c = arith.index_castui %n : index to i32
    linalg.yield %c : i32
  } -> tensor<4xi32>
  return %r : tensor<4xi32>
}
