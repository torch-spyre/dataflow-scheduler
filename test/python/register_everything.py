# RUN: python %s %test_source_root

import sys
from pathlib import Path

test_source_root = Path(sys.argv[1])

from mlir_scheduler.ir import DialectRegistry, Context, Module
from mlir_scheduler.passmanager import PassManager

from mlir_scheduler._mlir_libs._dataflow_scheduler import (
    register_all_dialects,
    register_all_extensions,
    register_all_passes,
)

register_all_passes()

registry = DialectRegistry()
register_all_dialects(registry)
context = Context()
context.append_dialect_registry(registry)
register_all_extensions(registry)
context.load_all_available_dialects()

with context:
    input_path = test_source_root / "Transforms" / "PathExpansion" / "basic.mlir"
    module = Module.parseFile(str(input_path))

    pm = PassManager.parse("builtin.module(path-expansion)")
    pm.run(module.operation)

    module.dump()
