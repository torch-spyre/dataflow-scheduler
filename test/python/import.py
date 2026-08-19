# RUN: python %s

# This is a trivial test that checks whether all the packages we expected to
# produce can be imported.

import mlir_scheduler
import mlir_scheduler.ir

import mlir_scheduler.dialects.arith
import mlir_scheduler.dialects.builtin
import mlir_scheduler.dialects.complex
import mlir_scheduler.dialects.func
import mlir_scheduler.dialects.index
import mlir_scheduler.dialects.linalg
import mlir_scheduler.dialects.math
import mlir_scheduler.dialects.memref
import mlir_scheduler.dialects.scf
import mlir_scheduler.dialects.tensor

import mlir_scheduler.dialects.ktdp

import mlir_scheduler.dialects.agen
import mlir_scheduler.dialects.dataflow
import mlir_scheduler.dialects.ktdf
import mlir_scheduler.dialects.ktdf_arch
import mlir_scheduler.dialects.ktdf_lowering
import mlir_scheduler.dialects.uniform
import mlir_scheduler.dialects.vectorchain
