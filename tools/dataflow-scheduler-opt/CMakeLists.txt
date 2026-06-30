add_dataflow_scheduler_tool(dataflow-scheduler-opt
    dataflow-scheduler-opt.cpp

  EXPORT_NAME opt

  SUPPORT_PLUGINS
)
target_link_libraries(dataflow-scheduler-opt PRIVATE
  DataflowScheduler::RegisterEverything
)

mlir_target_link_libraries(dataflow-scheduler-opt PRIVATE
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRRegisterAllExtensions
)
llvm_update_compile_flags(dataflow-scheduler-opt)
export_executable_symbols_for_plugins(dataflow-scheduler-opt)
