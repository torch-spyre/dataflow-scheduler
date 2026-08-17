add_dataflow_scheduler_library(DataflowSchedulerWrapProgramDFIR
    WrapProgramDFIR.cpp

  GROUP CONVERSION
  EXPORT_NAME WrapProgramDFIR

  DEPENDS
    DataflowSchedulerScheduleIRToDFIRPassesIncGen

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::DataflowDialect
    DataflowSchedulerDialects::SymbolDialect

  LINK_MLIR_LIBS
    MLIRFuncDialect
    MLIRIR
    MLIRPass
)
