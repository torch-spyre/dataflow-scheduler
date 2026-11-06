add_dataflow_scheduler_library(DataflowSchedulerWrapProgramDFIR
    MarkScheduledPrograms.cpp
    WrapProgramDFIR.cpp

  GROUP CONVERSION
  EXPORT_NAME WrapProgramDFIR

  DEPENDS
    DataflowSchedulerScheduleIRToDFIRPassesIncGen

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::DataflowDialect
    DataflowSchedulerDialects::SymbolDialect
    DataflowScheduler::Utils

  LINK_MLIR_LIBS
    MLIRFuncDialect
    MLIRIR
    MLIRPass
)
