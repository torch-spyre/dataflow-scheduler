add_dataflow_scheduler_library(DataflowSchedulerKTIRToScheduleIR
    ComputeGroupExtraction.cpp
    ConstructThreeStagePipeline.cpp
    KTIRLegalityCheck.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR

  GROUP CONVERSION
  EXPORT_NAME KTIRToScheduleIR

  DEPENDS
    DataflowSchedulerKTIRToScheduleIRPassesIncGen

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    KTIR::SpyreOpDialect
    DataflowScheduler::Analysis
    DataflowScheduler::ArchViews
    DataflowScheduler::KTDFAnalysis
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
    DataflowScheduler::KTDFTransforms
    DataflowScheduler::KTDFUtils
    DataflowScheduler::TransformUtils
    DataflowScheduler::Utils

  LINK_MLIR_LIBS
    MLIRIR
    MLIRArithDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTransformUtils
)
