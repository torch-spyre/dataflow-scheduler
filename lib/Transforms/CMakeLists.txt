add_subdirectory(Utils)

add_dataflow_scheduler_library(DataflowSchedulerTransforms
    PathExpansion/Materializer.cpp
    PathExpansion/Planner.cpp
    PathExpansion/PlannerFactories.cpp
    AddressAssignment.cpp
    AffineMinCanonicalization.cpp
    DoubleBuffering.cpp
    EnsureDeviceDeclaration.cpp
    NormalizeGridTo1D.cpp
    NormalizeSCFForLoops.cpp
    ParallelizeLoopsAcrossInstances.cpp
    PathExpansion.cpp
    ScalarBroadcastLegalization.cpp
    StripMineSCFForLoops.cpp
    TileSCFForLoops.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Transforms

  GROUP DIALECT
  EXPORT_NAME Transforms
  
  DEPENDS
    DataflowSchedulerTransformsPassesIncGen

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    DataflowScheduler::Analysis
    DataflowScheduler::ArchViews
    DataflowScheduler::KTDFAnalysis
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
    DataflowScheduler::KTDFTransforms
    DataflowScheduler::KTDFUtils
    DataflowScheduler::TransformUtils
    DataflowScheduler::Utils
)

mlir_target_link_libraries(DataflowSchedulerTransforms PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRPass
  MLIRTransformUtils
)
