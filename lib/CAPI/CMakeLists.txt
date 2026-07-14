add_dataflow_scheduler_public_c_api_library(DataflowSchedulerCAPIRegisterEverything
    RegisterEverything.cpp

  PARTIAL_SOURCES_INTENDED
  EXPORT_NAME CAPIRegisterEverything

  LINK_LIBS PUBLIC
    DataflowScheduler::RegisterEverything
)
