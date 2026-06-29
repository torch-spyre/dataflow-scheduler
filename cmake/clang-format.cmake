find_program(CLANG_FORMAT 
  NAMES "clang-format" 
  PATHS 
    ${LLVM_DIR}/../../../bin
)

if(NOT CLANG_FORMAT)
    message(FATAL_ERROR "clang-format not found in  ${LLVM_DIR}/../../../bin Enable clang during llvm build")
else()
    message(STATUS "Found clang-format at ${CLANG_FORMAT}")
endif()

file(GLOB_RECURSE DataflowScheduler_SOURCE_FILES
  ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/*.cpp
  ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/*.hpp
  ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/*.h
  ${DataflowScheduler_SOURCE_DIR}/lib/*.cpp
  ${DataflowScheduler_SOURCE_DIR}/lib/*.hpp
  ${DataflowScheduler_SOURCE_DIR}/lib/*.h
  ${DataflowScheduler_SOURCE_DIR}/python/*.cpp
)

add_custom_target(format-dataflow-scheduler
  COMMAND ${CLANG_FORMAT}
    -style=file
    -i
    ${DataflowScheduler_SOURCE_FILES}
)

add_custom_target(check-dataflow-scheduler-format
  COMMENT "Checking clang-format"
  COMMAND ${CLANG_FORMAT} -style=file
            --output-replacements-xml
            ${DataflowScheduler_SOURCE_FILES}
            | tee ${CMAKE_BINARY_DIR}/check_format_scheduler_file.txt | grep -c "replacement " |
            tr -d "[:cntrl:]" && echo " replacements necessary"
    # WARNING: fix to stop with error if there are problems
    COMMAND ! grep -c "replacement "
      ${CMAKE_BINARY_DIR}/check_format_scheduler_file.txt > /dev/null
    COMMENT "Checking scheduler format compliance"
)

add_dependencies(check-all check-dataflow-scheduler-format)
