#!/bin/bash

#
# Copyright 2026 The KTIR Scheduler Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Your mlir test file need to have at least one of the
# following RUN lines:
#
# RUN: scheduler --transform %s

# ---------------------------------------------------------------------------
# Derive the scheduler binary path and LLVM monorepo source root from a
# build directory.  Both are required to run the script.
#   CMakeCache.txt  →  LLVM_DIR  →  LLVMConfig.cmake  →  LLVM_BUILD_MAIN_SRC_DIR
#   LLVM_PROJ_SRC = parent of LLVM_BUILD_MAIN_SRC_DIR  (the llvm-project root)
# ---------------------------------------------------------------------------
function derivePathsFromBuildDir() {
  local build_dir="$1"
  local cmake_cache="${build_dir}/CMakeCache.txt"

  if [[ ! -f "${cmake_cache}" ]]; then
    echo "Error: CMakeCache.txt not found in '${build_dir}'"
    exit 1
  fi

  KTIRScheduler="${build_dir}/bin/dataflow-scheduler"

  # Extract the LLVM cmake directory recorded during the build.
  # CMakeCache.txt line format:  LLVM_DIR:PATH=<path>
  local llvm_cmake_dir
  llvm_cmake_dir=$(grep -m1 '^LLVM_DIR:PATH=' "${cmake_cache}" | cut -d= -f2-)
  if [[ -z "${llvm_cmake_dir}" ]]; then
    echo "Error: LLVM_DIR not found in '${cmake_cache}'"
    exit 1
  fi

  local llvm_config="${llvm_cmake_dir}/LLVMConfig.cmake"
  if [[ ! -f "${llvm_config}" ]]; then
    echo "Error: LLVMConfig.cmake not found at '${llvm_config}'"
    exit 1
  fi

  # LLVMConfig.cmake contains a line like:
  #   set(LLVM_BUILD_MAIN_SRC_DIR "/path/to/llvm-project/llvm")
  local llvm_src_dir
  llvm_src_dir=$(grep -m1 'LLVM_BUILD_MAIN_SRC_DIR' "${llvm_config}" \
                   | sed 's/.*"\(.*\)".*/\1/')
  if [[ -z "${llvm_src_dir}" ]]; then
    echo "Error: LLVM_BUILD_MAIN_SRC_DIR not found in '${llvm_config}'"
    exit 1
  fi

  # LLVM_BUILD_MAIN_SRC_DIR points to the llvm/ sub-directory of the
  # monorepo, so the monorepo root (which contains mlir/) is its parent.
  LLVM_PROJ_SRC="$(dirname "${llvm_src_dir}")"
}

# Usage
function printUsage() {
  echo "Usage: $(basename $0): "
  echo "          Run command:"
  echo "          ./scripts/test_update_filecheck.sh --build-dir=<path> --testfiles=<file_name.mlir>"
  echo
  echo "          --build-dir   path to the scheduler build directory (required). The"
  echo "                        dataflow-scheduler binary and LLVM source tree are derived"
  echo "                        automatically from it via CMakeCache.txt."
  echo "          --testfiles/-f name of the test file(s). This option can accept a directory too."
  echo
  echo "          Example:"
  echo "          ./scripts/test_update_filecheck.sh --build-dir=build -f=test/Dialects/basic.mlir"
  echo "          ./scripts/test_update_filecheck.sh --build-dir=build --testfiles=test/Dialects/basic.mlir"
  echo
  echo "          Note that your mlir test file need to have at least one of the following RUN lines:"
  echo "          // RUN: dataflow-scheduler <specific-pass-option> %s"
}

function parseCommandLine() {
  local build_dir=""

  for i in "$@"; do
    case $i in
      -f=*|--testfiles=*|--testfile=*)
        TESTFILES=${i#*=}
        ;;
      --build-dir=*)
        build_dir="${i#*=}"
        ;;
      -h|--help)
        printUsage;
        exit;
        ;;
      *)
        echo "Unrecognized option: $i"
        printUsage;
        exit 1;
        ;;
   esac
  done
  [[ -z "$@" ]] && printUsage && exit 1

  if [[ -z "${build_dir}" ]]; then
    echo "Error: --build-dir is required"
    printUsage
    exit 1
  fi

  derivePathsFromBuildDir "${build_dir}"
}

function run() {

  files="$(find $1 -name '*.mlir')"
  for file in $files ; do
    echo "test file:" $file
    # RUN lines
    run_scheduler=$(grep "RUN:" "$file" | grep "scheduler" )
    if [[ "$(grep "RUN:" "$file" | wc -l)" -eq "0" ]]; then
      echo "RUN line is needed"
      return 1
    fi

    # Check if this is a round-trip test (contains multiple scheduler invocations)
    is_roundtrip=0
    if echo "${run_scheduler}" | grep -q "scheduler.*|.*scheduler"; then
      is_roundtrip=1
      echo "Detected round-trip test"
    fi

#    echo "run command: $run_scheduler"

    # Capture the binary name (first word after RUN:) before stripping it
    run_binary=$(echo "${run_scheduler}" | sed -e 's/\/\///' | sed -e 's/RUN://' | sed -e 's/^ *//' | awk '{print $1}')

    # get the actual command (flags only — strip binary name, pipes, FileCheck, %s)
    if [ ${is_roundtrip} -eq 1 ]; then
      # For round-trip tests, extract only the flags before the first pipe
      run_scheduler=$(echo "${run_scheduler}" | sed -e 's/\/\///' | sed -e 's/RUN://' | sed -e 's/|.*//' | sed -e "s/${run_binary}//" | sed -e 's/%s//' | sed -e 's/^ *//')
    else
      # For non-round-trip tests, remove pipes and FileCheck
      run_scheduler=$(echo "${run_scheduler}" | sed -e 's/\/\///' | sed -e 's/RUN://' | sed -e 's/|//g' | sed -e 's/FileCheck.*//' | sed -e "s/${run_binary}//" | sed -e 's/%s//' | sed -e 's/^ *//')
    fi

#    echo "run command: $run_scheduler"

    # trim
    run_scheduler=$(echo "${run_scheduler}" | sed -e 's/^ *//' | sed -e 's/ *$//')

    echo "run command: $run_scheduler"

    # Delete all RUN lines (they will be regenerated)
    sed -e "/^\/\/ RUN:/d" < $file > /tmp/out && mv /tmp/out $file

    sed -e "/\/\/ NOTE: Assertions have been autogenerated by/d" < $file > /tmp/out && mv /tmp/out $file
    sed -e "/\/\/ CHECK/d" < $file > /tmp/out && mv /tmp/out $file
    if [ ! -z ${KTIRScheduler} ]; then
      if [ ! -z "${run_scheduler}" ]; then
        echo "${KTIRScheduler} ${run_scheduler} ${file} | $LLVM_PROJ_SRC/mlir/utils/generate-test-checks.py > /tmp/out_sent"
        eval ${KTIRScheduler} ${run_scheduler} ${file} | $LLVM_PROJ_SRC/mlir/utils/generate-test-checks.py > /tmp/out_sent
#       sed -e "s/\/\/ CHECK:/\/\/ CHECK:/" < ${file} > /tmp/out && mv /tmp/out /tmp/out_sent
        sed -e "s/\/\/ CHECK:      /\/\/ CHECK-NEXT:/" < /tmp/out_sent > /tmp/out && mv /tmp/out /tmp/out_sent
      else
        # Empty command means just parse the file (no transformation passes)
        echo "${KTIRScheduler} ${file} | $LLVM_PROJ_SRC/mlir/utils/generate-test-checks.py > /tmp/out_sent"
        eval ${KTIRScheduler} ${file} | $LLVM_PROJ_SRC/mlir/utils/generate-test-checks.py > /tmp/out_sent
        sed -e "s/\/\/ CHECK:      /\/\/ CHECK-NEXT:/" < /tmp/out_sent > /tmp/out && mv /tmp/out /tmp/out_sent
      fi
    else
      echo "scheduler not found!"
      return 1
    fi
    
    # For compute-group-extraction pass, add top-level module checks
    if echo "${run_scheduler}" | grep -q "compute-group-extraction"; then
      echo "Adding top-level module checks for compute-group-extraction..."
      # Replace first CHECK-LABEL:   module { with CHECK: module { and CHECK:   module {
      awk '!done && /^\/\/ CHECK-LABEL:   module \{$/ {print "// CHECK: module {"; print "// CHECK:   module {"; done=1; next} {print}' /tmp/out_sent > /tmp/out && mv /tmp/out /tmp/out_sent
      # Replace remaining CHECK-LABEL with CHECK for other child modules
      sed -e 's/^\/\/ CHECK-LABEL:   module {$/\/\/ CHECK:   module {/' < /tmp/out_sent > /tmp/out && mv /tmp/out /tmp/out_sent
      # Add closing brace for top-level module without extra newline
      printf "// CHECK: }" >> /tmp/out_sent
    fi
    
    echo "" | cat - $file > /tmp/out && mv /tmp/out $file

#    echo "file after:";    eval cat $file
#    echo "out_sent:"; eval cat /tmp/out_sent
    if [ ${is_roundtrip} -eq 1 ]; then
        # For round-trip tests, add the extra scheduler invocation
        if [ ! -z "${run_scheduler}" ]; then
            echo "// RUN: ${run_binary} ${run_scheduler} %s | ${run_binary} ${run_scheduler} | FileCheck %s" | cat - /tmp/out_sent | cat - $file > /tmp/out && mv /tmp/out $file
        else
            echo "// RUN: ${run_binary} %s | ${run_binary} | FileCheck %s" | cat - /tmp/out_sent | cat - $file > /tmp/out && mv /tmp/out $file
        fi
        sed -e "/NOTE: Assertions have been autogenerated by/d" < $file > /tmp/out && mv /tmp/out $file
    else
        # For non-round-trip tests
        if [ ! -z "${run_scheduler}" ]; then
            echo "// RUN: ${run_binary} ${run_scheduler} %s | FileCheck %s" | cat - /tmp/out_sent | cat - $file > /tmp/out && mv /tmp/out $file
            sed -e "/NOTE: Assertions have been autogenerated by/d" < $file > /tmp/out && mv /tmp/out $file
        else
            echo "// RUN: ${run_binary} %s | FileCheck %s" | cat - /tmp/out_sent | cat - $file > /tmp/out && mv /tmp/out $file
            sed -e "/NOTE: Assertions have been autogenerated by/d" < $file > /tmp/out && mv /tmp/out $file
        fi
    fi

    # No need to replace CHECK prefix anymore since we're using default CHECK

    # remove the auto-generated comment block from generate-test-checks.py
    sed -e '/^\/\/ The script is designed to make adding checks to$/,/^\/\/ minimized and named to reflect the test intent\.$/d' \
        < $file > /tmp/out && mv /tmp/out $file

  done
}

# main
function main() {
  parseCommandLine "$@"
  run "$TESTFILES"
  return 0
}

main "$@"
