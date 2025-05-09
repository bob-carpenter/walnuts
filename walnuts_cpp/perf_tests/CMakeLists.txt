# 1) Find perf on the system
find_program(PERF_EXECUTABLE
  NAMES perf
  DOC "Path to the Linux perf tool"
  REQUIRED
)

# perf record default arguments
set(WALNUTS_PERF_RECORD_ARGS
    -F 99 -g --call-graph dwarf
    CACHE STRING "Default arguments for 'perf record'")
# perf report default arguments
set(WALNUTS_PERF_REPORT_ARGS
    -g --call-graph graph --stdio --stdio-color=always
    CACHE STRING "Default arguments for 'perf report'")
# perf record output directory
set(WALNUTS_PERF_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/data CACHE STRING "Output directory for perf data")

# 3) Define a function to generate a profiling target
function(add_run_perf TARGET_NAME)
  # everything after TARGET_NAME in the invocation will become extra perf-record flags
  set(extra_args "${ARGN}")

  # sanity check
  if (NOT TARGET ${TARGET_NAME})
    message(FATAL_ERROR
      "add_run_perf(): '${TARGET_NAME}' is not a known target")
  endif()

  # Ensure our output directory exists
  file(MAKE_DIRECTORY ${WALNUTS_PERF_OUTPUT_DIR})

  # Create a custom target called run_perf_<TARGET_NAME>
  add_custom_target(run_perf_${TARGET_NAME}
    DEPENDS ${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Profiling ${TARGET_NAME} with perf"
    VERBATIM
    COMMAND_EXPAND_LISTS
    # 1st command: perf record
    COMMAND ${PERF_EXECUTABLE}
            record
            ${WALNUTS_PERF_RECORD_ARGS}
            ${extra_args}
            -o ${WALNUTS_PERF_OUTPUT_DIR}/perf_${TARGET_NAME}.data
            -- "$<TARGET_FILE:${TARGET_NAME}>"

  )
endfunction()

function(add_report_perf TARGET_NAME)
  # everything after TARGET_NAME in the invocation will become extra perf-record flags
  set(extra_args "${ARGN}")

  # sanity check
  if (NOT TARGET ${TARGET_NAME})
    message(FATAL_ERROR
      "add_run_perf(): '${TARGET_NAME}' is not a known target")
  endif()
  # Create a custom target called run_perf_<TARGET_NAME>
  add_custom_target(report_perf_${TARGET_NAME}
    DEPENDS ${TARGET_NAME} run_perf_${TARGET_NAME}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Profiling ${TARGET_NAME} with perf"
    VERBATIM
    COMMAND_EXPAND_LISTS
    BYPRODUCTS ${WALNUTS_PERF_OUTPUT_DIR}/perf_${TARGET_NAME}.data
    # 2nd command: perf report
    COMMAND ${PERF_EXECUTABLE}
            report
            ${WALNUTS_PERF_REPORT_ARGS}
            -i ${WALNUTS_PERF_OUTPUT_DIR}/perf_${TARGET_NAME}.data
  )
endfunction()


## NOTE: Everything after here assumes that targets have been made already.

add_run_perf(test_nuts)
add_report_perf(test_nuts)

# ──────────────────────────────────────────────────────────────
# At the top of perf_tests/CMakeLists.txt (or your common include)
# ──────────────────────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
  FlameGraph
  GIT_REPOSITORY https://github.com/brendangregg/FlameGraph.git
  GIT_TAG        master
)

FetchContent_Populate(FlameGraph)
# NOTE: This is a workaround for non-standard CMake setups
set(FLAMEGRAPH_DIR      "${FETCHCONTENT_BASE_DIR}/flamegraph-src")
set(FLAMEGRAPH_PL       "${FLAMEGRAPH_DIR}/flamegraph.pl")
set(STACKCOLLAPSE_PERF_PL "${FLAMEGRAPH_DIR}/stackcollapse-perf.pl")
message(STATUS "FlameGraph dir: ${FLAMEGRAPH_DIR}")
message(STATUS "FlameGraph script: ${FLAMEGRAPH_PL}")
message(STATUS "Stackcollapse script: ${STACKCOLLAPSE_PERF_PL}")
# (Optional sanity check)
if (NOT EXISTS "${FLAMEGRAPH_PL}")
  message(FATAL_ERROR "FlameGraph script not found at ${FLAMEGRAPH_PL}")
endif()


function(add_flame_perf TARGET_NAME)
  # Ensure the perf-data and report targets are already defined
  if (NOT TARGET report_perf_${TARGET_NAME})
    message(FATAL_ERROR
      "add_flame_perf(): run_perf_${TARGET_NAME} must be added first")
  endif()

  # Build paths
  set(out_dir  ${WALNUTS_PERF_OUTPUT_DIR})
  set(data    ${out_dir}/perf_${TARGET_NAME}.data)
  set(raw     ${out_dir}/perf_${TARGET_NAME}.perf)
  set(folded  ${out_dir}/perf_${TARGET_NAME}.folded)
  set(svg     ${out_dir}/perf_${TARGET_NAME}.svg)

  add_custom_target(flamegraph_${TARGET_NAME}
#  DEPENDS run_perf_${TARGET_NAME}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating flamegraph for ${TARGET_NAME}"
  VERBATIM
  COMMAND_EXPAND_LISTS
  # Run everything through /bin/sh -c
  COMMAND /bin/sh -c
    " \
      perf script -i ${data} \
      | ${STACKCOLLAPSE_PERF_PL} \
      | ${FLAMEGRAPH_PL} --title='${TARGET_NAME}' \
      > ${svg} \
    "
)

  message(STATUS "Added flamegraph target 'flamegraph_${TARGET_NAME}' -> ${svg}")
endfunction()

# And finally hook it up:
add_flame_perf(test_nuts)
