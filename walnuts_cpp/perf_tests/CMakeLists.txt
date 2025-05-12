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
set(STACKCOLLAPSE_DTRACE_PL "${FLAMEGRAPH_DIR}/stackcollapse.pl")
set(STACKCOLLAPSE_PERF_PL "${FLAMEGRAPH_DIR}/stackcollapse-perf.pl")
message(STATUS "FlameGraph dir: ${FLAMEGRAPH_DIR}")
message(STATUS "FlameGraph script: ${FLAMEGRAPH_PL}")
message(STATUS "Stackcollapse script: ${STACKCOLLAPSE_PERF_PL}")
# (Optional sanity check)
if (NOT EXISTS "${FLAMEGRAPH_PL}")
  message(FATAL_ERROR "FlameGraph script not found at ${FLAMEGRAPH_PL}")
endif()

if(APPLE)
  # ————————————————————————————————————————————————————————
  # 1) Find the necessary tools
  # ————————————————————————————————————————————————————————
  find_program(XCTRACE_EXECUTABLE xcrun REQUIRED DOC "Xcode command-line tool for Instruments")
  find_program(DTRACE_EXECUTABLE dtrace REQUIRED DOC "macOS DTrace tool")

  # reuse your existing output dir
  set(WALNUTS_PERF_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/data)

  # ————————————————————————————————————————————————————————
  # 2) Time Profiler via Xcode Instruments
  # ————————————————————————————————————————————————————————
  function(add_run_instruments TARGET_NAME)
    if(NOT TARGET ${TARGET_NAME})
      message(FATAL_ERROR "add_run_instruments(): '${TARGET_NAME}' is not a known target")
    endif()
    file(MAKE_DIRECTORY ${WALNUTS_PERF_OUTPUT_DIR})

    add_custom_target(run_instruments_${TARGET_NAME}
      DEPENDS ${TARGET_NAME}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Profiling ${TARGET_NAME} with Xcode Time Profiler"
      VERBATIM
      # ← remove any stale trace
      COMMAND rm -rf ${WALNUTS_PERF_OUTPUT_DIR}/trace_${TARGET_NAME}.trace
      # ← now record a fresh one
      COMMAND ${XCTRACE_EXECUTABLE}
              xctrace record
              -t "Time Profiler"
              --output ${WALNUTS_PERF_OUTPUT_DIR}/trace_${TARGET_NAME}.trace
              --launch -- $<TARGET_FILE:${TARGET_NAME}>
    )  
  endfunction()
  # usage:
  add_run_instruments(test_nuts)


  # ————————————————————————————————————————————————————————
  # 3) Flamegraph via DTrace + Brendan Gregg scripts
  # ————————————————————————————————————————————————————————
  # assume you’ve already fetched FlameGraph via FetchContent
  set(STACKCOLLAPSE_DTRACE_PL "${FLAMEGRAPH_DIR}/stackcollapse.pl")
  set(FLAMEGRAPH_PL          "${FLAMEGRAPH_DIR}/flamegraph.pl")

  function(add_flame_dtrace TARGET_NAME)
    if(NOT TARGET ${TARGET_NAME})
      message(FATAL_ERROR "add_flame_dtrace(): '${TARGET_NAME}' is not a known target")
    endif()
    file(MAKE_DIRECTORY ${WALNUTS_PERF_OUTPUT_DIR})

    set(data_dtrace  ${WALNUTS_PERF_OUTPUT_DIR}/perf_${TARGET_NAME}.dtrace)
    set(folded       ${WALNUTS_PERF_OUTPUT_DIR}/perf_${TARGET_NAME}.folded)
    set(svg          ${WALNUTS_PERF_OUTPUT_DIR}/perf_${TARGET_NAME}.svg)

    add_custom_target(flamegraph_${TARGET_NAME}
      DEPENDS ${TARGET_NAME}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating flamegraph for ${TARGET_NAME} via DTrace"
      VERBATIM
      # 1) sample stacks at 99 Hz for 10 seconds
      COMMAND sudo ${DTRACE_EXECUTABLE}
              -x ustackframes=100
              -c "$<TARGET_FILE:${TARGET_NAME}>"
              -n "profile-999 /execname == \"$<TARGET_FILE_NAME:${TARGET_NAME}>\"/ { @[ustack()] = count(); } tick-10s { exit(0); }"
              -o ${data_dtrace}
      # 2) collapse and flamegraph
      COMMAND ${STACKCOLLAPSE_DTRACE_PL} ${data_dtrace} > ${folded}
      COMMAND ${FLAMEGRAPH_PL}          --title="${TARGET_NAME}" ${folded} > ${svg}
    )
    message(STATUS "Added flamegraph target 'flamegraph_${TARGET_NAME}' -> ${svg}")
  endfunction()

  add_flame_dtrace(test_nuts)
else()

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
endif()