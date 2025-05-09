# Externally provided libraries
set(gtest_build_tests    OFF CACHE BOOL "" FORCE)
set(gtest_build_samples  OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)

FetchContent_MakeAvailable(googletest)

add_executable(test_nuts test.cpp)
target_link_libraries(test_nuts PRIVATE Eigen3::Eigen walnuts::walnuts)

add_executable(test_nuts_perf test.cpp)
target_link_libraries(test_nuts_perf PRIVATE Eigen3::Eigen walnuts::walnuts)
target_compile_options(test_nuts_perf PRIVATE -gdwarf -O3)

# find perf on the user’s PATH
find_program(PERF_EXECUTABLE
  NAMES perf
  DOC "Path to the Linux `perf` tool"
  REQUIRED)

# make sure test_nuts gets built first
add_custom_target(perf_test_nuts
  DEPENDS test_nuts_perf
  # 1) record with call-graph (dwarf) symbols into perf.data
  COMMAND ${PERF_EXECUTABLE}
          record
          -F 99                            # sample at 99 Hz
          -g                              # capture call graph
          --call-graph dwarf              # use DWARF (more accurate)
          -o test_nuts_perf.data                    # output file
          -- $<TARGET_FILE:test_nuts_perf>     # run the test binary

  # 2) immediately run report on that data
  COMMAND ${PERF_EXECUTABLE}
          report
          -i test_nuts_perf.data                    # read the data file
          --stdio                         # print to stdout
          -g                              # show the call-graph view
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Profiling test_nuts with perf; run 'cmake --build . --target perf_test_nuts'"
  VERBATIM
)
