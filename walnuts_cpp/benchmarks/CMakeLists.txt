# Externally provided libraries
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_EXAMPLES    OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)

FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main) # need master for benchmark::benchmark

FetchContent_MakeAvailable(googlebenchmark)
add_executable(normal_nuts normal_nuts.cpp)
target_link_libraries(normal_nuts
  PRIVATE
  benchmark::benchmark
  walnuts::walnuts
  Eigen3::Eigen
)
