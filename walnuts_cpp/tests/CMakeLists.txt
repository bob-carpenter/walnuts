# Externally provided libraries
set(gtest_build_tests    OFF CACHE BOOL "" FORCE)
set(gtest_build_samples  OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)

FetchContent_MakeAvailable(googletest)

