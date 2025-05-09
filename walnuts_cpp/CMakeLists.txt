cmake_minimum_required(VERSION 3.20.2)
project(
    walnuts
    VERSION 0.1.0
    LANGUAGES C CXX)

include(FetchContent)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
cmake_policy(SET CMP0135 NEW)
cmake_policy(SET CMP0077 NEW)
if (CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_VERBOSE_MAKEFILE YES)
endif()
##########################
## Cmake Options        ##
##########################
# NOTE: I'm turning tests and benchmarks on by default
# Though normally these are off to help build speeds and remove bloat
option(WALNUTS_BUILD_TESTS "Build the test targets for the library" ON)
option(WALNUTS_BUILD_BENCHMARKS "Build the benchmarks target" ON)
option(WALNUTS_BUILD_DOXYGEN "Build the doxygen docs target" OFF)

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(-Wno-deprecated-declarations)
endif()
# Build Types
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}
    CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel tsan asan lsan msan ubsan"
    FORCE)

##########################
## Sanitizer Options    ##
##########################
# ThreadSanitizer
set(CMAKE_C_FLAGS_TSAN
    "-fsanitize=thread -g -O1 -march=native -mtune=native"
    CACHE STRING "Flags used by the C compiler during ThreadSanitizer builds."
    FORCE)
set(CMAKE_CXX_FLAGS_TSAN
    "-fsanitize=thread -g -O1 -march=native -mtune=native"
    CACHE STRING "Flags used by the C++ compiler during ThreadSanitizer builds."
    FORCE)

# AddressSanitize
set(CMAKE_C_FLAGS_ASAN
    "-fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -Og -march=native -mtune=native"
    CACHE STRING "Flags used by the C compiler during AddressSanitizer builds."
    FORCE)
set(CMAKE_CXX_FLAGS_ASAN
    "-fsanitize=address -fno-optimize-sibling-calls -fsanitize-address-use-after-scope -fno-omit-frame-pointer -g -Og -Wall -march=native -mtune=native"
    CACHE STRING "Flags used by the C++ compiler during AddressSanitizer builds."
    FORCE)

# LeakSanitizer
set(CMAKE_C_FLAGS_LSAN
    "-fsanitize=leak -fno-omit-frame-pointer -g -O1"
    CACHE STRING "Flags used by the C compiler during LeakSanitizer builds."
    FORCE)
set(CMAKE_CXX_FLAGS_LSAN
    "-fsanitize=leak -fno-omit-frame-pointer -ggdb3 -O1"
    CACHE STRING "Flags used by the C++ compiler during LeakSanitizer builds."
    FORCE)
if(APPLE)
    set(CMAKE_CXX_FLAGS_DEBUG
        "-fno-omit-frame-pointer -glldb -Og -DWALNUTS_DEBUG=true"
        CACHE STRING "Flags used by the C++ compiler during Debug builds."
        FORCE)
else()
    set(CMAKE_CXX_FLAGS_DEBUG
        "-fno-omit-frame-pointer -ggdb3 -Og -DWALNUTS_DEBUG=true"
        CACHE STRING "Flags used by the C++ compiler during Debug builds."
        FORCE)
endif()

if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_CXX_FLAGS_RELEASE
      "-O3 -march=native -mtune=native"
      CACHE STRING "Flags used by the C++ compiler during Release builds."
      FORCE)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

##########################
## Global Dependencies  ##
##########################
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/cmake_deps)


if (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(
      -DNO_FPRINTF_OUTPUT
      -Wall)
endif()


##########################
# Library Setup
# The next section sets everything up so that
#  other cmake libraries can use FetchContent
#  to pull in this library and include it as
#  dependency.
# i.e.
# FetchContent_Declare(walnuts
#         GIT_REPOSITORY https://github.com/bob-carpenter/walnuts
#         GIT_TAG main)
#
# FetchContent_MakeAvailable(walnuts)
#
# add_executable(test_nuts test.cpp)
# target_link_libraries(test_nuts PRIVATE Eigen3::Eigen walnuts::walnuts)
#
# walnuts::walnuts links a target to our include and any
#  shared object or static libraries we make in the future.
##########################

# Define the include directory for this library
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Optionally, specify install rules for your library
# This is necessary for FetchContent to work properly
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

# Create and install the CMake configuration files for your library
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${VERSION}
    COMPATIBILITY SameMajorVersion
)
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Config.cmake
    NAMESPACE walnuts::
    DESTINATION lib/cmake/${PROJECT_NAME}
)

target_include_directories(${PROJECT_NAME} INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_library(walnuts::walnuts ALIAS ${PROJECT_NAME})

##########################
##       Testing        ##
##########################
if (WALNUTS_BUILD_TESTS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

##########################
##      Benchmarks      ##
##########################
if (WALNUTS_BUILD_BENCHMARKS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)
endif()

# Check if Doxygen is installed
if (WALNUTS_BUILD_DOXYGEN)
    set(Eigen_INCLUDE_DIR "${Eigen3_SOURCE_DIR}")
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        # Set input and output files
        set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
        set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/docs/Doxyfile)

        # Request to configure the file
        configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
        message(STATUS "Doxygen build started")

        # Note: do not put 'ALL' keyword since you probably don't want documentation
        # to be generated every time you build the project.
        add_custom_target( doc_doxygen
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM )
    else()
        message("Doxygen need to be installed to generate the doxygen documentation")
    endif()
endif()
