cmake_minimum_required(VERSION 3.14)
project(walnuts
  VERSION 0.1.0
  DESCRIPTION "Header-only NUTS sampler library"
  LANGUAGES CXX
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
if (CMAKE_BUILD_TYPE MATCHES DEBUG)
  set(CMAKE_VERBOSE_MAKEFILE YES)
endif()

##########################
## Apple fix            ##
##########################
if(APPLE)
  execute_process(
    COMMAND xcrun --sdk macosx --show-sdk-path
    OUTPUT_VARIABLE _sdk_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(CMAKE_OSX_SYSROOT "${_sdk_dir}" CACHE STRING "macOS SDK root")
  set(CMAKE_OSX_ARCHITECTURES "x86_64" CACHE STRING "Build for Intel")
endif()

##########################
## Cmake Options        ##
##########################
option(WALNUTS_BUILD_EXAMPLES "Build the example targets for the library" ON)
option(WALNUTS_BUILD_STAN "Build Stan example with BridgeStan" OFF)
option(WALNUTS_BUILD_TESTS "Build the test targets for the library" ON)
# Build Types
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING
    "Choose the type of build, options are: None Debug Release"
    FORCE)

##########################
## Global Dependencies  ##
##########################
find_package(Git REQUIRED)

# where to put Eigen
set(EIGEN_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/eigen")
set(EIGEN_TAG 3.4.0)
# only clone on first configure
if(NOT EXISTS "${EIGEN_SRC_DIR}/Eigen")
  message(STATUS "Cloning Eigen ${EIGEN_TAG}…")
  execute_process(
    COMMAND ${GIT_EXECUTABLE} clone --branch ${EIGEN_TAG} --depth 1 -- https://gitlab.com/libeigen/eigen.git ${EIGEN_SRC_DIR}
    RESULT_VARIABLE _git_result
  )
  if(NOT _git_result EQUAL 0)
    message(FATAL_ERROR "Git clone of Eigen failed")
  endif()
endif()

# now expose it as a pure‐header interface target
add_library(Eigen3::Eigen INTERFACE IMPORTED)
target_include_directories(Eigen3::Eigen INTERFACE
  "${EIGEN_SRC_DIR}"
)

include(FetchContent)
set(WALNUTS_BRIDGESTAN_TAG "1d4ff6a10d572ee22ed26b0274059bb4fe885187" CACHE STRING "Version of BridgeStan to use")
if(WALNUTS_BUILD_STAN)
    FetchContent_Declare(
            bridgestan
            GIT_REPOSITORY https://github.com/roualdes/bridgestan.git
            GIT_TAG ${WALNUTS_BRIDGESTAN_TAG}
    )
    FetchContent_MakeAvailable(bridgestan)

    add_library(BridgeStan::BridgeStan INTERFACE IMPORTED)
    target_include_directories(BridgeStan::BridgeStan INTERFACE
      "${CMAKE_BINARY_DIR}/_deps/bridgestan-src/src"
    )

  add_executable(test_stan ${CMAKE_CURRENT_SOURCE_DIR}/examples/test_stan.cpp)
  target_link_libraries(test_stan PRIVATE Eigen3::Eigen nuts::nuts BridgeStan::BridgeStan)
  target_compile_options(test_stan PRIVATE -O3 -Wall)
endif()

#############################
## Making Walnuts Library  ##
#############################

# 1) Define your header-only interface target
add_library(walnuts INTERFACE)
target_include_directories(walnuts INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>  # for builds
  $<INSTALL_INTERFACE:include>                            # for installs
)
# 2) Create the “namespace” alias
add_library(nuts::nuts ALIAS walnuts)

##########################
##       Example        ##
##########################
if (WALNUTS_BUILD_EXAMPLES)
  add_executable(test_nuts ${CMAKE_CURRENT_SOURCE_DIR}/examples/test.cpp)
  target_link_libraries(test_nuts PRIVATE Eigen3::Eigen nuts::nuts)
  target_compile_options(test_nuts PRIVATE -O3 -Wall)
endif()

##########################
##       Extras         ##
##########################
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extras/CMakeLists.txt")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extras")
endif()

##########################
##       Tests         ##
##########################
if (WALNUTS_BUILD_TESTS)
  enable_testing()
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
endif()

##########################
##       Format         ##
##########################
# Define list of files to format (you can make this recursive if desired)
file(GLOB_RECURSE ALL_SOURCE_FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

find_program(CLANG_FORMAT_BIN clang-format)

if (CLANG_FORMAT_BIN)
  add_custom_target(
    format
    COMMAND ${CLANG_FORMAT_BIN}
      -i
      -style=file
      ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format on source files"
  )
else()
  message(WARNING "clang-format not found. 'format' target will not be available.")
endif()
