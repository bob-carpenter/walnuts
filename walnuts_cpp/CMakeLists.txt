cmake_minimum_required(VERSION 3.14)
project(walnuts
  VERSION 0.1.0
  DESCRIPTION "Header-only NUTS sampler library"
  LANGUAGES CXX
)

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
option(WALNUTS_BUILD_EXAMPLES "Build the example targets for the library" ON)
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
      "-O3"
      CACHE STRING "Flags used by the C++ compiler during Release builds."
      FORCE)
endif()

##########################
## Global Dependencies  ##
##########################
include(FetchContent)

FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
)
FetchContent_Populate(Eigen)   # ⇒ eigen_SOURCE_DIR / eigen_BINARY_DIR

# 1) Create an INTERFACE library for the headers
add_library(eigen_headers INTERFACE)
target_include_directories(eigen_headers
  INTERFACE
    ${eigen_SOURCE_DIR}      # root of Eigen repo: contains Eigen/ and unsupported/
)

# 2) Alias it so downstream CMakeLists can do find-like targets
add_library(Eigen3::Eigen ALIAS eigen_headers)


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
add_library(walnuts::walnuts ALIAS walnuts)

# 3) Installation rules
install(TARGETS walnuts
  EXPORT walnutsTargets
  INCLUDES DESTINATION include
)
install(DIRECTORY include/ DESTINATION include)

# 4) Export the targets file
install(EXPORT walnutsTargets
  FILE walnutsTargets.cmake
  NAMESPACE walnuts::
  DESTINATION lib/cmake/walnuts
)

# 5) Generate a simple package config that just pulls in the exported targets
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/walnutsConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/walnutsConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/walnutsConfig.cmake"
  INSTALL_DESTINATION "lib/cmake/walnuts"
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/walnutsConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/walnutsConfigVersion.cmake"
  DESTINATION lib/cmake/walnuts
)

##########################
##       Example        ##
##########################
if (WALNUTS_BUILD_EXAMPLES)
  add_executable(test_nuts ${CMAKE_CURRENT_SOURCE_DIR}/examples/test.cpp)
  target_link_libraries(test_nuts PRIVATE Eigen3::Eigen walnuts::walnuts)
endif()


##########################
##       Extras         ##
##########################
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/extras/CMakeLists.txt")
  add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/extras")
endif()
