# This needs to be isolated because of bugs in Eigen's cmake files
# If we use FetchContent, Eigen includes all of its tests as targets
# this makes make help very useless for our project.
include(ExternalProject)

# where to put Eigen’s download/build dirs
set(EIGEN_PREFIX ${CMAKE_BINARY_DIR}/_deps/eigen)
set(EIGEN_SOURCE_DIR ${EIGEN_PREFIX}/src/Eigen)
set(EIGEN_BINARY_DIR ${EIGEN_PREFIX}/build)

ExternalProject_Add(Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  # isolate all the build trees off your main CMake cache
  PREFIX          ${EIGEN_PREFIX}
  SOURCE_DIR      ${EIGEN_SOURCE_DIR}
  BINARY_DIR      ${EIGEN_BINARY_DIR}
  # disable every test/example/doc target
  CMAKE_ARGS
    -DBUILD_TESTING=OFF
    -DEIGEN_BUILD_DOC=OFF
    -DEIGEN_BUILD_PKGCONFIG=OFF
    -DEIGEN_BUILD_BTL=OFF
    -DEIGEN_LEAVE_TEST_IN_ALL_TARGET=OFF
    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>  # unused for header‐only
  # we don’t need to actually "install" a header‐only lib:
  INSTALL_COMMAND  ""
)

# now create an IMPORTED interface‐only target that points at the Eigen headers
add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
add_dependencies(Eigen3::Eigen Eigen)  # make sure headers are fetched before use

# Eigen headers live in the repo root (they install into include/Eigen, but
# repo layout has Eigen/ in the root)
target_include_directories(Eigen3::Eigen
  INTERFACE
    ${EIGEN_SOURCE_DIR}
)

# optional: propagate that this is header‐only
set_target_properties(Eigen3::Eigen PROPERTIES
  INTERFACE_HEADER_ONLY TRUE
)
