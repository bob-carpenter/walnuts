# WALNUTS in C++

First, we are building and integrating NUTS.  Second, we will extend to WALNUTS.

## Compiling and running

From the `walnuts_cpp` folder call the following for the benchmarks and testing.

```bash
# /usr/bin/bash
# Assumes you start in walnuts_cpp directory

# Run cmake with
# The source directory as our current directory (-S .)
# and build in a new folder "build" (-B "build")
# Note that we use cmake's fetch_content and ExternalProject_Add
#  which manage our dependencies for us (just Eigen by default)
#  This means that creating the project from source
#  does require an internet connection.
cmake -S . -B "build" -DCMAKE_BUILD_TYPE=RELEASE
# After this call all of our build dependencies
# and make targets now exist in build
ls ./build
cd build
# List possible targets
make help
# Build and run the test_nuts example in ./example/
# Note: cmake will pull Eigen 3.4 down from gitlab
#  It will only do this the first time you run
#  a make command that depends on Eigen.
make -j3 test_nuts
./examples/test_nuts
```

## Structure

```
walnuts_cpp
# For runnning google benchmark
├── benchmarks
│   ├── CMakeLists.txt
│   └── normal_nuts.cpp
# Global dependencies
├── cmake_deps
│   └── CMakeLists.txt
├── CMakeLists.txt
# Example of nuts
├── examples
│   ├── CMakeLists.txt
│   ├── test_stan.cpp
│   └── test.cpp
# Headers for algorithms
├── include
│   └── walnuts
│       └── nuts.hpp
# For automated perf tests
├── perf_tests
│   ├── CMakeLists.txt
│   └── readme.md
├── README.md
└── tests
    └── CMakeLists.txt
```

## CMake Tips

### View Optional Project Flags

To view the optional flags for cmake with this project call `cmake -S . -B "build" -LH` and grep for cmake variables that start with walnuts.

```bash
# /usr/bin/bash
# From walnuts_cpp
# Same as other command but -LH lists all cached cmake variables
# along with their help comment
cmake -S . -B "build" -LH | grep "WALNUTS" -B1
# Output
$ WALNUTS_BUILD_BENCHMARKS:BOOL=ON
$ WALNUTS_BUILD_DOXYGEN:BOOL=OFF
$ WALNUTS_BUILD_TESTS:BOOL=ON
```

### Include Variables When Compiling

To set variables when compiling cmake we use `-DVARIABLE_NAME=VALUE` like setting a macro.

### Refresh CMake

Cmake stores a `CMakeCache.txt` file with the variables from your most recent build.
For an existing build you want to completely refresh use `--fresh` when building.

```bash
# /usr/bin/bash
# From walnuts_cpp
# Run a build but use --fresh to force 
# a hard reset of all cached variables
cmake -S . -B "build" --fresh
# All the cmake targets now exist in build
cd build
```

### View Project Targets

To see the available targets from the top level directory run the following after building

```bash
# /usr/bin/bash
# From walnuts_cpp
cmake -S . -B "build"
cmake --build build --target help
```

We can also use this to build from the top level directory

```bash
# /usr/bin/bash
# From walnuts_cpp
# This will take longer as we include depedencies
# google test and google benchmark
cmake -S . -B "build" -DCMAKE_BUILD_TYPE=RELEASE -DWALNUTS_BUILD_TESTS=ON -DWALNUTS_BUILD_BENCHMARKS=ON
# Now the build directory is setup and we we can build and run the benchmarks
cmake --build build --parallel 3 --target normal_nuts
# Now the benchmark is built in the build folder
ls ./build
# Now run the benchmark
./build/benchmarks/normal_nuts
```

One benefit of `cmake --build` is that it abstracts away the underlying build system.
For instance, if we have `ninja` installed locally we can use it with `--build` to make the target.

```bash
# /usr/bin/bash
# From walnuts_cpp
# Same as above but we use Ninja instead of make
# For the underlying generator
cmake -S . -B "build" -G Ninja -DCMAKE_BUILD_TYPE=RELEASE -DWALNUTS_BUILD_TESTS=ON -DWALNUTS_BUILD_BENCHMARKS=ON
cmake --build build --parallel 3 --target normal_nuts
```

When in the `build` directory you can call `cmake ..` to run cmake again.
This is nice for refreshing variables with `cmake .. --fresh`

### Debugging CMake

Setting `-DCMAKE_BUILD_TYPE=DEBUG` will make the make file generation verbose.
For all other build types you can add `VERBOSE=1` to your make call to see a trace of the actions CMake performs.

### Packaging For Release

Use cpack to put the includes and third party libraries in one tar. These following code will create the tarballs with dependencies for releases.

```bash
# /usr/bin/bash
# From walnuts_cpp
# Setup and direct installs to happen in our build file
cmake -S . -B "build" -DCMAKE_INSTALL_PREFIX=$PWD/build/install -DWALNUTS_BUILD_TARBALL=ON
# Run the install locally
cmake --build build --target install
# Package everything into a tar.gz file
cmake --build build --target package
```

## Formatting

```bash
cd walnuts_cpp
clang-format -i -style=LLVM nuts.hpp test.cpp
```

