# WALNUTS in C++

First, we are building and integrating NUTS.  Second, we will extend to WALNUTS.

## Running NUTS sampler

First, you need to download the Eigen 3.4.0 source and place it in a directory `walnuts_cpp/lib/eigen-3.4.0`.

### Compiling and running

From the `walnuts_cpp` folder call the following for the benchmarks and testing.

```bash
# Call cmake with our source as our current directory and build in a new folder "build"
cmake -S . -B "build" -DCMAKE_BUILD_TYPE=RELEASE -DWALNUTS_BUILD_TESTS=ON -DWALNUTS_BUILD_BENCHMARKS=ON
cd build
# List possible targets
make help
make -j3 normal_nuts
# After building
./benchmarks/normal_nuts
make -j3 test_nuts
./tests/test_nuts
```

#### CMake Tips

##### View Optional Project Flags

To view the optional flags for cmake with this project call `cmake -S . -B "build" -LH` and grep for cmake variables that start with walnuts.

```bash
cmake -S . -B "build" -LH | grep "WALNUTS" -B1
WALNUTS_BUILD_BENCHMARKS:BOOL=ON
WALNUTS_BUILD_DOXYGEN:BOOL=OFF
WALNUTS_BUILD_TESTS:BOOL=ON
```

##### Include Variables When Compiling

To set variables when compiling cmake we use `-DVARIABLE_NAME=VALUE` like setting a macro.

##### Refresh CMake

Cmake stores a `CMakeCache.txt` file with the variables from your most recent build.
For an existing build you want to completely refresh use `--fresh` when building.

```bash
cmake -S . -B "build" --fresh
```

##### View Project Targets

To see the available targets from the top level directory run the following after building

```bash
cmake --build build --target help
```

We can also use this to build from the top level directory

```bash
cmake -S . -B "build" -DCMAKE_BUILD_TYPE=RELEASE -DWALNUTS_BUILD_TESTS=ON -DWALNUTS_BUILD_BENCHMARKS=ON
cmake --build build --parallel 3 --target normal_nuts
./build/benchmarks/normal_nuts
```

One benefit of `cmake --build` is that it abstracts away the underlying build system.
For instance, if we have `ninja` installed locally we can use it with `--build` to make the target.

```bash
cmake -S . -B "build" -G Ninja -DCMAKE_BUILD_TYPE=RELEASE -DWALNUTS_BUILD_TESTS=ON -DWALNUTS_BUILD_BENCHMARKS=ON
cmake --build build --parallel 3 --target normal_nuts
```
