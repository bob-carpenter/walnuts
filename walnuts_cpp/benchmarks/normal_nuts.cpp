// benchmarks/benchmark_nuts.cpp

#include <benchmark/benchmark.h>
#include <walnuts/nuts.hpp>
#include <Eigen/Dense>
#include <random>

using S       = double;
using VectorS = Eigen::Matrix<S, -1, 1>;
using MatrixS = Eigen::Matrix<S, -1, -1>;

// simple standard-normal log-density + gradient
template <typename T>
void standard_normal_logp_grad(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x,
                               S& logp,
                               Eigen::Matrix<T, Eigen::Dynamic, 1>& grad) {
  logp = -0.5 * x.dot(x);
  grad = -x;
}

static void BM_nuts(benchmark::State& state) {
  const int D         = state.range(0);   // dimensionality
  const int N         = state.range(1);   // number of draws
  const S   step_size = 0.025;
  const int max_depth = 10;

  // one-time setup
  VectorS inv_mass = VectorS::Ones(D);
  VectorS theta0(D);
  std::mt19937            gen(333456);
  std::normal_distribution<S> dist(0.0, 1.0);
  for (int i = 0; i < D; ++i)
    theta0(i) = dist(gen);

  for (auto _ : state) {
    // each iteration we remake the output matrix
    MatrixS draws(D, N);
    // run the sampler
    nuts::nuts(gen,
               standard_normal_logp_grad<S>,
               inv_mass,
               step_size,
               max_depth,
               theta0,
               draws);
    // prevent the compiler from optimizing away draws
    benchmark::ClobberMemory();
  }

  // let google-benchmark know complexity is O(N)
  state.SetComplexityN(N);
}

// register a few (D,N) combinations:
BENCHMARK(BM_nuts)
    ->Args({10, 1000})
    ->Args({10, 10000})
    ->Args({20, 1000})
    ->Args({20, 10000})
    ->Complexity();

// main entrypoint
BENCHMARK_MAIN();
