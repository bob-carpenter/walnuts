
Performance tests in this file can be run like the following from the build folder

```bash
make run_perf_test_nuts
make report_perf_test_nuts
make flamegraph_test_nuts
chromium-browser ${PWD}/build/perf_tests/data/perf_test_nuts.svg &
```
