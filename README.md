# WALNUTS

WALNUTS is a Markov chain Monte Carlo (MCMC) sampler that locally
adapts step size for the No-U-Turn Sampler (NUTS).  WALNUTS is a
near-acronym for "within-oribt adaptive step-length no-U-turn
sampler."

This repository contains 3 projects, which are outlined below:

1. `WALNUTSpy`: Flexible and highly instrumented sampler coded in
   Python; used for all of the experiments in the paper.

2. `walnuts` package: Python implementation directly following the
   (inefficient) pseudocode.
   
3. `walnuts_cpp`: Performant C++ implementation.


## 1. Original `WALNUTSpy`

The original implementation is in `WALNUTSpy`.  This version 
includes the implicit midpoint integrator as well as the leapfrog
integrator and was used for all of the experiments in the paper. 

```
walnuts/
├── WALNUTSpy/
│   ├── adaptiveIntegrators.py 
│   ├── constants.py 
│   ├── example.py 
│   ├── mainGaussTransient.py 
│   ├── plotGaussTransient.py 
│   ├── targetDistr.py 
│   ├── MCMCutils.py 
│   ├── P2quantile.py
│   ├── WALNUTS.py 
```

`example.py` provides an example of the usage of this sampler. The file `adaptiveIntegrators.py` contain several different adaptive integrators, and also `fixedLeapFrog()` which turns the sampler into a regular NUTS sampler.


## 2. `walnuts` reference Python implementation

The implementation in the `walnuts` package is a reference
implementation that follows the pseudocode in the paper.

```
walnuts/
├── pyproject.toml
├── walnuts/
│   ├── __init__.py
│   ├── walnuts.py 
│   ├── walnuts_stan.py 
├── test/
│   ├── targets.py 
│   ├── test.py 
```

### Running the tests

To run the tests, execute the `test.py` script.  

```
$ cd walnuts
$ python3 test/test.py
```

The tests appear as a sequence of calls at the bottom of `test.py` and
some of them may be commented out for efficiency.


## 3. `walnuts_cpp`: Performant C++ implementation

See the [README for `walnuts_cpp`](walnuts_cpp/README.md)


## References

This is the paper on which the sampler is based.

* Nawaf Bou-Rabee, Bob Carpenter, Tore Selland Kleppe, and Sifan
Liu. (to appear). WALNUTS: The Within-Orbit Adaptive Step-Length
No-U-Turn Sampler.

