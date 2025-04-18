# WALNUTS

WALNUTS is a Markov chain Monte Carlo (MCMC) sampler that locally
adapts step size for the No-U-Turn Sampler (NUTS).

WALNUTS is a near-acronym for "within-oribt adaptive step-length
no-U-turn sampler."
 
## `walnuts` package

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

## Original `WALNUTSpy`

Tore Kleppe's original implementation is in `WALNUTSpy`.  This version 
includes the implicit midpoint integrator as well as the leapfrog
integrator. 

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

## References

1. Nawaf Bou-Rabee, Bob Carpenter, Tore Selland Kleppe, and Sifan
Liu. 2025.  The Within-Orbit Adaptive Step-Length No-U-Turn Sampler.
