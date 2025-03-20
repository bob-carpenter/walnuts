#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 10:09:28 2025

@author: torekleppe
"""

import numpy as np
import WALNUTS as wn
import adaptiveIntegrators as ai
import targetDistr as td
import matplotlib.pyplot as plt

np.random.seed(1)

initialTheta = np.array([1.0,0.0])

def gen(q):
    return(np.array([q[0],q[1],sum(q*q)]))

samples,diagnostics = wn.WALNUTS(td.corrGauss, # target distribution (here Gaussian with unit marginal variances)
                              q0=initialTheta, # initial state
                              generated=gen, # what to store
                              integrator=ai.adaptLeapFrogR2P, # see adaptiveIntegrators for available integrators
                              numIter=11000,
                              warmupIter=1000) 



plt.subplot(221)
plt.plot(samples[0,:])                          
plt.subplot(222)
plt.plot(samples[0,:],samples[1,:],"b.",markersize=1)
plt.subplot(223)
plt.plot(diagnostics[:,15])
plt.ylabel("big step size")
plt.subplot(224)
plt.plot(diagnostics[:,18])
plt.ylabel("local energy error tolerance")

