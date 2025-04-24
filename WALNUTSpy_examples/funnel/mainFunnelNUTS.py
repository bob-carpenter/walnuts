#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:56:00 2025

@author: torekleppe
"""

import sys
if(sys.path[-1].find("WALNUTSpy")==-1):
    sys.path.append("../WALNUTS_gh/walnuts/WALNUTSpy")

import WALNUTS as wn
import numpy as np
import targetDistr as td
import adaptiveIntegrators as ai
np.random.seed(seed=1)

q0 = np.zeros(11)
q0[0] = 3.0*np.random.normal(1)
for i in range(0,10): q0[i+1] = np.exp(0.5*q0[0])*np.random.normal(1)

def gen(q):
    return(np.array([q[0],q[1]]))

nIter = 1000000


# calculate step size leading to roughly equivalent work
diagW = np.load("funnelDiagnostics_FN.npy")
ol = diagW[:,2]
nstep = diagW[:,6] + diagW[:,7]
HNUTS = np.mean(ol/nstep)



samples,diagnostics = wn.WALNUTS(td.funnel10, 
                              q0,
                              generated=gen,
                              integrator=ai.fixedLeapFrog, 
                              M=12, 
                              H0=HNUTS,delta0=1.0,numIter=nIter,
                              warmupIter=0
                              )


np.save("funnelNUTSSamples_FN", samples)
np.save("funnelNUTSDiagnostics_FN", diagnostics)




