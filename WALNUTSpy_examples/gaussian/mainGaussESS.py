#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 13:42:49 2025

@author: torekleppe
"""

import sys
if(sys.path[-1].find("WALNUTSpy")==-1):
    sys.path.append("../WALNUTS_gh/walnuts/WALNUTSpy")

import numpy as np
import WALNUTS as wn
import targetDistr as td
import adaptiveIntegrators as ai
import arviz as az


ds = 2**np.arange(8,19) #2**np.arange(10,19)

nrep = 10
nIter = 1000
nMethod = 3
nStats = 4

results = np.zeros((len(ds),nrep,nMethod,nStats))

def gen(q):
    return(np.array([q[0],sum(q*q)]))
k = -1
for d in ds:
    k += 1
    
    H = 1.4*(d**(-1.0/4.0))
    delta = 0.3
    
    for rep in range(0,nrep):
        print("d = ",d," replica : ",rep)    
        q0 = np.random.normal(size=d)
        
        # WALNUTS R2P
        
        samples,diagnostics = wn.WALNUTS(td.stdGauss, 
                                      q0,
                                      generated=gen,
                                      integrator=ai.adaptLeapFrogR2P, 
                                      M=10,H0=H,delta0=delta,numIter=1000,warmupIter=0
                                      )
        neval = sum(diagnostics[:,6])+sum(diagnostics[:,7])
        results[k,rep,0,0] = az.ess(samples[0,:])/neval
        results[k,rep,0,1] = az.ess(samples[1,:])/neval
        neval = sum(diagnostics[:,6])
        results[k,rep,0,2] = az.ess(samples[0,:])/neval
        results[k,rep,0,3] = az.ess(samples[1,:])/neval
        
        # WALNUTS D
        
        samples,diagnostics = wn.WALNUTS(td.stdGauss, 
                                      q0,
                                      generated=gen,
                                      integrator=ai.adaptLeapFrogD, 
                                      M=10,H0=H,delta0=delta,numIter=1000,warmupIter=0
                                      )
        neval = sum(diagnostics[:,6])+sum(diagnostics[:,7])
        results[k,rep,1,0] = az.ess(samples[0,:])/neval
        results[k,rep,1,1] = az.ess(samples[1,:])/neval
        neval = sum(diagnostics[:,6])
        results[k,rep,1,2] = az.ess(samples[0,:])/neval
        results[k,rep,1,3] = az.ess(samples[1,:])/neval
        
        # NUTS
        
        samples,diagnostics = wn.WALNUTS(td.stdGauss, 
                                      q0,
                                      generated=gen,
                                      integrator=ai.fixedLeapFrog, 
                                      M=10,H0=H,
                                      numIter=1000,warmupIter=0
                                      )
        neval = sum(diagnostics[:,6])+sum(diagnostics[:,7])
        results[k,rep,2,0] = az.ess(samples[0,:])/neval
        results[k,rep,2,1] = az.ess(samples[1,:])/neval
        neval = sum(diagnostics[:,6])
        results[k,rep,2,2] = az.ess(samples[0,:])/neval
        results[k,rep,2,3] = az.ess(samples[1,:])/neval
        
        
    np.save("gaussESSRes",results)


