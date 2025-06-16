

import sys
if(sys.path[-1].find("WALNUTSpy")==-1):
    sys.path.append("../WALNUTS_gh/walnuts/WALNUTSpy")

import WALNUTS as wn
import adaptiveIntegrators as ai
import numpy as np
import targetDistr as td
import matplotlib.pyplot as plt

np.random.seed(seed=1)
ds = 2**np.arange(11,16)
nrep = 50
nstep = 100

res = np.zeros((nrep,len(ds),3,nstep+1))
evals = np.zeros((nrep,len(ds),3,nstep+1))
evalsF = np.zeros((nrep,len(ds),3,nstep+1))

def gen(q):
    return(np.array([sum(q*q)]))

k = -1
for dd in ds:
    print(dd)
    k += 1
    q0 = np.zeros(dd)
    
    H = (dd**(-1.0/4.0))
    Hnuts = (dd**(-1.0/2.0))
    delta = 0.3
    
    for rep in range(0,nrep):
        samples,diagnostics = wn.WALNUTS(td.stdGauss, 
                                      q0,
                                      generated=gen,
                                      integrator=ai.adaptLeapFrogD, 
                                      M=10,H0=H,delta0=delta,numIter=nstep,
                                      warmupIter=0,
                                      igrAux=ai.integratorAuxPar(),
                                      )
        res[rep,k,0,:] = samples[0,:]
        evals[rep,k,0,:] = np.r_[np.zeros(1),np.cumsum(diagnostics[:,6]+diagnostics[:,7])]
        evalsF[rep,k,0,:] = np.r_[np.zeros(1),np.cumsum(diagnostics[:,6])]
    
        samples,diagnostics = wn.WALNUTS(td.stdGauss, 
                                      q0,
                                      generated=gen,
                                      integrator=ai.adaptLeapFrogR2P, 
                                      M=10,H0=H,delta0=delta,numIter=nstep,
                                      warmupIter=0,
                                      igrAux=ai.integratorAuxPar(),
                                      )
        res[rep,k,1,:] = samples[0,:]
        evals[rep,k,1,:] = np.r_[np.zeros(1),np.cumsum(diagnostics[:,6]+diagnostics[:,7])]
        evalsF[rep,k,1,:] = np.r_[np.zeros(1),np.cumsum(diagnostics[:,6])]
        
        samples,diagnostics = wn.WALNUTS(td.stdGauss, 
                                      q0,
                                      generated=gen,
                                      integrator=ai.fixedLeapFrog, 
                                      M=10,H0=Hnuts,
                                      delta0=delta,numIter=nstep,
                                      warmupIter=0,
                                      igrAux=ai.integratorAuxPar(),
                                      )
        res[rep,k,2,:] = samples[0,:]
        evals[rep,k,2,:] = np.r_[np.zeros(1),np.cumsum(diagnostics[:,6])]
        evalsF[rep,k,2,:] = np.r_[np.zeros(1),np.cumsum(diagnostics[:,6])]

        

    np.save("gaussTransientRes",res)
    np.save("gaussTransientEvals",evals)
    np.save("gaussTransientEvalsForw",evalsF)
