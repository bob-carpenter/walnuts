
import numpy as np
import targets as td
import matplotlib.pyplot as plt
import pandas as pd
import microCanonical as mc
import hamiltonian as hmc
import samplers as sam
import dill



lp = td.corrGauss # td.smileDistr # td.funnel1 # td.modFunnel # td.stdGauss # 

q0 = np.array([0.0,-0.01])

MCtp = mc.MCtuningPars(delta=0.3,Cmax=10)

stepE = mc.adaptMCstepE()
stepF = mc.adaptMCstepFlow2()


HMCtp = hmc.HMCtuningPars(delta=0.3,Cmax=10)
stepHE = hmc.adaptHMCstepE()
stepHF = hmc.adaptHMCstepF()

nwarmup = 1000
nsamples = 1000
L = 1024

scale = np.array([3.0,1.0]) #np.array([3.0,np.sqrt(np.exp(4.5))])

np.random.seed(1)

(sE,dE,tpE) = sam.multinomialSampler(stepE, lp, q0, MCtp,L=L,niter=nwarmup,nwarmup=nwarmup,scale=scale)
(sE,dE,_,ominE,omaxE) = sam.multinomialSampler(stepE, lp, q0, tpE,L=L,niter=nsamples,nwarmup=0,scale=scale,orbitStats_=True)


(sF,dF,tpF) = sam.multinomialSampler(stepF, lp, q0, MCtp,L=L,niter=nwarmup,nwarmup=nwarmup)
(sF,dF,_,ominF,omaxF) = sam.multinomialSampler(stepF, lp, q0, tpF,L=L,niter=nsamples,nwarmup=0,orbitStats_=True)


(sHE,dHE,tpHE) = sam.multinomialSampler(stepHE, lp, q0, HMCtp,L=L,niter=nwarmup,nwarmup=nwarmup,scale=scale)
(sHE,dHE,_,ominHE,omaxHE) = sam.multinomialSampler(stepHE, lp, q0, tpHE,L=L,niter=nsamples,nwarmup=0,scale=scale,orbitStats_=True)


#(sHF,dHF,tpHF) = sam.multinomialSampler(stepHF, lp, q0, HMCtp,L=L,niter=nwarmup,nwarmup=nwarmup)
#(sHF,dHF,_,ominHF,omaxHF) = sam.multinomialSampler(stepHF, lp, q0, tpHF,L=L,niter=nsamples,nwarmup=0,orbitStats_=True)


dill.dump_session("smileDistr.pkl")

