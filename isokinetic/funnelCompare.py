import numpy as np
import math
import arviz as az
import copy
import matplotlib.pyplot as plt
import hamiltonian as hmc
import relativistic as rel
import isokinetic as iso
import correlatedMomentum as cor
import targets as td
import pandas as pd
import MCMCutils as ut
import dualAverage as da
import P2quantile as P2
import NUTSsampler as nuts


np.random.seed(0)

q0 = np.random.normal(size=11)
lp = td.funnel10 # td.stdGauss # 


niter = 2000
wISO = nuts.NUTSampler(debug=False)
wISO.run(lp,q0,step=iso.adaptIKstepE(), tp0=iso.IKtuningPars(hMacro=0.2,delta=0.3),niter=niter)


wHMC = nuts.NUTSampler(debug=False)
wHMC.run(lp,q0,step=hmc.adaptHMCstepE(), tp0=hmc.HMCtuningPars(hMacro=0.2,delta=0.3),niter=niter)

wREL = nuts.NUTSampler(debug=False)
wREL.run(lp,q0,step=rel.adaptRELIstepE(), tp0=rel.RELtuningPars(hMacro=0.2,delta=0.3,c=2.0),niter=niter)

wCM = nuts.NUTSampler(debug=False)
wCM.run(lp,q0,step=cor.adaptCMstepE(), tp0=cor.CMtuningPars(hMacro=0.2,delta=0.3),niter=niter)



print(np.mean(wISO.diagnostics.propBasic[1000:niter]))
print(np.mean(wHMC.diagnostics.propBasic[1000:niter]))
print(np.mean(wREL.diagnostics.propBasic[1000:niter]))
print(np.mean(wCM.diagnostics.propBasic[1000:niter]))
print('--')
print(np.mean(wISO.diagnostics.lwtsRange[1000:niter]))
print(np.mean(wHMC.diagnostics.lwtsRange[1000:niter]))
print(np.mean(wREL.diagnostics.lwtsRange[1000:niter]))
print(np.mean(wCM.diagnostics.lwtsRange[1000:niter]))
print('--')
print(np.quantile(wISO.diagnostics.lwtsRange[1000:niter],0.9))
print(np.quantile(wHMC.diagnostics.lwtsRange[1000:niter],0.9))
print(np.quantile(wREL.diagnostics.lwtsRange[1000:niter],0.9))
print(np.quantile(wCM.diagnostics.lwtsRange[1000:niter],0.9))
print('--')
print(1000*az.ess(wISO.samples[0,1000:niter])/np.sum(wISO.diagnostics.gradEvals[1000:niter]))
print(1000*az.ess(wHMC.samples[0,1000:niter])/np.sum(wHMC.diagnostics.gradEvals[1000:niter]))
print(1000*az.ess(wREL.samples[0,1000:niter])/np.sum(wREL.diagnostics.gradEvals[1000:niter]))
print(1000*az.ess(wCM.samples[0,1000:niter])/np.sum(wCM.diagnostics.gradEvals[1000:niter]))

plt.subplot(2,3,1)
plt.plot(wISO.samples[0,:])
plt.subplot(2,3,2)
plt.plot(wHMC.samples[0,:])
plt.subplot(2,3,3)
plt.plot(wREL.samples[0,:])
plt.subplot(2,3,4)
plt.plot(wISO.diagnostics.lwtsRange[1000:niter])
plt.subplot(2,3,5)
plt.plot(wHMC.diagnostics.lwtsRange[1000:niter])
plt.subplot(2,3,6)
plt.plot(wREL.diagnostics.lwtsRange[1000:niter])

