import matplotlib.pyplot as plt
import numpy as np
import dill
import arviz as az
import MCMCutils as ut

dill.load_session("smileDistr.pkl")



print(1000*az.ess(sE[0,:])/np.sum(dE.gradEvals))
print(1000*az.ess(sE[0,:]**2)/np.sum(dE.gradEvals))

print(1000*az.ess(sF[0,:])/np.sum(dF.gradEvals))
print(1000*az.ess(sF[0,:]**2)/np.sum(dF.gradEvals))

print(1000*az.ess(sHE[0,:])/np.sum(dHE.gradEvals))
print(1000*az.ess(sHE[0,:]**2)/np.sum(dHE.gradEvals))

#print(1000*az.ess(sHF[0,:])/np.sum(dHF.gradEvals))
#print(1000*az.ess(sHF[0,:]**2)/np.sum(dHF.gradEvals))




plt.subplot(2,2,1)
ut.qqnormal(sE[0,:],plot=True)
plt.subplot(2,2,2)
ut.qqnormal(sF[0,:],plot=True)
plt.subplot(2,2,3)
plt.plot(ominE[0,:],dE.maxIf,'.')
plt.subplot(2,2,4)
plt.plot(ominF[0,:],dF.maxIf,'.')