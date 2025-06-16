

# NOTE : assumes mainFunnelOrbitLen.py has run first

import sys
if(sys.path[-1].find("WALNUTSpy")==-1):
    sys.path.append("../WALNUTS_gh/walnuts/WALNUTSpy")

import numpy as np
import MCMCutils as mu
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.neighbors import KNeighborsRegressor


diags = np.load("funnelOrbitLenDiags.npy")
omaxs = np.load("funnelOrbitLenOmaxs.npy")
omins = np.load("funnelOrbitLenOmins.npy")

Hs = np.array([0.15,0.3,0.6])


cols = ["black","green","red"]

nknn = 200
hnknn = np.round(nknn/2).astype(np.int64)
neigh = KNeighborsRegressor(n_neighbors=nknn)

plt.subplot(131)
for i in range(len(Hs)):
    
    neigh.fit(omaxs[:,i].reshape(-1,1),diags[:,3,i])
    maxgrid= np.arange(start=np.sort(omaxs[:,i])[hnknn],stop=np.sort(omaxs[:,i])[-hnknn],step=0.05)
    
    plt.semilogy(maxgrid, neigh.predict(maxgrid.reshape(-1,1)),color=cols[i])   
    plt.semilogy(np.array([np.min(omaxs[:,:]),np.max(omaxs[:,:])]),
                 Hs[i]*np.ones(2),":",color=cols[i],label="_nolegend_")
    

plt.legend(["$h=0.15$","$h=0.3$","$h=0.6$"])
plt.xlabel("largest $\omega$ in orbit")
plt.ylabel("(accepted) orbit length")





plt.subplot(132)

for i in range(len(Hs)):
    
    mingrid= np.arange(start=np.sort(omins[:,i])[hnknn],stop=np.sort(omins[:,i])[-hnknn],step=0.05)
    neigh.fit(omins[:,i].reshape(-1,1),Hs[i]*(2**(-diags[:,9,i])))
    
    plt.semilogy(mingrid, neigh.predict(mingrid.reshape(-1,1)),color=cols[i])   
    neigh.fit(omins[:,i].reshape(-1,1),Hs[i]*(2**(-diags[:,8,i])))
    
    plt.semilogy(mingrid, neigh.predict(mingrid.reshape(-1,1)),"--",color=cols[i])   
    plt.semilogy(np.array([np.min(omins[:,:]),np.max(omins[:,:])]),
                 Hs[i]*np.ones(2),":",color=cols[i],label="_nolegend_")
    

plt.legend(["$h=0.15$, smallest","$h=0.15$, largest","$h=0.3$ smallest","$h=0.3$ largest","$h=0.6$, smallest", "$h=0.6$, largest"])
plt.xlabel("smallest $\omega$ in orbit")
plt.ylabel("micro step size $h \ell^{-1}$ required in orbit")



plt.subplot(133)

for i in range(len(Hs)):
    cen,hh,bins = mu.igrErrStatHistogram(diags[:,:,i],40)
    plt.plot(cen,hh,color=cols[i])
    if(i==0):
        for b in bins:
            plt.plot(b*np.ones(2),np.array([-1,10]),'0.6',":",linewidth=0.3,label="_nolegend_")
    
    
plt.axis([-0.05,1.05,0,10])
plt.xlabel("standardized distance from current state")
plt.ylabel("density-scaled histogram of orbits")
plt.legend(["$h=0.15$","$h=0.3$","$h=0.6$"])



fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14  
fig_size[1] = 7 
plt.rcParams["figure.figsize"] = fig_size
plt.savefig("funnelMultipleStepsizes.pdf")

