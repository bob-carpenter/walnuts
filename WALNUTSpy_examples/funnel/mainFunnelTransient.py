
import sys
if(sys.path[-1].find("WALNUTSpy")==-1):
    sys.path.append("../WALNUTS_gh/walnuts/WALNUTSpy")

import WALNUTS as wn
import numpy as np
import targetDistr as td
import adaptiveIntegrators as ai
import matplotlib.pyplot as plt

np.random.seed(seed=1)

q0 = np.zeros(11)
q0[0] = -30.0




def gen(q):
    print(q[0])
    return(np.array([q[0],q[1]]))

if(True):  # turn of for plotting only
    nIter = 1000
    samples,diagnostics,minorb,maxorb = wn.WALNUTS(td.funnel10, 
                              q0,
                              generated=gen,
                              integrator=ai.adaptLeapFrogR2P, 
                              M=12,H0=0.3,delta0=0.3,numIter=nIter,
                              warmupIter=0,
                              recordOrbitStats=True,
                              igrAux=ai.integratorAuxPar(minC=0,maxC=30)
                              )


    np.save("funnelTransient_samples",samples)
    np.save("funnelTransient_diagnostics",diagnostics)
    np.save("funnelTransient_minorb",minorb)
    np.save("funnelTransient_maxorb",maxorb)

samples = np.load("funnelTransient_samples.npy")
diagnostics = np.load("funnelTransient_diagnostics.npy")
minorb = np.load("funnelTransient_minorb.npy")
maxorb = np.load("funnelTransient_maxorb.npy")


plt.subplot(1,3,1)
plt.plot(np.arange(start=1,stop=301),maxorb[0,0:300],label="largest $\omega$ in orbit")
plt.plot(np.arange(start=1,stop=301),minorb[0,0:300],label="smallest $\omega$ in orbit")
plt.plot(np.arange(start=0,stop=301),samples[0,0:301],".",markersize=5,label="$\omega$ MCMC samples")

plt.legend()
plt.xlabel("MCMC iteration #")
plt.ylabel("$\omega$")


plt.subplot(1,3,2)
plt.semilogy(np.array([0,300]),np.array([0.3,0.3]),'0.8')
for i in range(300):
    plt.semilogy((i+1)*np.ones(2),
                 0.3*np.array([2.0**(-diagnostics[i,8]),
                               2.0**(-diagnostics[i,9])]),"b.-",markersize=1)
    plt.semilogy()
plt.xlabel("MCMC iteration #")
plt.ylabel("range of micro step sizes in orbit")


plt.subplot(1,3,3)
plt.semilogy(np.array([0,300]),np.array([0.3,0.3]),'0.8')
plt.semilogy(diagnostics[0:300,17])
plt.ylabel("orbit energy error")
plt.xlabel("MCMC iteration #")

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14
fig_size[1] = 7
plt.rcParams["figure.figsize"] = fig_size
plt.savefig("funnelTransient.pdf",)


