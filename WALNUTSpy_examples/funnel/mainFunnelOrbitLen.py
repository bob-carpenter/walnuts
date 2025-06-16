


import sys
if(sys.path[-1].find("WALNUTSpy")==-1):
    sys.path.append("../WALNUTS_gh/walnuts/WALNUTSpy")

import WALNUTS as wn
import numpy as np
import targetDistr as td
import adaptiveIntegrators as ai
np.random.seed(seed=0)

Hs = np.array([0.15,0.3,0.6])


q0 = np.zeros(11)
q0[0] = 3.0*np.random.normal(1)
for i in range(0,10): q0[i+1] = np.exp(0.5*q0[0])*np.random.normal(1)



def gen(q):
    return(np.array([q[0],q[1]]))

nIter = 50000

diags = np.zeros((nIter,24,len(Hs)))
omins = np.zeros((nIter,len(Hs)))
omaxs = np.zeros((nIter,len(Hs)))

for i in range(len(Hs)):

    samples,diagnostics,omin,omax = wn.WALNUTS(td.funnel10, 
                              q0,
                              generated=gen,
                              integrator=ai.adaptLeapFrogR2P, 
                              M=12,H0=Hs[i],delta0=0.1,numIter=nIter,
                              warmupIter=0,
                              recordOrbitStats=True
                              )
    
    for j in range(24): diags[:,j,i] = diagnostics[:,j]
    omins[:,i] = omin[0,:]
    omaxs[:,i] = omax[0,:]
    
    

np.save("funnelOrbitLenDiags",diags)
np.save("funnelOrbitLenOmaxs",omaxs)
np.save("funnelOrbitLenOmins",omins)





