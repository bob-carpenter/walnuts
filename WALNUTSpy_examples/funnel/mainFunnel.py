

import sys
if(sys.path[-1].find("WALNUTSpy")==-1):
    sys.path.append("../WALNUTS_gh/walnuts/WALNUTSpy")

import WALNUTS as wn
import numpy as np
import targetDistr as td
import adaptiveIntegrators as ai
np.random.seed(seed=0)

q0 = np.zeros(11)
q0[0] = 3.0*np.random.normal(1)
for i in range(0,10): q0[i+1] = np.exp(0.5*q0[0])*np.random.normal(1)



def gen(q):
    return(np.array([q[0],q[1]]))

nIter = 1000000

samples,diagnostics = wn.WALNUTS(td.funnel10, 
                              q0,
                              generated=gen,
                              integrator=ai.adaptLeapFrogR2P, 
                              M=12,H0=0.3,delta0=0.3,numIter=1000,
                              warmupIter=1000,
                              adaptDeltaTarget=0.6,
                              recordOrbitStats=False
                              )

samples,diagnostics,omin,omax = wn.WALNUTS(td.funnel10, 
                              q0,
                              generated=gen,
                              integrator=ai.adaptLeapFrogR2P, 
                              M=12,H0=diagnostics[-1,15],delta0=diagnostics[-1,18],numIter=nIter,
                              warmupIter=0,
                              recordOrbitStats=True
                              )


np.save("funnelSamples_FN", samples)
np.save("funnelDiagnostics_FN", diagnostics)
np.save("funnelOmin_FN", omin)
np.save("funnelOmax_FN", omax)






