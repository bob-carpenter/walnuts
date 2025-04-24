


import sys
if(sys.path[-1].find("WALNUTSpy")==-1):
    sys.path.append("../WALNUTS_gh/walnuts/WALNUTSpy")


import bridgestan as brs
import numpy as np
import WALNUTS as wn
import adaptiveIntegrators as ai
import matplotlib.pyplot as plt

mdlInnov = brs.StanModel("sw_innov.stan","swdata.json")

d = mdlInnov.param_unc_num()
Nchains = 10

def lpFunInnov(q,hessian=False):
    try:
        f,g = mdlInnov.log_density_gradient(q)
        return(f,g)
    except:
        #print("numeric exception")
        return(np.nan,0.0*q)

def generatedInnov(q):
    return(mdlInnov.param_constrain(q,include_tp=True))

nms = mdlInnov.param_names(include_tp=True)

q0 = np.load("initq.npy")
dg = len(generatedInnov(q0))



np.random.seed(1)


samples,diagnostics = wn.WALNUTS(lpFun=lpFunInnov, q0=q0,
                                 generated=generatedInnov,
                                 integrator=ai.adaptLeapFrogD,
                                 M=14,
                                 H0 = 0.1,
                                 delta0=0.3,
                                 numIter=11000,
                                 warmupIter=0,
                                 igrAux=ai.integratorAuxPar(minC=3))


np.save("sw_walnuts_innov2_res",samples)
np.save("sw_walnuts_innov2_diag",diagnostics)  

np.random.seed(1)

samples,diagnostics = wn.WALNUTS(lpFun=lpFunInnov, q0=q0,
                                 generated=generatedInnov,
                                 integrator=ai.fixedLeapFrog,
                                 M=14,
                                 H0 = 0.002,
                                 numIter=11000,
                                 warmupIter=0)



np.save("sw_nuts_innov2_res",samples)
np.save("sw_nuts_innov2_diag",diagnostics)   


np.random.seed(1)
samples,diagnostics = wn.WALNUTS(lpFun=lpFunInnov, q0=q0,
                                 generated=generatedInnov,
                                 integrator=ai.adaptLeapFrogR2P,
                                 M=14,
                                 H0 = 0.1,
                                 delta0=0.3,
                                 numIter=11000,
                                 warmupIter=0,
                                 igrAux=ai.integratorAuxPar(minC=3))


np.save("sw_walnutsR2P_innov2_res",samples)
np.save("sw_walnutsR2P_innov2_diag",diagnostics)  

