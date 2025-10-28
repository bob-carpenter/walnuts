

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy as sp
import arviz as az

def ESS(x):
    return az.ess(x)

def qqnormal(x,loc=0.0,scale=1.0,plot=False):
    ys = np.sort(x)
    n = len(x)
    ps = np.arange(start=1.0,stop=n+1.0)/(n+1.0)
    xs = sp.stats.norm.ppf(ps,loc=loc,scale=scale)
    if(plot):
        plt.plot(xs,ys,".k",markersize=2)
        plt.plot(np.array([xs[0],xs[-1]]),np.array([xs[0],xs[-1]]),"r")
        plt.xlabel("theoretical quantiles")
        plt.ylabel("sample quantiles")
    return(xs,ys)


def igrErrStatHistogram(diagnostics,Nbins=50,plot=False):
    statistics = np.abs(diagnostics[:,23])
    centers = np.arange(Nbins)/(Nbins-1.0)
    bw = centers[1]-centers[0]
    bins = np.append(-0.5*bw + centers,1.0+0.5*bw)
    hs,_ = np.histogram(statistics,bins=bins,density=True)
    
    
    
    if(plot):
        plt.plot(centers,hs)
    
    return(centers,hs,bins)

