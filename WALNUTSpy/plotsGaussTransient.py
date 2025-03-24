#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 08:19:21 2025

@author: torekleppe
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import chi2

res = np.load("gaussTransientRes.npy")
evals = np.load("gaussTransientEvals.npy")

ds = 2**np.arange(11,16)
nrep = 50
nstep = 50

ms = 2.0
k = 0
for dds in np.array([0,2,4]):
    print(dds)
    k += 1
    
    plt.subplot(230+k)
    
    for i in range(0,nrep):
        plt.plot(res[i,dds,1,:],'r.',markersize=ms)
        plt.plot(res[i,dds,2,:],'g.',markersize=ms)
        
    
    chix = np.array([-0.0,50.0])
    chilev = np.ones(2)
    plt.plot(chix,chi2.ppf(0.995,df=ds[dds])*chilev,'0.8')
    plt.plot(chix,chi2.ppf(0.005,df=ds[dds])*chilev,'0.8')
    
    plt.xlabel("iteration #")
    if(k==1): plt.ylabel(" $\\theta^T \\theta $")
    plt.title("standard Gaussian, d = "+str(ds[dds]))
    plt.legend(["WALNUTS","NUTS"],markerscale=10)
    
    
    plt.subplot(233+k)
    
    
    
    for i in range(0,nrep):
        
        
        plt.semilogx(evals[i,dds,1,:],res[i,dds,1,:],'r.',markersize=ms)
        plt.semilogx(evals[i,dds,2,:],res[i,dds,2,:],'g.',markersize=ms)
        
    
    chix = np.array([min(np.matrix.flatten(evals[:,dds,:,:])),max(np.matrix.flatten(evals[:,dds,:,:]))])
    chilev = np.ones(2)
    plt.semilogx(chix,chi2.ppf(0.995,df=ds[dds])*chilev,'0.8')
    plt.semilogx(chix,chi2.ppf(0.005,df=ds[dds])*chilev,'0.8')
    
    cols = ["b","r","g","k"]
    
    for i in range(1,3):
        smoothed = sm.nonparametric.lowess(exog=np.matrix.flatten(evals[:,dds,i,1:nstep]), 
                                       endog=np.matrix.flatten(res[:,dds,i,1:nstep]), frac=0.05)
        plt.semilogx(smoothed[:,0], smoothed[:,1],cols[i],linewidth=5)
        plt.semilogx(smoothed[:,0], smoothed[:,1],"w",linewidth=1)
    
    plt.xlabel("# gradient evals")
    if(k==1): plt.ylabel("$\\theta^T \\theta$")
    plt.legend(["WALNUTS","NUTS"],markerscale=10)
    
    
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14  
fig_size[1] = 7 
plt.rcParams["figure.figsize"] = fig_size
plt.savefig("gaussTransient.pdf",)
