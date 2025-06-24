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
hmin = np.load("gaussTransientHmin.npy")
hmax = np.load("gaussTransientHmax.npy")
ds = 2**np.arange(11,16)
nrep = 50
nstep = 30

ms = 3.0
k = 0
for dds in np.array([0,2,4]):
    print(dds)
    k += 1
    
    plt.subplot(330+k)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)   
    
    chix = np.array([-0.0,nstep])
    chilev = np.ones(2)
    plt.plot(chix,chi2.ppf(0.995,df=ds[dds])*chilev,"grey",label="_nolegend_")
    plt.plot(chix,chi2.ppf(0.005,df=ds[dds])*chilev,"grey",label="_nolegend_")
    
    for i in range(0,nrep):
        plt.plot(np.arange(start=0,stop=nstep+1)+np.random.uniform(low=-0.3,high=0.3,size=nstep+1),res[i,dds,1,0:(nstep+1)],'.',color="red",markersize=ms)
        plt.plot(np.arange(start=0,stop=nstep+1)+np.random.uniform(low=-0.3,high=0.3,size=nstep+1),res[i,dds,2,0:(nstep+1)],'.',color="black",markersize=ms)
        
    
    
    
    plt.xlabel("MCMC iteration #",fontsize=14)
    if(k==1): plt.ylabel(" $\\theta^T \\theta $",fontsize=16)
    plt.title("standard Gaussian, d = "+str(ds[dds]),fontsize=16)
    plt.legend(["WALNUTS-R2P","NUTS"],markerscale=5,fontsize=12)
    
    
    plt.subplot(333+k)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)   
    
    
    plt.plot(chix,chilev*ds[dds]**(-1.0/4.0),color="grey")
    plt.plot(chix,chilev*ds[dds]**(-1.0/2.0),color="black")
    
    for i in range(0,nrep):
        plt.plot(np.arange(start=1,stop=nstep+1),hmin[i,dds,1,0:(nstep)]*(1+np.random.uniform(low=-0.1,high=0.1,size=nstep)),'.',color="red",markersize=ms)
    
    
    if(k==1):  plt.ylabel("smallest $h \\tilde{\ell}^{-1}$ (jittered)",fontsize=16) 
    plt.xlabel("MCMC iteration #",fontsize=14)
    plt.legend([" $h=d^{-1/4}$","$d^{-1/2}$","smallest $h \\tilde{\ell}^{-1}$"],markerscale=5,fontsize=12)
    
    
    plt.subplot(336+k)
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)   
    
    
    for i in range(0,nrep):
        
        
        plt.plot(np.arange(start=1,stop=nstep+1),evals[i,dds,1,1:(nstep+1)],'.',color="red",markersize=ms)
        plt.plot(np.arange(start=1,stop=nstep+1),evals[i,dds,2,1:(nstep+1)],'.',color="black",markersize=ms)
        
    
    
    
    plt.xlabel("MCMC iteration #",fontsize=14)
    if(k==1):  plt.ylabel("# gradient evals",fontsize=16) 
    plt.legend(["WALNUTS-R2P","NUTS"],markerscale=5,fontsize=12)
    
    
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14  
fig_size[1] = 12 
plt.rcParams["figure.figsize"] = fig_size
plt.savefig("gaussTransient.pdf",)
