#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:28:36 2025

@author: torekleppe
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

ds = 2**np.arange(8,19)
results = np.load("gaussESSRes.npy")

stat = 1
maxd = 0

def CI10(dta):
    q = t.ppf(0.975,df=9)/np.sqrt(9.0)
    m = np.mean(dta)
    s = np.sqrt(np.var(dta))
    ci = np.array([m-s*q,m+s*q])
    return(ci)

for i in range(len(ds)):
    if(np.abs(results[i,0,0,0])>0.0):
        maxd = i


m1=np.zeros(len(ds))
m2=np.zeros(len(ds))
m3=np.zeros(len(ds))
maxind = 0

for i in range(len(ds)):
    if(np.abs(results[i,0,0,0])>0.0):
        maxind = i
        # R2P
        dta = results[i,:,0,stat]*1000
        ci = CI10(dta)
        plt.loglog(ds[i]*np.ones(2),ci,"r",label="_nolegend_")
        plt.loglog(ds[i]*np.array([0.99,1.01]),ci[0]*np.ones(2),"r",label="_nolegend_")
        plt.loglog(ds[i]*np.array([0.99,1.01]),ci[1]*np.ones(2),"r",label="_nolegend_")
        plt.loglog(ds[i],np.mean(dta),"or",markersize=10)
        m1[i] = np.mean(dta)
        
        # D
        dta = results[i,:,1,stat]*1000
        ci = CI10(dta)
        plt.loglog(ds[i]*np.ones(2),ci,"grey",label="_nolegend_")
        plt.loglog(ds[i]*np.array([0.99,1.01]),ci[0]*np.ones(2),"grey",label="_nolegend_")
        plt.loglog(ds[i]*np.array([0.99,1.01]),ci[1]*np.ones(2),"grey",label="_nolegend_")
        plt.loglog(ds[i],np.mean(dta),"d",color="grey",markersize=10)
        m2[i] = np.mean(dta)
        
        # NUTS
        dta = results[i,:,2,stat]*1000
        ci = CI10(dta)
        plt.loglog(ds[i]*np.ones(2),ci,"k",label="_nolegend_")
        plt.loglog(ds[i]*np.array([0.99,1.01]),ci[0]*np.ones(2),"k",label="_nolegend_")
        plt.loglog(ds[i]*np.array([0.99,1.01]),ci[1]*np.ones(2),"k",label="_nolegend_")
        plt.loglog(ds[i],np.mean(dta),"sk",markersize=10)
        m3[i] = np.mean(dta)
        
        if(i==0):
            plt.loglog(ds[0:(maxd+1)],15*(ds[0:(maxd+1)]/1024)**(-1.0/4.0),'0.8')
        


plt.plot(ds[0:maxind+1],m1[0:maxind+1],"r--")
plt.plot(ds[0:maxind+1],m2[0:maxind+1],"--",color="grey")
plt.plot(ds[0:maxind+1],m3[0:maxind+1],"k--")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("dimension $d$",fontsize=20)
plt.ylabel("ESS per 1000 grandient evaluations",fontsize=20)
plt.legend(["WALNUTS R2P","WALNUTS D","NUTS","$\propto d^{-1/4}$"],fontsize=20)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14  
fig_size[1] = 7 
plt.rcParams["figure.figsize"] = fig_size
plt.savefig("gaussESS.pdf",)



