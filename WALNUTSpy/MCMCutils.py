#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 10:04:50 2025

@author: torekleppe
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy as sp


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
 