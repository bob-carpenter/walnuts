#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:51:30 2025

@author: torekleppe
"""

import numpy as np
import scipy.stats as sps

# -------------------------------------
# example target distributions
# -------------------------------------

# standard gaussian distribution

def stdGauss(q):
    lp = -0.5*np.sum(q*q)
    grad = -q
    return [lp,grad]

# bivariate zero-mean gaussian with unit marginal variances and correlation=rho

def corrGauss(q):
    rho = 0.5
    tmp = 1.0-rho**2
    lp = -0.5*q[0]**2 -(0.5/tmp)*(q[1]-rho*q[0])**2
    grad = np.array([-(q[0]-rho*q[1])/tmp,
                     -(q[1]-rho*q[0])/tmp])
    return [lp,grad]

# distribution q[0] \sim N(0,1), q[1]|q[0] \sim N(q[0]^2,1)
def smileDistr(q):
    lp = -0.5*q[0]**2 - 0.5*(q[1]-q[0]**2)**2
    grad = np.array([-q[0] + 2.0*q[0]*q[1] - 2.0*q[0]**3,
                     q[0]**2 - q[1]])
    return [lp,grad]


def modFunnel(q):
    x = q[0]
    y = q[1]
    t1 = np.exp(-3.0 * x)
    t2 = 1.0 + t1
    t3 = 0.1e1 / t2
    t4 = y**2
    t5 = -0.1e1 / 0.2e1
    lp = t5 * (t2 * t4 + np.log(t3) + x**2)
    grad = np.array([0.3e1 / 0.2e1 * t1 * (t4 - t3) - x,-y * t2])
    return [lp,grad]


def funnel10(q):
    lp=sps.norm.logpdf(q[0],loc=0.0,scale=3.0) + sum(sps.norm.logpdf(q[1:11],loc=0.0,scale=np.exp(0.5*q[0])))
    grad=np.r_[np.array([-5.0-q[0]/9.0 + 0.5*np.exp(-q[0])*sum(q[1:11]*q[1:11])])
               ,-q[1:11]*np.exp(-q[0])]
    return [lp,grad]


def funnel10rescaled(q):
    S = np.ones(11)
    S[0] = 3.0
    qb = S*q
    lp,g = funnel10(qb)
    return [lp,S*g]

def funnel1(q):
    lp = sps.norm.logpdf(q[0],loc=0.0,scale=3.0) + sps.norm.logpdf(q[1],loc=0.0,scale=np.exp(0.5*q[0]))
    grad = np.array([-0.5-q[0]/9 + 0.5*q[1]**2*np.exp(-q[0]),
                     -q[1]*np.exp(-q[0])])
    return [lp,grad]


