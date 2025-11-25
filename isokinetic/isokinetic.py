import numpy as np
import targets as td
import matplotlib.pyplot as plt
import MCMCutils as ut
import pandas as pd
import copy
from abc import ABC



LOG_ZERO_THRESH = -700.0
DELTA_THRESH = 100.0



class IKState:
    
    def __init__(self):
        self.Ham = np.nan
    
    def firstEval(self,lpFun,q,tp):
        [f,g] = lpFun(q)
        self.q = q
        self.f = f
        self.g = g
        self.u = np.repeat(np.nan,len(q))
        self.Ham = -f
    
    def momentumRefresh(self,lpFun,tp):
        p = np.random.normal(size=len(self.q))
        self.u = (1.0/np.linalg.norm(p))*p 
        
    def partialMomentumRefresh(self,lpFun,tp,c1):
        Z = np.random.normal(size=len(self.q))
        Z = Z/np.sqrt(len(self.q))
        t = c1*self.u + np.sqrt(1.0-c1**2)*Z
        self.u = t/np.linalg.norm(t)
        
    def velocity(self):
        return(self.u)
        
    
    def momentumFlip(self):
        self.u = -self.u
    
    def __str__(self):
        return ("IKState class:\nq: " + str(self.q) + "\nu: " + str(self.u) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("IKState class:\nq: " + str(self.q) + "\nu: " + str(self.u) + "\nHamiltonian: " + str(self.Ham))





class IKtuningPars:
    
    def __init__(self, hMacro=0.1, delta=0.1, Cmin=0,Cmax=16):
        self.hMacro = hMacro
        self.delta = delta
        self.Cmin = Cmin
        self.Cmax = Cmax

class IKintegrators: #(ABC):
    
    def BAB(self,s,lpFun,h=0.1,nstep=1):
        d = len(s.q)        
        hh = h #np.sqrt(d-1)*h
        q = s.q
        u = s.u
        g = s.g
        gnorm = np.linalg.norm(g)
        delta = (0.5*hh/(d-1))*gnorm
        ee = (1.0/gnorm)*g
        ljac = 0.0
        errEst = 0.0
        Hold = -s.f
        
        for i in range(nstep):
            
            sh = np.sinh(delta)
            ch = np.cosh(delta)
            
            ip = np.dot(ee,u)
            t1 = ch+ip*sh
            
            ljacStep = -(d-1)*np.log(t1)
            
            wtu = 1.0/t1
            wte = (sh + ip*(ch-1.0))/t1
            
            uh = wtu*u + wte*ee
            uh = uh/np.linalg.norm(uh)
            
            q = q + hh*uh
            [f,g] = lpFun(q)
                        
            gnorm = np.linalg.norm(g)
            delta = (0.5*hh/(d-1))*gnorm
            ee = (1.0/gnorm)*g
            
            sh = np.sinh(delta)
            ch = np.cosh(delta)
            ip = np.dot(ee,uh)
            t1 = ch+ip*sh
            
            ljacStep -= (d-1)*np.log(t1)
            
            wtu = 1.0/t1
            wte = (sh + ip*(ch-1.0))/t1
            
            u = wtu*uh + wte*ee
            u = u/np.linalg.norm(u)
            
            stepErr = Hold + f + ljacStep
            errEst += np.abs(stepErr)/nstep # max(errEst,np.abs(stepErr))
            Hold = -f
            ljac += ljacStep
            
        sOut = IKState()
        sOut.q = q
        sOut.u = u
        sOut.f = f
        sOut.g = g
        sOut.Ham = -f
        
        Cest = errEst*h**(-3.0)
        
        return(sOut,ljac,Cest)
            
          
class orderChecks(IKintegrators):
    def oneStep(self,s,lpFun):
        hs = 2.0**(-np.linspace(0.0, 14.0,num=100))
        es = np.zeros_like(hs)
        for i in range(len(hs)):
            (s1,ljac) = self.BAB(s, lpFun,h=hs[i])
            es[i] = np.abs(s.Ham - s1.Ham + ljac)
        
        
        plt.loglog(hs,es,'.',label='observed E-error')
        plt.loglog(hs,hs**3*es[-1]/(hs[-1]**3),label='propto h^3')
        plt.legend()
        plt.xlabel("h")

    def fixedTime(self,s,lpFun,Tmax=3.0):
        
        nsteps = 2**np.arange(16)
        hs = Tmax/nsteps
        es = np.zeros_like(hs)
        for i in range(len(nsteps)):
            (s1,ljac) = self.BAB(s, lpFun,h=hs[i],nstep=nsteps[i])
            es[i] = np.abs(s.Ham - s1.Ham + ljac)
        
        
        plt.loglog(hs,es,'.',label='observed E-error')
        plt.loglog(hs,hs**2*es[-1]/(hs[-1]**2),label='propto h^2')
        plt.legend()
        plt.xlabel("h")



class adaptIKstepE(IKintegrators):
    
    def __init__(self):
        self.gradEval = 0
        self.HamErrs = []
        self.logJacs = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []
    
    
    def reset(self):
        self.gradEval = 0
        self.HamErrs = []
        self.logJacs = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []

    def diagnostics(self):
        
        return (pd.Series([self.gradEval, np.max(np.abs(self.HamErrs)),
                          np.min(self.Ifs),
                          np.max(self.Ifs),
                          np.mean(self.basic)],
                         index=['gradEvals', 'energyErr',
                                'minIf', 'maxIf', 'propBasic']))


    
    def getState(self):
        return(IKState())
    
    def propBasic(self):
        return(np.mean(self.basic))
    
    def Cobs(self):
        return(np.mean(self.Cobs))
    
    def __call__(self,s,lpFun,tp):
        
        
        
        If = tp.Cmax
        for c in range(tp.Cmin,tp.Cmax+1):
            nstep = 2**c
            (sOut,Wout,Cest) = self.BAB(s,lpFun,h=(tp.hMacro/nstep),nstep=nstep)
            self.gradEval += nstep
            locAcc = -sOut.Ham + s.Ham + Wout
            
            
            #print(locAcc)
            if(np.abs(locAcc) < tp.delta):
                If = c
                break
        
        self.HamErrs.append(locAcc)
        self.Cobs.append(Cest)
        Ib = If
        if(If>tp.Cmin):
            
            self.basic.append(0)
            
            sF = copy.deepcopy(sOut)
            sF.momentumFlip()
            
            for c in range(tp.Cmin,If):
                nstep = 2**c
                (sTmp,Wtmp,_) = self.BAB(sF,lpFun,h=(tp.hMacro/nstep),nstep=nstep)
                self.gradEval += nstep
                
                locAcc = -sTmp.Ham + sF.Ham + Wtmp
                #print(locAcc)
                
                if(np.abs(locAcc) < tp.delta):
                    Ib = c
                    break
        else:
            
            self.basic.append(1)

        
        self.logJacs.append(Wout)
        self.Ifs.append(If)
        self.Ibs.append(Ib)
        
        if(Ib < If): Wout += LOG_ZERO_THRESH 
        
        return(sOut,Wout)
        






        
lp = td.funnel10 #td.modFunnel   # td.stdGauss #  
s= IKState()
tp = IKtuningPars(hMacro=0.5)
s.firstEval(lp, 3*np.random.normal(size=11), tp) 
s.momentumRefresh(lp, tp)   

step = adaptIKstepE()
(s1,ljac) = step(s,lp,tp)

#ff = orderChecks()
#ff.fixedTime(s, lp)


