import targets as td
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy
import pandas as pd
from abc import ABC

LOG_ZERO_THRESH = -700.0



class CMtuningPars:
    
    def __init__(self, rho=0.9, hMacro=0.1, delta=0.1, Cmin=0,Cmax=16):
        self.rho = rho
        self.hMacro = hMacro
        self.delta = delta
        self.Cmin = Cmin
        self.Cmax = Cmax

class CMState:
    
    def __init__(self):
        self.Ham = np.nan
        
    
    def firstEval(self,lpFun,q,tp):
        [f,g] = lpFun(q)
        self.q = q
        self.f = f
        self.g = g
        self.p = np.repeat(np.nan,len(q))
        self.qe = np.repeat(np.nan,len(q))
        self.pe = np.repeat(np.nan,len(q))
        self.Ham = np.nan
        
        
    def momentumRefresh(self,lpFun,tp):
        
        self.p = np.random.normal(size=len(self.q))
        self.pe = tp.rho*self.p + np.sqrt(1.0-tp.rho**2)*np.random.normal(size=len(self.q))
        self.qe = np.random.normal(size=len(self.q))
        self.Ham = -self.f + 0.5*np.dot(self.qe,self.qe) + 0.5/(1.0-tp.rho**2)*(np.dot(self.p,self.p) + np.dot(self.pe,self.pe) - 2.0*tp.rho*np.dot(self.p,self.pe))
        
        
    def velocity(self):
        return(self.p)
        
    
    def momentumFlip(self):
        self.p = -self.p
        self.pe = -self.pe
    
    def __str__(self):
        return ("CMState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("CMState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))


class CMintegrators: #(ABC):
    def LF(self,s,lpFun,rho,h,nstep=1):
        q = s.q
        p = s.p 
        qe = s.qe
        pe = s.pe
        g = s.g
        
        w1 = 1.0/(1.0-rho**2)
        w2 = -rho/(1.0-rho**2)
        
        for i in range(nstep):
            ph = p + 0.5*h*g
            peh = pe - 0.5*h*qe
            
            q = q + h*(w1*ph + w2*peh)
            qe = qe + h*(w1*peh + w2*ph)
            
            [f,g] = lpFun(q)
            
            p = ph + 0.5*h*g
            pe = peh - 0.5*h*qe
            
        sOut = CMState()
        sOut.q = q
        sOut.p = p
        sOut.qe = qe
        sOut.pe = pe
        sOut.f = f
        sOut.g = g
        sOut.Ham = -f + 0.5*np.dot(qe,qe) + 0.5/(1.0-rho**2)*(np.dot(p,p) + np.dot(pe,pe) - 2.0*rho*np.dot(p,pe))
        return(sOut)
        
    def splitting(self,s,lpFun,rho,h,nstep=1):
        q = s.q
        p = s.p 
        qe = s.qe
        pe = s.pe
        g = s.g
        
        w1 = 1.0/(1.0-rho**2)
        w2 = rho/(1.0-rho**2)
        
        cc = np.cos(np.sqrt(w1)*h)
        ss = np.sin(np.sqrt(w1)*h)
        
        for i in range(nstep):
            ph = p + 0.5*h*g
            
            
            peN = (-ss/np.sqrt(w1))*qe + (-cc*w2/w1)*ph + cc*pe + (w2/w1)*ph
            qeN = np.sqrt(w1)*ss*pe - ss*w2/np.sqrt(w1)*ph + cc*qe
            qN = (w1*h- h*w2**2/w1)*ph - cc*w2/w1*qe + (ss*w2**2/(w1**1.5))*ph - ss*w2/np.sqrt(w1)*pe + q + (w2/w1)*qe
            
            
            pe = peN
            q = qN
            qe = qeN
            
            [f,g] = lpFun(q)
            
            p = ph + 0.5*h*g
            
            
        sOut = CMState()
        sOut.q = q
        sOut.p = p
        sOut.qe = qe
        sOut.pe = pe
        sOut.f = f
        sOut.g = g
        sOut.Ham = -f + 0.5*np.dot(qe,qe) + 0.5/(1.0-rho**2)*(np.dot(p,p) + np.dot(pe,pe) - 2.0*rho*np.dot(p,pe))
        return(sOut)
        
        
class adaptCMstepE(CMintegrators):
    
    def __init__(self):
        self.gradEval = 0
        self.Hams = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []
    
    
    def reset(self):
        self.gradEval = 0
        self.Hams = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []

    def diagnostics(self):
        
        return (pd.Series([self.gradEval, np.max(self.Hams)-np.min(self.Hams),
                          np.min(self.Ifs),
                          np.max(self.Ifs),
                          np.mean(self.basic)],
                         index=['gradEvals', 'energyErr',
                                'minIf', 'maxIf', 'propBasic']))


    
    def getState(self):
        return(CMState())
    
    def propBasic(self):
        return(np.mean(self.basic))
    
    def Cobs(self):
        return(np.median(self.Cobs))
    
    
    def __call__(self,s,lpFun,tp):
        
        if not self.Hams:
            self.Hams.append(s.Ham)
            
        If = tp.Cmax
        
        for c in range(tp.Cmin,tp.Cmax+1):
            nstep = 2**c
            
            sOut = self.LF(s=s,lpFun=lpFun,rho=tp.rho,h=tp.hMacro/nstep,nstep=nstep)
            self.gradEval += nstep
            
            
            
            Eerr = np.abs(sOut.Ham-s.Ham)
            
            
            Cobs = np.abs(Eerr)*nstep**2/(tp.hMacro**3)
            
            if(Eerr < tp.delta):
                If = c
                break
            
        Ib = If
        self.Cobs.append(Cobs)
        
        if(If>tp.Cmin):
            self.basic.append(0)
            sB = copy.deepcopy(sOut)
            sB.momentumFlip()
            
            for c in range(tp.Cmin,If):
                nstep = 2**c
                
                sTmp = self.LF(sB,lpFun,rho=tp.rho,h=tp.hMacro/nstep,nstep=nstep)
                self.gradEval += nstep
                
                Eerr = np.abs(sTmp.Ham-sB.Ham)
                
                if(Eerr < tp.delta):
                    Ib = c
                    break
        
        else:
            self.basic.append(1)

        self.Ifs.append(If)
        self.Ibs.append(Ib)
        self.Hams.append(sOut.Ham)
        
        lwt = 0.0 if Ib==If else LOG_ZERO_THRESH
        
        
        return(sOut,lwt)
        
     

        
# s = CMState()
# tp = CMtuningPars()

# s.firstEval(td.funnel1, np.array([1.0,0.3]), tp)       
# s.momentumRefresh(td.funnel1, tp)


# s1 = CMintegrators().splitting(s, td.funnel1, tp.rho, h=0.1)
# s1.momentumFlip()
# s0 = CMintegrators().splitting(s1, td.funnel1, tp.rho, h=0.1)

# print(s)
# print(s1)
# print(s0)









