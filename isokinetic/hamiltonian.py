import numpy as np
import targets as td
import matplotlib.pyplot as plt
import MCMCutils as ut
import pandas as pd
import copy
from abc import ABC



LOG_ZERO_THRESH = -700.0




class HMCstate:
    
    def firstEval(self,lpFun,q,tp):
        [f,g] = lpFun(q)
        self.q = q
        self.f = f
        self.g = g
        self.p = np.repeat(np.nan,len(q))
        self.Ham = np.nan
        
        
    def momentumRefresh(self,lpFun,tp):
        self.p = np.random.normal(size=len(self.q))
        self.Ham = -self.f + 0.5*np.dot(self.p,self.p)
        
    def momentumFlip(self):
        self.p = -self.p
    
    def __str__(self):
        return ("MCState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("MCState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))

class HMCtuningPars:
    
    def __init__(self, hMacro=0.1, delta=0.1, Cmin=0,Cmax=10):
        
        self.hMacro = hMacro
        self.delta = delta
        
        self.Cmin = Cmin
        self.Cmax = Cmax



class HMCintegrators(ABC):
    
    def integrateLeapFrog(self,s,lpFun,h=0.1,nstep=1):
        
        q = s.q
        p = s.p
        g = s.g 
        
        for i in range(nstep):
            ph = p + 0.5*h*g
            q = q + h*ph
            [f,g] = lpFun(q)
            
            p = ph + 0.5*h*g
        
        sOut = HMCstate()
        sOut.q = q
        sOut.p = p
        sOut.f = f
        sOut.g = g
        sOut.Ham = -f + 0.5*np.dot(p,p)
        
        return(sOut)

class fixedHMCstep(HMCintegrators):
    def __call__(self,s,lpFun,h):
        return(self.integrateLeapFrog(s,lpFun,h=h,nstep=1))
    
    


class adaptHMCstepE(HMCintegrators):
    
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
        return(HMCstate())
    
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
            
            sOut = self.integrateLeapFrog(s,lpFun,h=tp.hMacro/nstep,nstep=nstep)
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
                
                sTmp = self.integrateLeapFrog(sB,lpFun,h=tp.hMacro/nstep,nstep=nstep)
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



class adaptHMCstepF(HMCintegrators):
    
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
        return(HMCstate())
    
    def propBasic(self):
        return(np.mean(self.basic))
    
    def Cobs(self):
        return(np.max(self.Cobs))
    
    def __call__(self,s,lpFun,tp,h=0.1):
        if not self.Hams:
            self.Hams.append(s.Ham)
            
        If = tp.Cmax
        
        qOld = np.repeat(1.0e100,len(s.q))
        pOld = np.repeat(1.0e100,len(s.q))
        
        for c in range(tp.Cmin,tp.Cmax+1):
            nstep = 2**c
            
            sOut = self.integrateLeapFrog(s,lpFun,h=tp.hMacro/nstep,nstep=nstep)
            self.gradEval += nstep
            
            Ferr = np.max(np.abs(qOld-sOut.q))
            Ferr = max(Ferr,np.max(np.abs(pOld-sOut.p)))
            
            Cobs = Ferr*(nstep//2)**2/(tp.hMacro**3)
            
            #print(Ferr)
            if(Ferr < tp.delta):
                nstepb = 2**(c-1)
                sF = copy.deepcopy(sOut)
                sF.momentumFlip()
                
                sTmp = self.integrateLeapFrog(sF,lpFun,h=tp.hMacro/nstepb,nstep=nstepb)
                self.gradEval += nstepb
                
                Ferrb = np.max(np.abs(s.q-sTmp.q))
                Ferrb = max(Ferrb,np.max(np.abs(s.p+sTmp.p)))
                
                Cobs = max(Cobs,Ferrb*(nstep//2)**2/(tp.hMacro**3))
                
                #print("Ferrb : " + str(Ferrb))
                if(Ferrb < tp.delta):
                    If = c
                    break
            
            qOld = sOut.q
            pOld = sOut.p
         
        self.Cobs.append(Cobs)
        
        Ib = If
        
        if(If>tp.Cmin+1):
            self.basic.append(0)
            sB = copy.deepcopy(sOut)
            sB.momentumFlip()
            
            qOld = np.repeat(1.0e100,len(s.q))
            pOld = np.repeat(1.0e100,len(s.q))
            
            for c in range(tp.Cmin,If):
                nstep = 2**c
                
                sTmp = self.integrateLeapFrog(sB,lpFun,h=tp.hMacro/nstep,nstep=nstep)
                self.gradEval += nstep
                
                Ferr = np.max(np.abs(qOld-sTmp.q))
                Ferr = max(Ferr,np.max(np.abs(pOld-sTmp.p)))
                #print(Ferr)
                
                if(Ferr < tp.delta):
                    nstepb = 2**(c-1)
                    sF = copy.deepcopy(sTmp)
                    sF.momentumFlip()
                    
                    sTmp2 = self.integrateLeapFrog(sF,lpFun,h=tp.hMacro/nstepb,nstep=nstepb)
                    self.gradEval += nstep
                    
                    
                    Ferrb = np.max(np.abs(sOut.q-sTmp2.q))
                    Ferrb = max(Ferrb,np.max(np.abs(sOut.p-sTmp2.p)))
                    #print("Ferrb : " + str(Ferrb))
                    if(Ferrb < tp.delta):
                        Ib = c
                        break
                    
                
                qOld = sTmp.q
                pOld = sTmp.p
        
        else:
            self.basic.append(1)

        self.Ifs.append(If)
        self.Ibs.append(Ib)
        self.Hams.append(sOut.Ham)
        
        lwt = 0.0 if Ib==If else LOG_ZERO_THRESH
        
        
        return(sOut,lwt)


# s = HMCstate()

# q0 = np.array([-6.0,0.0])

# lp = td.funnel1

# tp = HMCtuningPars()

# s.firstEval(lp, q0,  tp)
# s.momentumRefresh(lp,tp)

# step = adaptHMCstepF()
# step(s,lp,tp,h=0.5)
# print(step.gradEval)
