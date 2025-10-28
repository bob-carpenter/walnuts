import numpy as np
import targets as td
import matplotlib.pyplot as plt
import MCMCutils as ut
import pandas as pd
import copy
from abc import ABC



LOG_ZERO_THRESH = -700.0
DELTA_THRESH = 100.0



class MCstate:
    
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
        
        
    def momentumFlip(self):
        self.u = -self.u
    
    def __str__(self):
        return ("MCState class:\nq: " + str(self.q) + "\nu: " + str(self.u) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("MCState class:\nq: " + str(self.q) + "\nu: " + str(self.u) + "\nHamiltonian: " + str(self.Ham))


def badMCState(d):
    s = MCstate()
    s.q = np.repeat(np.nan, d)
    s.u = np.repeat(np.nan, d)
    return(s)
    


class MCtuningPars:
    
    def __init__(self, hMacro=0.1, delta=0.1, Cmin=0,Cmax=16):
        self.hMacro = hMacro
        self.delta = delta
        self.Cmin = Cmin
        self.Cmax = Cmax

class MCintegrators(ABC):
    
    def integrateSplitting(self,s,lpFun,h=0.1,nstep=1):
        
        q = s.q
        u = s.u
        g = s.g
        f0 = s.f
        
        d = len(q)
        gnorm = np.linalg.norm(g)
        delta = 0.5*h*gnorm/(d-1)
        e = (1.0/gnorm)*g
        W = 0.0
        
        if(delta>DELTA_THRESH): return(badMCState(len(q)),np.nan)
        
        
        for i in range(nstep):
            
            
            ep = np.dot(e,u)
            
            tmp = np.cosh(delta) + ep*np.sinh(delta)
            
            if(tmp<1.0e-14): return(badMCState(len(q)),np.nan)
            
            W += (d-1)*np.log(tmp)
            uh = (1.0/tmp)*u + ((np.sinh(delta) + ep*(np.cosh(delta)-1.0))/tmp)*e
            
            q = q+h*uh
            
            f,g = lpFun(q)
            
            
            gnorm = np.linalg.norm(g)
            delta = 0.5*h*gnorm/(d-1)
            
            if(delta>DELTA_THRESH): return(badMCState(len(q)),np.nan)
            
            e = (1.0/gnorm)*g
            
            ep = np.dot(e,uh)
            
            tmp = np.cosh(delta) + ep*np.sinh(delta)
            
            if(tmp<1.0e-14): return(badMCState(len(q)),np.nan)
            
            W += (d-1)*np.log(tmp)
            u = (1.0/tmp)*uh + ((np.sinh(delta) + ep*(np.cosh(delta)-1.0))/tmp)*e

        #W += f0-f
        
        sOut = MCstate()
        sOut.q = q
        sOut.u = u
        sOut.f = f
        sOut.g = g
        sOut.Ham = -f
        
        return(sOut,W)
    
    def integrateSplittingErrEst(self,s,lpFun,h=0.1,nstep=1):
        
        q = s.q
        u = s.u
        g = s.g
        f0 = s.f
        
        d = len(q)
        gnorm = np.linalg.norm(g)
        delta = 0.5*h*gnorm/(d-1)
        e = (1.0/gnorm)*g
        W = 0.0
        
        if(delta>DELTA_THRESH): return(badMCState(len(q)),np.nan,np.nan)
        
        errEstQ = np.zeros_like(q)
        errEstU = np.zeros_like(q)
        
        for i in range(nstep):
            
            eulQforw = q + h*u
            eulUforw = u + (h/(d-1))*(g-np.dot(g,u)*u)
            eulUforw = eulUforw/np.linalg.norm(eulUforw)
            
            qOld = q
            uOld = u
            
            ep = np.dot(e,u)
            
            tmp = np.cosh(delta) + ep*np.sinh(delta)
            
            if(tmp<1.0e-14): return(badMCState(len(q)),np.nan,np.nan)
            
            W += (d-1)*np.log(tmp)
            uh = (1.0/tmp)*u + ((np.sinh(delta) + ep*(np.cosh(delta)-1.0))/tmp)*e
            
            q = q+h*uh
            
            f,g = lpFun(q)
            
            
            gnorm = np.linalg.norm(g)
            delta = 0.5*h*gnorm/(d-1)
            
            if(delta>DELTA_THRESH): return(badMCState(len(q)),np.nan,np.nan)
            
            e = (1.0/gnorm)*g
            
            ep = np.dot(e,uh)
            
            tmp = np.cosh(delta) + ep*np.sinh(delta)
            
            if(tmp<1.0e-14): return(badMCState(len(q)),np.nan,np.nan)
            
            W += (d-1)*np.log(tmp)
            u = (1.0/tmp)*uh + ((np.sinh(delta) + ep*(np.cosh(delta)-1.0))/tmp)*e
            
            
            
            errqF = np.abs(q-eulQforw)
            erruF = np.abs(u-eulUforw)
            
            errqB = np.abs(qOld - (q-h*u))
            
            uback = -u + (h/(d-1))*(g-np.dot(g,u)*u)
            uback = uback/np.linalg.norm(uback)
            
            erruB = np.abs(-uOld - uback)
            
            errEstQ += np.maximum(errqF,errqB)
            errEstU += np.maximum(erruF,erruB)
            
        #print("errEst")
        
        errEst = max(np.max(errEstQ),np.max(errEstU))    

        #print(errEst)
        #W += f0-f
        
        sOut = MCstate()
        sOut.q = q
        sOut.u = u
        sOut.f = f
        sOut.g = g
        sOut.Ham = -f
        
        return(sOut,W,errEst)
    
    
    
class fixedMCstep(MCintegrators):
    def __call__(self,s,lpFun,h=0.1):
        return(self.integrateSplitting(s,lpFun,h=h,nstep=1))
    
    

class adaptMCstepE(MCintegrators):
    
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
        return(MCstate())
    
    def propBasic(self):
        return(np.mean(self.basic))
    
    def Cobs(self):
        return(np.max(self.Cobs))
    
    def __call__(self,s,lpFun,tp):
        
        
        
        If = tp.Cmax
        for c in range(tp.Cmin,tp.Cmax+1):
            nstep = 2**c
            (sOut,Wout) = self.integrateSplitting(s,lpFun,h=(tp.hMacro/nstep),nstep=nstep)
            self.gradEval += nstep
            locAcc = -sOut.Ham - Wout + s.Ham
            
            Cobs = np.abs(locAcc)*nstep**2/(tp.hMacro**3)
            
            #print(locAcc)
            if(np.abs(locAcc) < tp.delta):
                If = c
                break
        
        self.HamErrs.append(locAcc)
        self.Cobs.append(Cobs)
        Ib = If
        if(If>tp.Cmin):
            
            self.basic.append(0)
            
            sF = copy.deepcopy(sOut)
            sF.momentumFlip()
            
            for c in range(tp.Cmin,If):
                nstep = 2**c
                (sTmp,Wtmp) = self.integrateSplitting(sF,lpFun,h=(tp.hMacro/nstep),nstep=nstep)
                self.gradEval += nstep
                
                locAcc = -sTmp.Ham - Wtmp + sF.Ham
                #print(locAcc)
                
                if(np.abs(locAcc) < tp.delta):
                    Ib = c
                    break
        else:
            
            self.basic.append(1)

        Wout = -Wout
        self.logJacs.append(Wout)
        self.Ifs.append(If)
        self.Ibs.append(Ib)
        
        if(Ib < If): Wout += LOG_ZERO_THRESH 
        
        return(sOut,Wout)
        





class adaptMCstepFlow(MCintegrators):
    
    def __init__(self):
        self.gradEval = 0
        self.Hams = []
        self.logJacs = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []
    
    def reset(self):
        self.gradEval = 0
        self.Hams = []
        self.logJacs = []
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
        return(MCstate())
    
    def propBasic(self):
        return(np.mean(self.basic))
    
    def Cobs(self):
        return(np.median(self.Cobs))
    
    def __call__(self,s,lpFun,tp):
        
        if not self.Hams:
             self.Hams.append(s.Ham)
        
        If = tp.Cmax
        
        oldQ = np.repeat(1.0e100,len(s.q))
        oldU = np.repeat(1.0e100,len(s.q))
        
        
        
        for c in range(tp.Cmin,tp.Cmax+1):
            nstep = 2**c
            (sOut,Wout) = self.integrateSplitting(s,lpFun,h=(tp.hMacro/nstep),nstep=nstep)
            self.gradEval += nstep
            
            maxErr = np.max(np.abs(sOut.q-oldQ))
            maxErr = max(maxErr,np.max(np.abs(sOut.u-oldU)))
            
            Cobs = maxErr*(nstep//2)**2/(tp.hMacro**3)
            
            
            
            if(maxErr < tp.delta):
                sF = copy.deepcopy(sOut)
                sF.momentumFlip()
                nstepb = 2**(c-1)
                (sTmp,WTmp) = self.integrateSplitting(sF,lpFun,h=(tp.hMacro/nstepb),nstep=nstepb)
                self.gradEval += nstepb
                maxErrb = np.max(np.abs(sTmp.q-s.q))
                maxErrb = max(maxErrb,np.max(np.abs(sTmp.u+s.u)))
                Cobs = max(Cobs,maxErrb*(nstep//2)**2/(tp.hMacro**3))
                if(maxErrb< tp.delta):
                    If = c
                    break
                
                
            
            
            oldQ = sOut.q
            oldU = sOut.u
        
        Ib = If
        self.Hams.append(s.Ham)
        self.Cobs.append(Cobs)

        sF = copy.deepcopy(sOut)
        sF.momentumFlip()
        
        oldQ = np.repeat(1.0e100,len(s.q))
        oldU = np.repeat(1.0e100,len(s.q))
        
        
        if(If>tp.Cmin+1):
            
            self.basic.append(0)
            
            for c in range(tp.Cmin,If):
                nstep = 2**c
                (sB,_) = self.integrateSplitting(sF,lpFun,h=(tp.hMacro/nstep),nstep=nstep)
                self.gradEval += nstep
                maxErr = np.max(np.abs(sB.q-oldQ))
                maxErr = max(maxErr,np.max(np.abs(sB.u-oldU)))
                
                if(maxErr < tp.delta):
                    sBF = copy.deepcopy(sB)
                    sBF.momentumFlip()
                    nstepb = 2**(c-1)
                    (sTmp,WTmp) = self.integrateSplitting(sBF,lpFun,h=(tp.hMacro/nstepb),nstep=nstepb)
                    self.gradEval += nstepb
                    maxErrb = np.max(np.abs(sTmp.q-sF.q))
                    maxErrb = max(maxErrb,np.max(np.abs(sTmp.u+sF.u)))
                    
                    if(maxErrb< tp.delta):
                        Ib = c
                        break
                    
                    
                
                
                oldQ = sB.q
                oldU = sB.u
            
            
        else:
            
            self.basic.append(1)

        self.Hams.append(sOut.Ham)
        self.logJacs.append(Wout)
        self.Ifs.append(If)
        self.Ibs.append(Ib)
            
       
        Wout = -Wout
        
        if(Ib < If): Wout += LOG_ZERO_THRESH 
        
        return(sOut,Wout)



class adaptMCstepFlow2(MCintegrators):
    
    def __init__(self):
        self.gradEval = 0
        self.Hams = []
        self.HamErrs = []
        self.logJacs = []
        self.Ifs = []
        self.Ibs = []
        self.basic = []
        self.Cobs = []
    
    def reset(self):
        self.gradEval = 0
        self.Hams = []
        self.HamErrs = []
        self.logJacs = []
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
        return(MCstate())
    
    def propBasic(self):
        return(np.mean(self.basic))
    
    def Cobs(self):
        return(np.median(self.Cobs))
    
    def __call__(self,s,lpFun,tp):
        
        If = tp.Cmax
        for c in range(tp.Cmin,tp.Cmax+1):
            nstep = 2**c
            (sOut,Wout,errEst) = self.integrateSplittingErrEst(s,lpFun,h=(tp.hMacro/nstep),nstep=nstep)
            self.gradEval += nstep
            locAcc = -sOut.Ham - Wout + s.Ham
            
            Cobs = np.abs(locAcc)*nstep**2/(tp.hMacro**3)
            #print(errEst)
            
            if(errEst<tp.delta):
                If = c
                break
        
        self.HamErrs.append(locAcc)
        self.Cobs.append(Cobs)
        
        #print(If)
        Ib = If
        
        if(If>tp.Cmin):
            self.basic.append(0)
            sF = copy.deepcopy(sOut)
            sF.momentumFlip()
            
            for c in range(tp.Cmin,If+1):
                nstep = 2**c
                (_,_,errEst) = self.integrateSplittingErrEst(sF,lpFun,h=(tp.hMacro/nstep),nstep=nstep)
                self.gradEval += nstep
                if(errEst<tp.delta):
                    Ib = c
                    break
            
            
            
        else:
            self.basic.append(1)
                
            
            
        #print((If,Ib)) 

        self.Hams.append(sOut.Ham)
        self.logJacs.append(Wout)
        self.Ifs.append(If)
        self.Ibs.append(Ib)
            
       
        Wout = -Wout
        
        if(Ib < If): Wout += LOG_ZERO_THRESH 
        
        return(sOut,Wout)

#s = MCstate()




#q0 = np.array([-3.0,0.2])

#lp = td.funnel1

#tp = MCtuningPars()
#tp.hMacro = 0.3

#s.firstEval(lp, q0,  tp)
#s.momentumRefresh(lp,tp)




# step = adaptMCstepE()
# step(s,lp,tp,h=0.05)
# print(step.gradEval)
#stepf = adaptMCstepFlow2()
#sF,W = stepf(s,lp,tp)

#sF.momentumFlip()
#print("back")
#sb0,Wb = stepf(sF,lp,tp)


# print(stepf.gradEval)




