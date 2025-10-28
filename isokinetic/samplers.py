import numpy as np

import copy
import matplotlib.pyplot as plt
import pandas as pd
import dualAverage as da
import P2quantile as P2
LOG_ZERO_THRESH = -700.0





class fullOrbit_:
    def __init__(self,s):
        self.ind = 0
        self.qs = s.q.reshape(-1,1)
        self.Hams = np.array([s.Ham])
        

    def pushF(self,s):
        self.qs = np.hstack([self.qs, s.q.reshape(-1,1)])
        self.Hams = np.concat((self.Hams, np.array([s.Ham])))
    
    def pushB(self,s):
        self.qs = np.hstack([s.q.reshape(-1,1),self.qs])
        self.Hams = np.concat((np.array([s.Ham]),self.Hams))
        self.ind += 1

    def plot(self):
        plt.subplot(1,2,1)
        plt.plot(self.qs[0,:],self.qs[1,:],".-")
        plt.plot(self.qs[0,self.ind],self.qs[1,self.ind],"gs")
        plt.subplot(1,2,2)
        grd = np.linspace(-self.ind, len(self.Hams)-1 - self.ind,num=len(self.Hams))
        plt.plot(grd,self.Hams,'.-')
        plt.plot(grd[self.ind],self.Hams[self.ind],"gs")


def plotOrbit(step,lpFun,q0,tp,
                       generated=lambda q : q, 
                       h0=0.2,L=20):
    h = h0
    sc = step.getState()
    sc.firstEval(lpFun,q0,tp)
    sc.momentumRefresh(lpFun,tp)
    orbit_ = fullOrbit_(sc)
    sf = copy.deepcopy(sc)
    acc = 0.0
    for i in range(1,L+1):
        [sf,accLWt] = step(sf,lpFun,tp,h=h)
        orbit_.pushF(sf)
        acc += accLWt
        print(acc)
        
    orbit_.plot()
    print(step.diagnostics())

def multinomialSampler(step,lpFun,q0,tp0,WASPS=True,
                       generated=lambda q : q, 
                       L=20,niter=1000,
                       nwarmup=500,
                       scale=1.0,
                       ESSTarget=0.99,
                       basicTarget=0.9,
                       center=0.0,
                       storeOrbit_=False,
                       orbitStats_=False
                       ):
    d = len(q0)
    
    if (np.ndim(scale) == 0):
        svec = np.repeat(scale, d)
    else:
        svec = scale
    
    def lpScaled(q):
        f,g = lpFun(svec*q)
        return(f,g*svec)
    
    
    
    tp = copy.deepcopy(tp0)

    sc = step.getState()
    sc.firstEval(lpScaled,q0/svec,tp)
    
    if(WASPS):
        if (np.ndim(center) == 0):
            cen = np.repeat(center, d)/svec
        else:
            cen = center/svec
    
    g0 = generated(q0)
    samples = np.zeros((len(g0),niter+1))
    samples[:,0] = g0
    
    diagnostics = []
    
    
    
    
    
    
    
    
    if(nwarmup>0):
        deltaDA = da.dualAverage(tp0.delta, ESSTarget)
        hP2 = P2.P2quantile(basicTarget)
        

    if(orbitStats_):
        orbitMin_ = np.zeros((len(g0),niter))
        orbitMax_ = np.zeros((len(g0),niter))


    for it in range(niter):
        
        if((it+1) % 1000 == 0): print("iteration # " + str(it+1))
        
        sc.momentumRefresh(lpScaled,tp)
        
        if(WASPS):
            z1 = np.random.normal(size=d)
            z2 = np.random.normal(size=d)
            eta = (1.0/np.sum(z1**2))*z1
            z2 = z2 - (np.sum(z2*eta))*eta
            gam = (1.0/np.sum(z2**2))*z2
            
        
        step.reset()
        
        lwts = np.zeros(L)
        
        nf = np.random.random_integers(low=0,high=L-1,size=1)[0]
        nb = L - nf - 1
        
        
        
        if(storeOrbit_): orbit_ = fullOrbit_(sc)
        
        if(orbitStats_):
            gen = generated(sc.q*svec)
            orbitMin_[:,it] = gen
            orbitMax_[:,it] = gen
        
        sSampled = copy.deepcopy(sc)
        sInd = 0
        
        mnwtSum = 1.0
        nStepsF = 0
        nStepsB = 0
        sf = copy.deepcopy(sc)
        
        accLogWtSum = 0.0
        
        deadForw = 0
        deadBack = 0
        
        # forward integration
        
        for i in range(1,nf+1):
            if(WASPS): qOld = sf.q
            
            [sf,accLWt] = step(sf,lpScaled,tp)
            
            if(storeOrbit_): orbit_.pushF(sf)
            
            if(orbitStats_): 
                gen = generated(sf.q*svec)
                orbitMin_[:,it] = np.minimum(orbitMin_[:,it],gen)
                orbitMax_[:,it] = np.maximum(orbitMax_[:,it],gen)
                
            accLogWtSum += accLWt
            
            if(accLogWtSum < LOG_ZERO_THRESH+10.0): 
                deadForw = 1
                break
            
            if(WASPS):
                cqs = sf.q-cen
                cq = qOld-cen
                
                stop1 = np.dot(cqs,eta)*np.dot(cq,eta) < 0.0 
                stop2 = max(np.dot(gam,cqs),np.dot(gam,cq)) > 0.0
                
                if(stop1 and stop2):
                    break
            
            lwts[nb+i] = sc.Ham - sf.Ham + accLogWtSum
            nStepsF += 1
            
            wt = np.exp(lwts[nb+i])
            mnwtSum += wt
            if(np.random.uniform()<wt/mnwtSum):
                sInd = i
                sSampled = copy.deepcopy(sf)
            
            #if(partialRefresh):
            #    sf.partialMomentumRefresh(lpFun,tp,np.exp(-tp.hMacro/LP))
            
        
        sf = copy.deepcopy(sc)
        sf.momentumFlip()
        accLogWtSum = 0.0
        
        # backward integration
        
        for i in range(1,nb+1):
            if(WASPS): qOld = sf.q
            
            [sf,accLWt] = step(sf,lpScaled,tp)
            
            if(storeOrbit_): orbit_.pushB(sf)
            
            if(orbitStats_): 
                gen = generated(sf.q*svec)
                orbitMin_[:,it] = np.minimum(orbitMin_[:,it],gen)
                orbitMax_[:,it] = np.maximum(orbitMax_[:,it],gen)
            
            accLogWtSum += accLWt
            
            if(accLogWtSum < LOG_ZERO_THRESH+10.0): 
                deadBack = 1
                break
            
            if(WASPS):
                cqs = sf.q-cen
                cq = qOld-cen
                
                stop1 = np.dot(cqs,eta)*np.dot(cq,eta) < 0.0 
                stop2 = max(np.dot(gam,cqs),np.dot(gam,cq)) > 0.0
                
                if(stop1 and stop2):
                    break
            
            lwts[nb-i] = sc.Ham - sf.Ham + accLogWtSum
            nStepsB += 1
            
            wt = np.exp(lwts[nb-i])
            mnwtSum += wt
            if(np.random.uniform()<wt/mnwtSum):
                sInd = -i
                sSampled = copy.deepcopy(sf)
                
            #if(partialRefresh):
            #    sf.partialMomentumRefresh(lpFun,tp,np.exp(-tp.hMacro/LP))
        
        sc = copy.deepcopy(sSampled)
        samples[:,it+1] = generated(sc.q*svec)
        
        lwtRange = np.max(lwts) - np.min(lwts)
        
        usedWts = lwts[(nb-nStepsB):(nb+nStepsF+1)]
        usedWts = usedWts - np.max(usedWts)
        usedWts = np.exp(usedWts)
        ESSfrac = np.sum(usedWts)**2/(len(usedWts)*np.sum(usedWts**2))
        
        if(it<nwarmup):
            
        
            deltaDA.observe(ESSfrac)
            if(it>10): tp.delta = deltaDA.par()
        
            hP2.pushVec(np.log(step.Cobs))
            
            
            if(it>10): tp.hMacro = (tp.delta/np.exp(hP2.quantile()))**(1.0/3.0)
        
        
        samplerDiag = pd.Series([tp.hMacro,nf,sInd,
                                 deadForw,
                                 deadBack,lwtRange,nStepsF+nStepsB,
                                 ESSfrac,
                                 tp.delta],index=['h','numForw','sampleIndex',
                                                  'deF',
                                                  'deB','lwtRange','nSteps',
                                                  'ESSfrac',
                                                  'delta'])
        diagnostics.append(pd.concat([samplerDiag,step.diagnostics()]))
        
        
        
        
    if(storeOrbit_): orbit_.plot()
        
    diagDf = pd.DataFrame(diagnostics)    
    
    if(orbitStats_):
        return(samples,diagDf,tp,orbitMin_,orbitMax_)
    else:
        return(samples,diagDf,tp)


def dampedLangevin(step,lpFun,q0,tp0,
                       generated=lambda q : q, 
                       L=20,niter=1000,
                       nwarmup=500,
                       accTarget=0.99,
                       basicTarget=0.9):
    
    d = len(q0)
    
    tp = copy.deepcopy(tp0)

    sc = step.getState()
    sc.firstEval(lpFun,q0,tp)
    
    
    g0 = generated(q0)
    samples = np.zeros((len(g0),niter+1))
    samples[:,0] = g0
    
    diagnostics = []
    
    
    for it in range(niter):
        
        if((it+1) % 1000 == 0): print("iteration # " + str(it+1))
        
        sc.momentumRefresh(lpScaled,tp)
        
        
        for i in range(L):
            
            sc.partialMomemtumRefresh(lpFun,tp,np.exp(-0.5*tp.hMacro/L))
            
            
            
            
            
        
    
        

