import numpy as np
import math
import arviz as az
import copy
import matplotlib.pyplot as plt
import hamiltonian as hmc
import microCanonical as mc
import targets as td
import pandas as pd
import MCMCutils as ut


class fullOrbit_:
    def __init__(self,s):
        self.ind = 0
        self.qs = s.q.reshape(-1,1)
        
        

    def pushF(self,s):
        self.qs = np.hstack([self.qs, s.q.reshape(-1,1)])
        
    
    def pushB(self,s):
        self.qs = np.hstack([s.q.reshape(-1,1),self.qs])
        
        self.ind += 1

    def plot(self):
        
        plt.plot(self.qs[0,:],self.qs[1,:],".-")
        plt.plot(self.qs[0,self.ind],self.qs[1,self.ind],"gs")


class IDcontainer:
    def __init__(self,s,id_):
        self.s = s
        self.id = id_

class stateStore:
    
    def __init__(self):
        self.states = []
        
    def push(self,s,id_):
        self.states.append(IDcontainer(s,id_))
    
    def getIds(self):
        return(np.array([obj.id for obj in self.states]))
    
    def remove(self,id_):
        for i in range(len(self.states)):
            if(self.states[i].id==id_):
                self.states.pop(i)
                break
            
    def removeRange(self,idFrom,idTo):
        i = 0
        while(i<len(self.states)):
            if(self.states[i].id >= idFrom and self.states[i].id <= idTo):
                self.states.pop(i)
            else:
                i += 1
        
    def reset(self):
        self.states = []            
    
    def getState(self,id_):
        for i in range(len(self.states)):
            if(self.states[i].id==id_):
                return(self.states[i].s)
        print("stateStore: could not find state")
        return(0)
            
    
    def __repr__(self):
        string = "stateStore object, size: " + str(len(self.states)) + ", ids: \n"
        string += str(self.getIds()) + "\n"
        return(string)
    
class lwtVector:
    def __init__(self, capacity):
        self.c = capacity
        self.v = np.repeat(np.nan, 2*capacity+1)
    
    def reset(self):
        self.v = np.repeat(np.nan,2*self.c+1)
    
    def __setitem__(self,key,val):
        self.v[key+self.c] = val
    
    def __getitem__(self,key):
        return(self.v[key+self.c])
    
    def __repr__(self):
        return("vals: \n" + str(self.v) + 
               "\n indexes: \n" + 
               str(np.linspace(-self.c, self.c,num=2*self.c+1,dtype=np.int32)))
    
    def normalizedWts(self,start,end):
        
        return(np.exp(self.v[(self.c+start):(self.c+end+1)]-self.v[self.c]))
    
    def normalizedWt(self,key):
        return(np.exp(self.v[self.c + key]-self.v[self.c]))
    
    
def UturnCond(sM,sP):
    tmp = sP.q - sM.q
    return(np.dot(sP.velocity(),tmp)<0.0 or np.dot(sM.velocity(),tmp)<0.0)


class NUTSampler:
    
    
    
    
    def __init__(self,debug=True):
        self.debug = debug
        
    
    
    
    
    def subTreePlan(self,nleaf):
        
        self.checks = np.zeros((nleaf-1,2),dtype=int)
        self.k = 0
        def Uturn(a,b):
            
            self.checks[self.k,0] = a
            self.checks[self.k,1] = b
            self.k += 1

        def subUturn(a,b):
            if(a!=b):
                m = math.floor((a+b)/2)
                subUturn(a,m)
                subUturn(m+1,b)
                Uturn(a,b)
        subUturn(1,nleaf)
        return(self.checks)

    

    def buildOrbit(self,scurr):
        
        self.stateList.reset()
        
        
        # endpoints of orbit (P=forward, M=backward)
        a = 0
        b = 0
        sP = copy.deepcopy(scurr)
        sM = copy.deepcopy(scurr)
        sSampled = copy.deepcopy(scurr)
        
        
        # cumulative jacobians
        cljacP = 0.0
        cljacM = 0.0
        
        
        # log-weights
        self.lwts.reset()
        self.lwts[0] = -scurr.Ham
        
        accWtsum = 1.0
        
        # keep track of which state gets sampled (all trailing underscores are instrumentation)
        L_ = 0
        
        if(self.debug): fo = fullOrbit_(scurr)
            
        
        
        
        # first integration step
        if(np.random.uniform()<0.5):
            #print("first forward")
            # forward integration
            (sP,lJac) = self.step(sP,self.lp,self.tp)
            cljacP += lJac
            b = 1
            self.lwts[b] = -sP.Ham + lJac
            if(self.debug): fo.pushF(sP)
            
            subOrbitWtSum = np.sum(self.lwts.normalizedWts(b, b))
            if(np.random.uniform()<subOrbitWtSum/accWtsum):
                sSampled = copy.deepcopy(sP)
                L_ = b
            
        else:
            #print("first backward")
            # backward integration
            sM.momentumFlip()
            (sM,lJac) = self.step(sM,self.lp,self.tp)
            sM.momentumFlip()
            cljacM += lJac
            a = -1
            self.lwts[a] = -sM.Ham + lJac
            if(self.debug): fo.pushB(sM)
            
            subOrbitWtSum = np.sum(self.lwts.normalizedWts(a, a))
            if(np.random.uniform()<subOrbitWtSum/accWtsum):
                sSampled = copy.deepcopy(sM)
                L_ = a
        
        # check first u-turn
        if(self.stopC(sM,sP)):
            dd=pd.Series([0,L_,a,b,a,b,0],index=['NutsIter','L','a','b','aInt','bInt','NUTtype'])
            
            #print("Uturn at first doubling")
            return(sSampled,dd)
        
        # done with first integration leg, and continuing
        for i in range(1,self.M+1):
            # weight sum of (previously) accepted part of orbit
            accWtsum += subOrbitWtSum
            subOrbitWtSum = 0.0
            self.stateList.reset()
            
            # determine integration direction
            if(np.random.uniform()<0.5):
                # forward integration
                bnew = b
                anew = a
                tasks = b + self.plans[i]
                
                
                for j in range(tasks.shape[0]):
                    
                    
                    if(np.abs(tasks[j,0]-tasks[j,1])==1):
                        # do a pair of integration steps
                        for k in range(2):
                            # integrate and store weight
                            (sP,lJac) = self.step(sP,self.lp,self.tp)
                            cljacP += lJac
                            if(self.debug): fo.pushF(sP)
                            bnew += 1
                            self.lwts[bnew] = -sP.Ham + cljacP
                            
                            # put state in list
                            self.stateList.push(copy.deepcopy(sP), bnew)
                            
                            # maintain sampled state from suborbit
                            wt = self.lwts.normalizedWt(bnew)
                            subOrbitWtSum += wt
                            if(subOrbitWtSum> 0.0 and np.random.uniform() < wt/subOrbitWtSum):
                                subSampled = copy.deepcopy(sP)
                                Lsub_ = bnew
                    if(self.debug): fo.plot()
                    
                    
                    # Uturn-checks
                    if(self.stopC(self.stateList.getState(tasks[j,0]),
                                  self.stateList.getState(tasks[j,1]))):
                        # rejecting the current suborbit
                        dd=pd.Series([i,L_,a,b,anew,bnew,1],index=['NutsIter','L','a','b','aInt','bInt','NUTtype'])
                        #print("sub-orbit: " + str(tasks[j,]))
                        return(sSampled,dd)
                    # clean up memory (intermediate states no longer needed)
                    if(np.abs(tasks[j,0]-tasks[j,1])>1):
                        self.stateList.removeRange(np.min(tasks[j,])+1, np.max(tasks[j,])-1)
                
                
                
            else:
                # backward integration
                
                anew = a
                bnew = b
                tasks = a - self.plans[i]
                
                for j in range(tasks.shape[0]):
                    
                    
                    if(np.abs(tasks[j,0]-tasks[j,1])==1):
                        # do a pair of integration steps
                        for k in range(2):
                            # integrate and store weight
                            sM.momentumFlip()
                            (sM,lJac) = self.step(sM,self.lp,self.tp)
                            sM.momentumFlip()
                            
                            cljacM += lJac
                            if(self.debug): fo.pushB(sM)
                            anew -= 1
                            
                            self.lwts[anew] = -sM.Ham + cljacM
                            
                            # put state in list
                            self.stateList.push(copy.deepcopy(sM), anew)
                            
                            # maintain sampled state from suborbit
                            wt = self.lwts.normalizedWt(anew)
                            subOrbitWtSum += wt
                            if(subOrbitWtSum> 0.0 and np.random.uniform() < wt/subOrbitWtSum):
                                subSampled = copy.deepcopy(sM)
                                Lsub_ = anew
                            
                        
                        if(self.debug): fo.plot()
                    # Uturn-checks
                    if(self.stopC(self.stateList.getState(tasks[j,1]),
                                  self.stateList.getState(tasks[j,0]))):
                        # rejecting the current suborbit
                        dd=pd.Series([i,L_,a,b,anew,bnew,1],index=['NutsIter','L','a','b','aInt','bInt','NUTtype'])
                        #print("sub-orbit U-turn: " + str(tasks[j,]))
                        return(sSampled,dd)
                    
                    # clean up memory (intermediate states no longer needed)
                    if(np.abs(tasks[j,0]-tasks[j,1])>1):
                        self.stateList.removeRange(np.min(tasks[j,])+1, np.max(tasks[j,])-1)
                            
                
                
            # done forward/backward if
            if(np.random.uniform()<subOrbitWtSum/accWtsum):
                #print("accepted state from proposed subOrbit: a = " + str(subOrbitWtSum/accWtsum))
                L_ = Lsub_
                sSampled = copy.deepcopy(subSampled)
            
            # check global U-turn
            if(self.stopC(sM,sP)):
                dd=pd.Series([i,L_,anew,bnew,anew,bnew,0],index=['NutsIter','L','a','b','aInt','bInt','NUTtype'])
                #print("global U-turn")
                return(sSampled,dd)
            
            
            # prepare for new suborbit
            a = anew
            b = bnew
        
        #print("expended available integration steps")
        dd=pd.Series([self.M,L_,a,b,a,b,2],index=['NutsIter','L','a','b','aInt','bInt','NUTtype'])
        return(sSampled,dd)
        

    def run(self,lpFun,q0,
            step=hmc.adaptHMCstepE(),
            tp0=hmc.HMCtuningPars(),
            generated=lambda q : q, 
            niter=1000,
            M=10,
            stopCondition=UturnCond):
        
        # pre-compute U-turn check plans
        self.plans = []
        for i in range(0,M+1):
            self.plans.append(self.subTreePlan(2**i))
        
        
        self.lp = lpFun
        self.step = step
        self.stopC = stopCondition
        self.tp = copy.deepcopy(tp0)
        
        self.M = M
        self.stateList = stateStore()
        self.lwts = lwtVector(2**(M+1))
        
        g0 = generated(q0)
        self.samples = np.zeros((len(g0),niter+1))
        self.samples[:,0] = g0
        
        diagnostics = []
        
        sc = step.getState()
        sc.firstEval(self.lp,q0,self.tp)
        
        # main MCMC iteration loop
        for it in range(niter):
            
            self.step.reset()
            
            sc.momentumRefresh(self.lp, self.tp)
            (sc,diagRow) = self.buildOrbit(sc)
            
            self.samples[:,it+1] = generated(sc.q)
            diagnostics.append(pd.concat([diagRow,step.diagnostics()]))
        
        
        self.diagnostics = pd.DataFrame(diagnostics)

q0 = np.random.normal(size=2)
lp = td.modFunnel


niter = 100000
wMC = NUTSampler(debug=False)
wMC.run(lp,q0,step=mc.adaptMCstepE(), tp0=mc.MCtuningPars(hMacro=0.2,delta=0.3),niter=niter)

wHMC = NUTSampler(debug=False)
wHMC.run(lp,q0,step=hmc.adaptHMCstepE(), tp0=hmc.HMCtuningPars(hMacro=0.15,delta=0.3),niter=niter)


print(np.mean(wMC.diagnostics.propBasic))
print(np.mean(wHMC.diagnostics.propBasic))

print(1000*az.ess(wMC.samples[0,:])/np.sum(wMC.diagnostics.gradEvals))
print(1000*az.ess(wHMC.samples[0,:])/np.sum(wHMC.diagnostics.gradEvals))
