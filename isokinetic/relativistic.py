import targets as td
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import copy
import pandas as pd
from abc import ABC
from relativisticKineticDistribution import globalRelativisticKineticDistribution

LOG_ZERO_THRESH = -700.0



class RELunivarMomentum:
    
    
    def logpdf(self,x,c=4.0):
  
        return(sp.stats.genhyperbolic.logpdf(x,p=1,a=c**2,b=0.0,scale=c))
    
    def rng(self,size=1,c=4.0):
        return(sp.stats.genhyperbolic.rvs(p=1,a=c**2,b=0.0,scale=c,size=size))
        
    
    def testPars(self,c=4.0):
        xx = np.linspace(start=-4.0,stop=4.0,num=1000)
        ff = self.logpdf(xx,c)
        plt.plot(xx,ff-(-c**2*np.sqrt(1.0 + xx**2/(c**2))+c**2))
        
        

class RELtuningPars:
    
    def __init__(self, c=1.0, hMacro=0.1, delta=0.1, Cmin=0,Cmax=16):
        self.c = c
        self.cSq = c**2
        self.hMacro = hMacro
        self.delta = delta
        self.Cmin = Cmin
        self.Cmax = Cmax





class RELstate:
    
    def __init__(self):
        self.Ham = np.nan
        self.simMod = globalRelativisticKineticDistribution
    
    def firstEval(self,lpFun,q,tp):
        [f,g] = lpFun(q)
        self.q = q
        self.f = f
        self.g = g
        self.p = np.repeat(np.nan,len(q))
        self.Ham = np.nan
        
        
    def momentumRefresh(self,lpFun,tp):
        self.simMod.checkCurrent(d=len(self.q),c=tp.c,ngrid=2048)
        self.p = self.simMod.sample()
        self.vfac = np.sqrt(1.0+np.dot(self.p,self.p)/(tp.cSq))
        self.Ham = -self.f + tp.cSq*self.vfac
        
        
    def velocity(self):
        return((1.0/self.vfac)*self.p)
        
    
    def momentumFlip(self):
        self.p = -self.p
    
    def __str__(self):
        return ("RELState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("RELState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))


class RELIstate:
    
    def __init__(self):
        self.Ham = np.nan
    
    def firstEval(self,lpFun,q,tp):
        [f,g] = lpFun(q)
        self.q = q
        self.f = f
        self.g = g
        self.p = np.repeat(np.nan,len(q))
        self.v = np.repeat(np.nan,len(q))
        self.Ham = np.nan
        
        
    def momentumRefresh(self,lpFun,tp):
        
        self.p = RELunivarMomentum().rng(size=len(self.q),c=tp.c)
        self.v = self.p/np.sqrt(1.0 + (self.p/tp.c)**2)
        self.Ham = -self.f + (tp.cSq)*np.sum(np.sqrt(1.0 + (self.p/tp.c)**2))
        
        
    def velocity(self):
        
        return(self.v)
        
    
    def momentumFlip(self):
        self.p = -self.p
        self.v = -self.v
    
    def __str__(self):
        return ("RELIState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))

    def __repr__(self):
        return ("RELIState class:\nq: " + str(self.q) + "\np: " + str(self.p) + "\nHamiltonian: " + str(self.Ham))



class RELintegrators(ABC):

    # regular Leap frog integrator
    def LFp(self,s,lpFun,c,h,nstep=1):
        g = s.g
        q = s.q
        p = s.p
        for i in range(nstep):
            ph = p + 0.5*h*g
            M = np.sqrt(1.0 + np.dot(ph,ph)/(c**2))
            q = q + h*ph/M
            [f,g] = lpFun(q)
            p = ph + 0.5*h*g
        
        sOut = RELstate()
        sOut.q = q
        sOut.p = p
        sOut.f = f
        sOut.g = g
        sOut.vfac = np.sqrt(1.0+np.dot(p,p)/(c**2))
        sOut.Ham = -f + (c**2)*sOut.vfac
        return(sOut)
    
    
    def LFpI(self,s,lpFun,c,h,nstep=1):
        g = s.g
        q = s.q
        p = s.p
        for i in range(nstep):
            ph = p + 0.5*h*g
            v = ph/np.sqrt((ph/c)**2 + 1.0)
            q = q + h*v
            [f,g] = lpFun(q)
            p = ph + 0.5*h*g
        
        sOut = RELIstate()
        sOut.q = q
        sOut.p = p
        sOut.f = f
        sOut.g = g
        sOut.v = p/np.sqrt((p/c)**2 + 1.0)
        sOut.Ham = -f + (c**2)*np.sum(np.sqrt(1.0 + (p/c)**2))
        return(sOut)
    
    def LFvI(self,s,lpFun,c,h,nstep=1):
        g = s.g
        q = s.q
        p = s.p
        v = p/np.sqrt(1.0+(p/c)**2)
        w = np.arctanh(v/c)
        
        for i in range(nstep):
            wh = w
            badIt = True
            for it in range(10):
                dev = w + 0.5*h/(c*np.cosh(wh))*g - wh
                ddev = -1.0 - 0.5*(h/c)*g*np.sinh(wh)/(np.cosh(wh)**2)
                whNew = wh - dev/ddev
                
                err = np.max(np.abs(whNew-wh))
                if(err < 1.0e-14*max(1.0,np.max(np.abs(wh)))):
                    badIt = False
                    break
                #print(err)
                wh = whNew
            
            if(badIt):
                return(RELIstate())
            
            q = q + h*c*np.tanh(wh)
            [f,g] = lpFun(q)
            w = wh + 0.5*h/(c*np.cosh(wh))*g
            
        sOut = RELIstate()
        sOut.q = q
        sOut.p = 0.5*c*(np.exp(2.0*w)-1.0)/np.exp(w)
        sOut.f = f
        sOut.g = g
        sOut.v = sOut.p/np.sqrt((sOut.p/c)**2 + 1.0)
        sOut.Ham = -f + (c**2)*np.sum(np.sqrt(1.0 + (sOut.p/c)**2))
        return(sOut)        
            
            
    
    
    
    # generalized leapfrog in v-variable
    def LFv(self,lpFun,q,p,h):
        
        v = (1.0/np.sqrt(1.0+np.dot(p,p)/(c**2)))*p
        
        [f,g] = lpFun(q)
        
        vh = v
        
        for i in range(100):
            vtv = np.dot(vh,vh)
            print(c**2-vtv)
            vhNew = v + 0.5*h*np.sqrt((c**2-vtv)/(c**2))*(g - (np.dot(g,vh)/(c**2))*vh)
            if(np.dot(vhNew,vhNew)>c**2):
                Delta = vhNew-vh
                alph = 0.5
                for j in range(20):
                    vhNew = vh + alph*Delta
                    print(alph)
                    print(c**2 - np.dot(vhNew,vhNew))
                    if(np.dot(vhNew,vhNew)<c**2):
                        break
                    alph *= 0.5
            err = np.max(np.abs(vhNew-vh))
            print(err)
            vh = vhNew
            if(err<1.0e-14*max(1.0,np.max(np.abs(vh)))):
                break
            
        
        qn = q + h*vh
        [f,g] = lpFun(qn)
        print(vh)
        vtv = np.dot(vh,vh)
        vn = vh + 0.5*h*np.sqrt((c**2-vtv)/(c**2))*(g - (np.dot(g,vh)/(c**2))*vh)
        
        vntvn = np.dot(vn,vn)
        pn = np.sqrt(1.0 + vntvn/(c**2-vntvn))*vn
        print("Eerr: " + str(Ham(np.concat((q,p)))-Ham(np.concat((qn,pn)))))
        return(qn,pn)


class adaptRELstepE(RELintegrators):
    
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
        return(RELstate())
    
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
            
            sOut = self.LFp(s=s,lpFun=lpFun,c=tp.c,h=tp.hMacro/nstep,nstep=nstep)
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
                
                sTmp = self.LFp(sB,lpFun,c=tp.c,h=tp.hMacro/nstep,nstep=nstep)
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
        
        
        
class adaptRELIstepE(RELintegrators):
    
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
        return(RELIstate())
    
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
            
            sOut = self.LFpI(s=s,lpFun=lpFun,c=tp.c,h=tp.hMacro/nstep,nstep=nstep)
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
                
                sTmp = self.LFpI(sB,lpFun,c=tp.c,h=tp.hMacro/nstep,nstep=nstep)
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
        
     
        
# class foo(RELintegrators):
#     def __call__(self,s,lpFun,tp,h=0.1,nstep=1):
#         return(self.LFvI(s=s,lpFun=lpFun,c=tp.c,h=h,nstep=nstep))



# tp = RELtuningPars(c=4.0,delta=1.0e-3)
# f = RELIstate()
# f.firstEval(td.stdGauss,np.array([-1.0,0.2]),tp)
# f.momentumRefresh(td.stdGauss, tp)

# s = foo()


# f1 = s(f,td.stdGauss,tp,h=0.5)

# f1.momentumFlip()

# f0b = s(f1,td.stdGauss,tp,h=0.5)











