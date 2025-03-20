#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 10:22:43 2025

@author: torekleppe
"""
from constants import __logZero
import numpy as np
import sys

# -------------------------------------
# integrators
# -------------------------------------

# common return object
class integratorReturn:
    def __init__(self,q1,v1,lp1,grad1,nEvalF,nEvalB,If,Ib,c,lwt,igrConst):
        self.q = q1
        self.v = v1
        self.lp = lp1
        self.grad = grad1
        self.nEvalF = nEvalF
        self.nEvalB = nEvalB
        self.If = If
        self.Ib = Ib
        self.c = c
        self.lwt = lwt
        self.igrConst = igrConst
        
    def __str__(self):
        return "dim: " + str(len(self.q)) + " If: " + str(self.If) + " Ib: " + str(self.Ib) + " c: " + str(self.c)


# class used for passing assorted tuning parameters to the integrators
class integratorAuxPar:
    def __init__(self,maxC=10,R2Pprob0=2.0/3.0,maxFPiter=30,FPtol=1.0e-9):
        self.maxC=maxC
        self.R2Pprob0=R2Pprob0
        self.maxFPiter=maxFPiter
        self.FPtol=FPtol
    
        

# basic leapfrog (used along with WALNUTS below results in NUTS)
def fixedLeapFrog(q,v,g,Ham0,h,xi,lpFun,delta,auxPar):
    vh = xi*v + 0.5*h*g
    qq = q + h*vh
    fnew,gnew = lpFun(qq)
    vv = vh + 0.5*h*gnew

    H1 = -fnew + 0.5*sum(vv*vv)

    return integratorReturn(qq,xi*vv,fnew,gnew,1,0,0,0,0,
                            0.0,
                            h*(abs(Ham0-H1)**(-1.0/3.0)))




# adaptive Leap Frog with Deterministic choice of precsion parameter
def adaptLeapFrogD(q,v,g,Ham0,h,xi,lpFun,delta,auxPar):
    
    nEvalF = 0
    If = auxPar.maxC
    for c in range(0,auxPar.maxC+1):
        nstep = 2**c
        hh = h/nstep
        qq = q
        vv = xi*v
        gg = g
        Hams = np.zeros(nstep+1)
        Hams[0] = Ham0
        
        for i in range(1,nstep+1):
            vh = vv + 0.5*hh*gg
            qq = qq + hh*vh
            fnew,gg = lpFun(qq)
            nEvalF += 1
            vv = vh + 0.5*hh*gg
            Hams[i] = -fnew + 0.5*sum(vv*vv)
        
        #maxErr = np.max(Hams) - np.min(Hams)
        maxErr = abs(Hams[0]-Hams[-1])
        #print(Hams)
        #print(maxErr)
        
        
        if(all(np.isfinite(Hams)) and maxErr < delta):
            If = c
            break
    
    qOut = qq
    vOut = vv
    fOut = fnew
    gOut = gg
    
    igrConst = hh*(np.max(np.abs(np.diff(Hams)))**(-1.0/3.0))
    
    
    Ib = If
    nEvalB = 0

    if(If>0):
    
        
        H0b = Hams[-1]
        for c in range(0,If):
            nstep = 2**c
            hh = h/nstep
            qq = qOut
            vv = -vOut
            gg = gOut
            Hams = np.zeros(nstep+1)
            Hams[0] = H0b
            
            for i in range(1,nstep+1):
                vh = vv + 0.5*hh*gg
                qq = qq + hh*vh
                fnew,gg = lpFun(qq)
                nEvalB += 1
                vv = vh + 0.5*hh*gg
                Hams[i] = -fnew + 0.5*sum(vv*vv)
            
            #maxErr = np.max(Hams) - np.min(Hams)
            maxErr = abs(Hams[0]-Hams[-1])
            if(all(np.isfinite(Hams)) and maxErr < delta):
                Ib = c
                break    
    
    
    return integratorReturn(qOut,xi*vOut,fOut,gOut,nEvalF,nEvalB,If,Ib,If,
                            (If!=Ib)*__logZero,
                            igrConst)
    

def adaptLeapFrogR2P(q,v,g,Ham0,h,xi,lpFun,delta,auxPar):
    
    nEvalF = 0
    If = auxPar.maxC
    for c in range(0,auxPar.maxC+1):
        nstep = 2**c
        hh = h/nstep
        qq = q
        vv = xi*v
        gg = g
        Hams = np.zeros(nstep+1)
        Hams[0] = Ham0
        
        for i in range(1,nstep+1):
            vh = vv + 0.5*hh*gg
            qq = qq + hh*vh
            fnew,gg = lpFun(qq)
            nEvalF += 1
            vv = vh + 0.5*hh*gg
            Hams[i] = -fnew + 0.5*sum(vv*vv)
        
        #maxErr = np.max(Hams) - np.min(Hams)
        maxErr = abs(Hams[0]-Hams[-1])
        #print(Hams)
        #print(maxErr)
        
        
        if(all(np.isfinite(Hams)) and maxErr < delta):
            If = c
            break
    
    
    if(np.random.uniform()<auxPar.R2Pprob0):
        # simulation occur at minimal accepted precision
        qOut = qq
        vOut = vv
        fOut = fnew
        gOut = gg
        cSim = If
        igrConst = hh*(np.max(np.abs(np.diff(Hams)))**(-1.0/3.0))
    else:
        # simulation occur at minimal + 1 
        c = If+1
        nstep = 2**c
        hh = h/nstep
        qq = q
        vv = xi*v
        gg = g
        Hams = np.zeros(nstep+1)
        Hams[0] = Ham0
        
        for i in range(1,nstep+1):
            vh = vv + 0.5*hh*gg
            qq = qq + hh*vh
            fnew,gg = lpFun(qq)
            nEvalF += 1
            vv = vh + 0.5*hh*gg
            Hams[i] = -fnew + 0.5*sum(vv*vv)
            
        qOut = qq
        vOut = vv
        fOut = fnew
        gOut = gg
        cSim = If+1
        igrConst = hh*(np.max(np.abs(np.diff(Hams)))**(-1.0/3.0))
        

    # done forward simulation pass, now do backward simulations
    nEvalB = 0
    
    if(cSim==If):
        maxTry = If-1
        Ib = If
        lwtf = np.log(auxPar.R2Pprob0)
    else:
        maxTry = auxPar.maxC
        Ib = auxPar.maxC
        lwtf = np.log(1.0-auxPar.R2Pprob0)
        
        
    if(maxTry>=0):
    
        
        H0b = Hams[-1]
        for c in range(0,maxTry+1):
            nstep = 2**c
            hh = h/nstep
            qq = qOut
            vv = -vOut
            gg = gOut
            Hams = np.zeros(nstep+1)
            Hams[0] = H0b
            
            for i in range(1,nstep+1):
                vh = vv + 0.5*hh*gg
                qq = qq + hh*vh
                fnew,gg = lpFun(qq)
                nEvalB += 1
                vv = vh + 0.5*hh*gg
                Hams[i] = -fnew + 0.5*sum(vv*vv)
            
            maxErr = abs(Hams[0]-Hams[-1])
            if(all(np.isfinite(Hams)) and maxErr < delta):
                Ib = c
                break    
    
    # done backward simulation pass, now work out backward probability
    lwtb = __logZero
    if(cSim==Ib):
        lwtb = np.log(auxPar.R2Pprob0)
    elif(cSim==Ib+1):
        lwtb = np.log(1.0-auxPar.R2Pprob0)    
        
    return integratorReturn(qOut,xi*vOut,fOut,gOut,nEvalF,nEvalB,If,Ib,cSim,
                            lwtb-lwtf,
                            igrConst)


def adaptImplicitMidpoint(q,v,g,Ham0,h,xi,lpFun,delta,auxPar):
    
    nEvalF = 0
    If = auxPar.maxC
    for c in range(0,auxPar.maxC+1):
        
        nstep = 2**c
        hh = h/nstep
        qq = q
        vv = xi*v
        gg = g
        Hams = np.zeros(nstep+1)
        Hams[0] = Ham0
        numCompleted = 0
        for i in range(1,nstep+1):
            # initial guess based on leap frog
            qt = qq + hh*(vv+0.5*hh*gg)
            
            # controls for fixed point iterations
            converged = False
            oldMaxErr = 1.0e100
            
            for iter in range(1,auxPar.maxFPiter+1):
                mpq = 0.5*(qt+qq)
                fmp,gmp = lpFun(mpq)
                nEvalF += 1
                qtNew = qq + hh*vv + (0.5*hh*hh)*gmp
                #print(qt)
                #print(qtNew)
                maxErr = np.max(np.abs(qtNew-qt))
                qt = qtNew
                #print(maxErr)
                if(maxErr<auxPar.FPtol): 
                    converged = True
                    break 
                
                
                if(maxErr>1.1*oldMaxErr):
                    #print("divergent iteration")
                    break
                oldMaxErr = maxErr

            if(not converged): 
                #print("step not converged")
                break
            
            # step used and evaluation at mesh times
            mpq = 0.5*(qt+qq)
            fmp,gmp = lpFun(mpq)
            nEvalF += 1
            qq = qq + hh*vv + (0.5*hh*hh)*gmp
            vv = vv + hh*gmp 
            
            fnew,gg = lpFun(qq)
            nEvalF += 1
            Hams[i] = -fnew + 0.5*sum(vv*vv)
            numCompleted +=1
            
        #print(Hams)
        maxHErr = abs(Hams[0]-Hams[-1])
        if(all(np.isfinite(Hams)) and maxHErr < delta and numCompleted==nstep):
            If = c
            break 
    
    if(not converged):
        print("numerical problems in adaptImplicitMidpoint, consider increasing maxC")
        sys.exit()
 
    
    
    qOut = qq
    vOut = vv
    fOut = fnew
    gOut = gg
    
    
    
    igrConst = hh*(np.max(np.abs(np.diff(Hams)))**(-1.0/3.0))
    
    
    Ib = If
    nEvalB = 0

    if(If>0):
        
        Hb0 = -fOut + 0.5*sum(vOut*vOut)
        for c in range(0,If):
            nstep = 2**c
            hh = h/nstep
            qq = qOut
            vv = -vOut
            gg = gOut
            Hams = np.zeros(nstep+1)
            Hams[0] = Hb0
            numCompleted = 0
            for i in range(1,nstep+1):
                # initial guess based on leap frog
                qt = qq + hh*(vv+0.5*hh*gg)
                
                # controls for fixed point iterations
                converged = False
                oldMaxErr = 1.0e100
                
                for iter in range(1,auxPar.maxFPiter+1):
                    mpq = 0.5*(qt+qq)
                    fmp,gmp = lpFun(mpq)
                    nEvalB += 1
                    qtNew = qq + hh*vv + (0.5*hh*hh)*gmp
                    maxErr = np.max(np.abs(qtNew-qt))
                    qt = qtNew
                    #print(maxErr)
                    if(maxErr<auxPar.FPtol): 
                        converged = True
                        break 
                    if(maxErr>1.1*oldMaxErr):
                        #print("divergent iteration")
                        break
                    oldMaxErr = maxErr
                    

                if(not converged): 
                    #print("backward step not converged")
                    break
                
                # step used 
                mpq = 0.5*(qt+qq)
                fmp,gmp = lpFun(mpq)
                nEvalB += 1
                qq = qq + hh*vv + (0.5*hh*hh)*gmp
                vv = vv + hh*gmp 
                
                fnew,gg = lpFun(qq)
                nEvalB += 1
                Hams[i] = -fnew + 0.5*sum(vv*vv)
                
            #print(Hams)
            maxHErr = abs(Hams[0]-Hams[-1])
            if(all(np.isfinite(Hams)) and maxHErr < delta and numCompleted==nstep):
                Ib = c
                break           
            
    #print("If : ",If," Ib : ",Ib)
           
    return integratorReturn(qOut,xi*vOut,fOut,gOut,nEvalF,nEvalB,If,Ib,If,
                            (If!=Ib)*__logZero,
                            igrConst)

# import targetDistr as td

# lpFun = td.modFunnel
# q0 = np.array([1.0,2.0])
# v0 = np.array([-1.0,0.5])
# f,g = lpFun(q0)

# Ham0 = -f + 0.5*sum(v0*v0)
# h = 0.5
# xi = 1
# delta=0.05
# auxPar = integratorAuxPar()

# out = adaptImplicitMidpoint(q0, v0, g, Ham0, h, xi, lpFun, delta, auxPar)

# Hb = -out.lp + 0.5*sum(out.v*out.v)

# outb = adaptImplicitMidpoint(out.q, -out.v, out.grad, Hb, h, xi, lpFun, delta, auxPar)



# outLF = adaptLeapFrogD(q0, v0, g, Ham0, h, xi, lpFun, delta, auxPar)
        



