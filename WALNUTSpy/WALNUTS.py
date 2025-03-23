import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import sys

from constants import __logZero,__wtSumThresh
import adaptiveIntegrators as ai
import P2quantile as P2



#--------------------------------------------------
# NUTS sequence of checks (precomputed)
#-------------------------------------------------



# sequence of stop-checks during NUTS iteration
checks = 0
k = 0
def subTreePlan(nleaf):
    global checks
    global k
    checks = np.zeros((nleaf-1,2),dtype=int)
    k = 0
    def Uturn(a,b):
        global k
        global checks
        checks[k,0] = a
        checks[k,1] = b
        k += 1

    def subUturn(a,b):
        if(a!=b):
            m = math.floor((a+b)/2)
            subUturn(a,m)
            subUturn(m+1,b)
            Uturn(a,b)
    subUturn(1,nleaf)
    return(checks)


                     

# utilities for storing states (avoiding storing the complete orbit)

class stateStore:
    def __init__(self,d,n):
        self.__stateStack = np.zeros((d,n))
        self.__stateStackId = np.zeros(n,dtype=int)
        self.__stateStackUsed = np.full(n,False,dtype=bool)
      
        
    def statePush(self,id_,state):
        full = True
        for i in range(len(self.__stateStackId)):
            if(not self.__stateStackUsed[i]):
                self. __stateStack[:,i] = state
                self.__stateStackId[i] = id_
                self.__stateStackUsed[i] = True
                full = False
                break
        if(full):
            sys.exit("stack full")
        
    def stateRead(self,id_):
        inds = np.where(self.__stateStackId==id_)[0]
        if(len(inds>0)):
           for i in range(len(inds)):
               if(self.__stateStackUsed[inds[i]]):
                   return(self.__stateStack[:,inds[i]])
        sys.exit("element not found for read in stack")
        
    def stateDeleteRange(self,from_,to_):
        self.__stateStackUsed[np.logical_and(self.__stateStackId >= from_,self.__stateStackId <= to_)] = False 
        

    def stateReset(self):
        self.__stateStackUsed = np.full(len(self.__stateStackUsed),False,dtype=bool)
        
    def dump(self):
        print("values")
        print(self.__stateStack)
        print("ids")
        print(self.__stateStackId)
        print("used")
        print(self.__stateStackUsed)


#--------------------------------------
# NUT stop condition
#--------------------------------------

def stopCondition(qm,vm,qp,vp):
    tmp = qp-qm
    return(sum(vp*tmp)<0.0 or sum(vm*tmp)<0.0)


#def stopCondition(qm,vm,qp,vp,thresh=-0.8):
#    tmp = qp-qm
#    nn = np.sqrt(sum(tmp*tmp))    
#    return(sum(vp*tmp)/(nn*np.sqrt(sum(vp*vp))) <thresh or sum(vm*tmp)/(nn*np.sqrt(sum(vm*vm)))<thresh)


#-----------------------------------------
# WALNUTS sampler
# 
#  revCheck:    True: stepsize adaptivity handled by deterministic reversibility checks
#               False: stepsize adaptivity handled by multinomial weigths correction
#
#-----------------------------------------

def WALNUTS(lpFun,
            q0, # intial position configuration
            generated=lambda q: q, # apply this function to q before storing samples
            integrator=ai.fixedLeapFrog,
            H0=0.2, #initial big step size / fixed big step size if adaptH=False
            stepSizeRandScale=0.2, # step size randomization scale
            delta0=0.05, # initial integrator tolerance
            multinomial=False, # if not, then biased progressive
            numIter=2000,
            warmupIter=1000,
            M=10, # number of NUTS iterations
            igrAux=ai.integratorAuxPar(),
            adaptH=True,
            adaptHtarget=0.8, # desired fraction of steps where crudest step size is accepted
            adaptDelta=True,
            adaptDeltaTarget=0.6,
            adaptDeltaQuantile=0.9,
            revCheck=False,  
            debug=False, #
            recordOrbitStats=False
            #oldGIST=False # retained for reference, NOT to be used 
            ): 
    
    #-----------------------------------
    # check that provided sampling plan makes sense
    #-----------------------------------
    if(revCheck):
        #requires one of the deterministic integrators
        if(integrator != ai.fixedLeapFrog and integrator != ai.adaptLeapFrogD):
            sys.exit("reversibility-check-based adaptivity requires a deterministic integrator")
    
    # if(oldGIST):
    #     print("WARNING: old GIST-based adaptivity not optimized, and should generally not be used")
    #     #requires one of the deterministic integrators
    #     if(integrator != fixedLeapFrog and integrator != adaptLeapFrogD):
    #         sys.exit("old GIST-based adaptivity requires a deterministic integrator")

    #-----------------------------------
    # setup
    #-----------------------------------
    
    
    # big step size
    H = H0
    
    if(adaptH):
        if(adaptHtarget<0.0 or adaptHtarget>1.0): sys.exit("bad adaptHtarget")
        igrConstQ = P2.P2quantile(1.0-adaptHtarget)
    
    # integrator tolerance
    delta = delta0
    if(adaptDelta):
        if(adaptDeltaTarget<0.0): sys.exit("bad adaptDeltaTarget")
        energyErrorInfFacs = np.zeros(warmupIter)
    
    
    
    # make tables for all of the sub-uturn checks that needs to be done during one iteration
    plans = []
    for i in range(0,M):
        plans.append(subTreePlan(2**i))

    


    # allocated memory for samples 
    d = q0.size
    g0 = generated(q0)
    dg = g0.size
    samples = np.zeros((dg,numIter+1))
    samples[:,0] = g0

    # allocate memomry for intermediate states
    states = stateStore(3*d,2*(M+1)+1)

    # memory for quantities stored for all states in orbit
    Hs = np.zeros(2**M)
    Ifs = np.zeros(2**M,dtype=int)
    Ibs = np.zeros(2**M,dtype=int)
    cs = np.zeros(2**M,dtype=int)
    lwts = np.zeros(2**M)

    # assorted stored states
    qc = q0 # current state

    # diagnositics info
    diagnostics = np.zeros((numIter,20))
    
    if(recordOrbitStats):
        orbitMin = np.zeros((dg,numIter))
        orbitMax = np.zeros((dg,numIter))

    #--------------------------------------
    # main MCMC iteration loop
    #--------------------------------------
    for iterN in range(1,numIter+1):
        if(iterN % 1000 == 0):
            print("iteration # ",iterN," H = ",H," delta = ",delta)

        #-------------------------------------
        # per iteration diagnostics info
        #-------------------------------------
        nevalF = 0
        nevalB = 0
        Lsub_ = 0
        L_ = 0
        orbitLen_ = 0.0
        orbitLenSam_ = 0.0
        Ndoubl = M
        
        warmup = iterN<=warmupIter
        
        #--------------------------------------
        # set up before starting integrating
        #--------------------------------------

        # integration directions: 1=backward, 0=forward
        B = np.floor(random.uniform(low=0.0,high=2.0,size=M)).astype(int)
        
        
        # how many backward steps could there possibly be
        nleft = sum(B*(2**np.arange(0,M)))

        # I0 is the index of the zeroth state in Hs, Ifs, Ibs
        I0 = nleft
        Ifs[I0] = 0 # not used
        Ibs[I0] = 0 # not used
        cs[I0] = 0 # not used
        lwts[I0] = 0.0

        # endpoints of accepted and proposed trajectory
        a = 0
        b = 0
        maxFint = 0
        maxBint = 0

        # full momentum refresh
        v = random.normal(size=d)

        # endpoints of current orbit (p=forward, m=backward)
        qp = qc
        qm = qc
        vp = v
        vm = v

        # current proposal
        qProp = qc
        qPropLast = qc

        # evaluate at current state
        f0,grad0 = lpFun(qc)

        # gradients at either endpoint
        gp = grad0
        gm = grad0

        # Hamiltonian at initial point
        Hs[I0] = -f0 + 0.5*sum(v**2)

        # index selection-related quantities
        multinomialLscale = Hs[I0]
        multinomialW = 1.0
        biasedProgWold = 0.0

        lwtSumb = 0.0
        lwtSumf = 0.0

        # reject orbit if numerical problems occur
        forcedReject = False
        
        # stop if both multinomial bias at both ends are zero
        bothEndsPassive = False
        stopCode = 0
        
        
        if(debug):
            qTrace = np.zeros((d,1))
            qTrace[:,0] = qc
            stateNum = np.array([0])
            
        if(recordOrbitStats):
            orbitMin[:,iterN-1] = generated(qc)
            orbitMax[:,iterN-1] = generated(qc)

        #------------------------------------------
        # NUT iteration loop
        #------------------------------------------
        for i in range(M):
            # integration direction
            xi = (-1)**B[i]
            # proposed new endpoints
            at = a + xi*(2**i)
            bt = b + xi*(2**i)
            

            # more bookkeeping
            expandFurther = True
            qPropLast = qProp

            if(i==0): # single first integration step required
                HLoc = random.uniform(low=H*(1-stepSizeRandScale),
                                      high=H*(1+stepSizeRandScale),size=1)[0]
                orbitLen_ += HLoc
                if(xi==1): # forward integration
                    intOut = integrator(qp,vp,gp,Hs[I0],HLoc,xi,lpFun,delta,igrAux)
                    qp = intOut.q
                    vp = intOut.v
                    gp = intOut.grad
                    nevalF += intOut.nEvalF
                    nevalB += intOut.nEvalB
                    Hs[I0+1] = -intOut.lp + 0.5*sum(vp*vp)
                    Ifs[I0+1] = intOut.If
                    Ibs[I0+1] = intOut.Ib
                    cs[I0+1] = intOut.c
                    lwts[I0+1] = intOut.lwt
                    if(warmup and adaptH): igrConstQ.push(np.log(intOut.igrConst))
                    maxFint = 1
                    if(not np.isfinite(Hs[I0+1])):
                        forcedReject = True
                        stopCode = 999
                        break
                    
                    if(not revCheck): lwtSumf = lwts[I0+1]
                    incr = np.exp(-Hs[I0+1]+multinomialLscale+lwtSumf)
                    multinomialW += incr

                    if(random.uniform()<min(1.0,incr)):
                        qProp = qp
                        Lsub_ = 1

                    if(debug):
                        tmp = np.zeros((d,1))
                        tmp[:,0] = qp
                        qTrace = np.hstack((qTrace,tmp))
                        stateNum = np.hstack((stateNum,1))
                    
                    if(recordOrbitStats):
                        orbitMin[:,iterN-1] = np.minimum(orbitMin[:,iterN-1],generated(qp))
                        orbitMax[:,iterN-1] = np.maximum(orbitMax[:,iterN-1],generated(qp))

                else:
                    intOut = integrator(qm,vm,gm,Hs[I0],HLoc,xi,lpFun,delta,igrAux)
                    qm = intOut.q
                    vm = intOut.v
                    gm = intOut.grad
                    nevalF += intOut.nEvalF
                    nevalB += intOut.nEvalB
                    Hs[I0-1] = -intOut.lp + 0.5*sum(vm*vm)
                    Ifs[I0-1] = intOut.If
                    Ibs[I0-1] = intOut.Ib
                    cs[I0-1] = intOut.c
                    lwts[I0-1] = intOut.lwt
                    if(warmup and adaptH): igrConstQ.push(np.log(intOut.igrConst))
                    maxBint = -1
                    if(not np.isfinite(Hs[I0-1])):
                        forcedReject = True
                        stopCode = 999
                        break
                    if(not revCheck): lwtSumb = lwts[I0-1]
                    incr = np.exp(-Hs[I0-1]+multinomialLscale + lwtSumb)
                    multinomialW += incr

                    if(random.uniform()<min(1.0,incr)):
                        qProp = qm
                        Lsub_ = -1

                    if(debug):
                        tmp = np.zeros((d,1))
                        tmp[:,0] = qm
                        qTrace = np.hstack((tmp,qTrace))
                        stateNum = np.hstack((-1,stateNum))
                    
                    if(recordOrbitStats):
                        orbitMin[:,iterN-1] = np.minimum(orbitMin[:,iterN-1],generated(qm))
                        orbitMax[:,iterN-1] = np.maximum(orbitMax[:,iterN-1],generated(qm))

                
                biasedProgWold = 1.0+incr
                
                
                if(revCheck):
                    # check if adaptive integration step is reversible
                    # if multinomial biasing is not used
                    if(np.abs(Ifs[I0+xi]-Ibs[I0+xi])>0):
                        
                        expandFurther = False
                        qProp = qc
                        L_ = 0
                        Ndoubl = 0
                        stopCode = -1
                        break
                
                # done building orbit, now check stop condition:
                if(stopCondition(qm,vm,qp,vp)):
                    expandFurther = False
                    qProp = qc
                    L_ = 0
                    Ndoubl = 0
                    stopCode = 1
                    break
                
            else: # more than a single integration step, these require sub-u-turn checks
                # work out which sub-u-turn-checks we are doing
                plan = 0
                if(xi==1):
                    plan = b + plans[i]
                else:
                    plan = a - plans[i]

                if(not multinomial):
                    multinomialW = 0.0

                for j in range(len(plan)): # loop over U-turn-checks
                    #print(plan[j,:])
                    if(abs(plan[j,0]-plan[j,1])==1): # new integration steps needed
                        HLoc1 = random.uniform(low=H*(1-stepSizeRandScale),
                                      high=H*(1+stepSizeRandScale),size=2)
                        
                        if(xi==-1): # backward integration
                            i1 = plan[j,0]
                            intOut = integrator(qm,vm,gm,Hs[I0+i1+1],HLoc1[0],xi,lpFun,delta,igrAux)
                            qm = intOut.q
                            vm = intOut.v
                            gm = intOut.grad
                            nevalF += intOut.nEvalF
                            nevalB += intOut.nEvalB
                            Hs[I0+i1] = -intOut.lp + 0.5*sum(vm*vm)
                            Ifs[I0+i1] = intOut.If
                            Ibs[I0+i1] = intOut.Ib
                            cs[I0+i1] = intOut.c
                            lwts[I0+i1] = intOut.lwt
                            if(warmup and adaptH): igrConstQ.push(np.log(intOut.igrConst))
                            maxBint = i1
                            if(not np.isfinite(Hs[I0+i1])):
                                forcedReject = True
                                stopCode = 999
                                break

                            if(revCheck and np.abs(Ifs[I0+i1]-Ibs[I0+i1])>0):
                                
                                expandFurther = False
                                stopCode = -2
                                break

                            # multinomial/progressive sampling
                            if(not revCheck): lwtSumb += lwts[I0+i1]
                            
                            incr = np.exp(-Hs[I0+i1]+multinomialLscale + lwtSumb)
                            multinomialW += incr
                            if(multinomialW > __wtSumThresh and random.uniform()<incr/multinomialW):
                                qProp = qm
                                Lsub_ = i1
                            # store state for future u-turn-checking
                            states.statePush(i1,np.concatenate([qm,vm,gm]))
                            orbitLen_ += HLoc1[0]

                            if(debug):
                                tmp = np.zeros((d,1))
                                tmp[:,0] = qm
                                qTrace = np.hstack((tmp,qTrace))
                                stateNum = np.hstack((i1,stateNum))
                            
                            if(recordOrbitStats):
                                orbitMin[:,iterN-1] = np.minimum(orbitMin[:,iterN-1],generated(qm))
                                orbitMax[:,iterN-1] = np.maximum(orbitMax[:,iterN-1],generated(qm))

                            qtmp = qm
                            vtmp = vm

                            # second integration step
                            i2 = plan[j,1]
                            intOut = integrator(qm,vm,gm,Hs[I0+i2+1],HLoc1[1],xi,lpFun,delta,igrAux)
                            qm = intOut.q
                            vm = intOut.v
                            gm = intOut.grad
                            nevalF += intOut.nEvalF
                            nevalB += intOut.nEvalB
                            Hs[I0+i2] = -intOut.lp + 0.5*sum(vm*vm)
                            Ifs[I0+i2] = intOut.If
                            Ibs[I0+i2] = intOut.Ib
                            cs[I0+i2] = intOut.c
                            lwts[I0+i2] = intOut.lwt
                            if(warmup and adaptH): igrConstQ.push(np.log(intOut.igrConst))
                            maxBint = i2
                            if(not np.isfinite(Hs[I0+i2])):
                                forcedReject = True
                                break
                            
                            if(revCheck and np.abs(Ifs[I0+i1]-Ibs[I0+i1])>0):
                                expandFurther = False
                                stopCode = -2
                                break
                            
                            # multinomial/progressive sampling
                            if(not revCheck): lwtSumb += lwts[I0+i2]
                            
                            incr = np.exp(-Hs[I0+i2]+multinomialLscale + lwtSumb)
                            multinomialW += incr
                            if(multinomialW > __wtSumThresh and random.uniform()<incr/multinomialW):
                                qProp = qm
                                Lsub_ = i2

                            # store state for future u-turn-checking
                            states.statePush(i2,np.concatenate([qm,vm,gm]))
                            orbitLen_ += HLoc1[1]

                            if(debug):
                                tmp = np.zeros((d,1))
                                tmp[:,0] = qm
                                qTrace = np.hstack((tmp,qTrace))
                                stateNum = np.hstack((i2,stateNum))
                            
                            if(recordOrbitStats):
                                orbitMin[:,iterN-1] = np.minimum(orbitMin[:,iterN-1],generated(qm))
                                orbitMax[:,iterN-1] = np.maximum(orbitMax[:,iterN-1],generated(qm))

                            # uturn check
                            if(stopCondition(qm,vm,qtmp,vtmp)):
                                if(debug):
                                    print("stop: ",i1," ",i2)
                                expandFurther = False
                                stopCode = 3
                                break

                            

                        else: # forward integration
                            i1 = plan[j,0]
                            intOut = integrator(qp,vp,gp,Hs[I0+i1-1],HLoc1[0],xi,lpFun,delta,igrAux)
                            qp = intOut.q
                            vp = intOut.v
                            gp = intOut.grad
                            nevalF += intOut.nEvalF
                            nevalB += intOut.nEvalB
                            Hs[I0+i1] = -intOut.lp + 0.5*sum(vp*vp)
                            Ifs[I0+i1] = intOut.If
                            Ibs[I0+i1] = intOut.Ib
                            cs[I0+i1] = intOut.c
                            lwts[I0+i1] = intOut.lwt
                            if(warmup and adaptH): igrConstQ.push(np.log(intOut.igrConst))
                            maxFint = i1
                            if(not np.isfinite(Hs[I0+i1])):
                                forcedReject = True
                                stopCode = 999
                                break

                            if(revCheck and np.abs(Ifs[I0+i1]-Ibs[I0+i1])>0):
                                
                                expandFurther = False
                                stopCode = -2
                                break

                            # multinomial/progressive sampling
                            if(not revCheck): lwtSumf += lwts[I0+i1]
                            incr = np.exp(-Hs[I0+i1]+multinomialLscale + lwtSumf)
                            multinomialW += incr
                            if(multinomialW > __wtSumThresh and random.uniform()<incr/multinomialW):
                                qProp = qp
                                Lsub_ = i1
                            # store state for future u-turn-checking
                            states.statePush(i1,np.concatenate([qp,vp,gp]))
                            orbitLen_ += HLoc1[0]

                            if(debug):
                                tmp = np.zeros((d,1))
                                tmp[:,0] = qp
                                qTrace = np.hstack((qTrace,tmp))
                                stateNum = np.hstack((stateNum,i1))
                                
                            if(recordOrbitStats):
                                orbitMin[:,iterN-1] = np.minimum(orbitMin[:,iterN-1],generated(qp))
                                orbitMax[:,iterN-1] = np.maximum(orbitMax[:,iterN-1],generated(qp))

                            qtmp = qp 
                            vtmp = vp

                            # second integration step
                            i2 = plan[j,1]
                            intOut = integrator(qp,vp,gp,Hs[I0+i2-1],HLoc1[1],xi,lpFun,delta,igrAux)
                            qp = intOut.q
                            vp = intOut.v
                            gp = intOut.grad
                            nevalF += intOut.nEvalF
                            nevalB += intOut.nEvalB
                            Hs[I0+i2] = -intOut.lp + 0.5*sum(vp*vp)
                            Ifs[I0+i2] = intOut.If
                            Ibs[I0+i2] = intOut.Ib
                            cs[I0+i2] = intOut.c
                            lwts[I0+i2] = intOut.lwt
                            if(warmup and adaptH): igrConstQ.push(np.log(intOut.igrConst))
                            maxFint = i2
                            if(not np.isfinite(Hs[I0+i2])):
                                forcedReject = True
                                break
                            
                            if(revCheck and np.abs(Ifs[I0+i2]-Ibs[I0+i2])>0):
                                
                                expandFurther = False
                                stopCode = -2
                                break
                            
                            # multinomial/progressive sampling
                            if(not revCheck): lwtSumf += lwts[I0+i2]
                            
                            incr = np.exp(-Hs[I0+i2]+multinomialLscale + lwtSumf)
                            multinomialW += incr
                            if(multinomialW > __wtSumThresh and random.uniform()<incr/multinomialW):
                                qProp = qp
                                Lsub_ = i2

                            # store state for future u-turn-checking
                            states.statePush(i2,np.concatenate([qp,vp,gp]))
                            orbitLen_ += HLoc1[1]

                            if(debug):
                                tmp = np.zeros((d,1))
                                tmp[:,0] = qp
                                qTrace = np.hstack((qTrace,tmp))
                                stateNum = np.hstack((stateNum,i2))
                            
                            if(recordOrbitStats):
                                orbitMin[:,iterN-1] = np.minimum(orbitMin[:,iterN-1],generated(qp))
                                orbitMax[:,iterN-1] = np.maximum(orbitMax[:,iterN-1],generated(qp))

                            if(stopCondition(qtmp,vtmp,qp,vp)):
                                if(debug):
                                    print("stop: ",i1," ",i2)
                                expandFurther = False
                                stopCode = 2
                                break
                        # done forward integration 
                    else: # no new integration steps needed, only U-turn checks
                        # delete states not needed further
                        
                        im = min(plan[j,:])
                        ip = max(plan[j,:])
                        #print(" not integrating ",im," ",ip)
                        states.stateDeleteRange(im+1,ip-1)
                        statep = states.stateRead(ip)
                        statem = states.stateRead(im)

                        if(stopCondition(statem[0:d],
                                         statem[d:(2*d)],
                                         statep[0:d],
                                         statep[d:(2*d)])):
                            if(debug):
                                print("stop B ",im," ",ip)
                            expandFurther = False
                            stopCode = 3
                            break
                # done loop over j (16)

                if(forcedReject):
                    print("numerical problems")
                    break

                if(not multinomial and expandFurther):
                    # note: reject (and not accept) of proposed state from last doubling
                    if(not random.uniform() < min(1.0,multinomialW/biasedProgWold)):
                        Lsub_ = L_
                        qProp = qPropLast
                    biasedProgWold += multinomialW
            
            # done i>0 (12)                         
            
            if(expandFurther):          
                joinedCrit = stopCondition(qm,vm,qp,vp)
                # stop simulation if multinomial weights at either end are effectivly zero
                bothEndsPassive = lwtSumb < __logZero+1.0 and lwtSumf < __logZero+1.0
                if(joinedCrit or bothEndsPassive):
                    if(debug):
                        print("stop global")
                    if(joinedCrit):
                        stopCode = 4
                    else:
                        stopCode = -4
                    
                
                    #qProp = qPropLast
                    #Ndoubl = i
                    Ndoubl = i+1
                    break
            else:
                if(debug):
                    print("stop local")
                qProp = qPropLast
                Ndoubl = i
                break
            # from now on, it is clear that a new doubling will be attempted        
            L_ = Lsub_
            orbitLenSam_ = orbitLen_
                          
            a = min(a,at)
            b = max(b,bt)
            states.stateReset()
            
            
        # done NUTS loop
        
        if(debug):
            print(qTrace)
            plt.plot(qTrace[0,:],qTrace[1,:])
            plt.show()


        # if(oldGIST):
        #     # note: oldGIST version not optimized, only retained as reference
        #     # this method adds a GIST-style accept-reject step to account for
        #     # integrator adaptivity. Only implemented for the deterministic adaptive sampler,
        #     # hence check for that first:
        #     #print(Ifs[(I0+maxBint):(I0+maxFint)])
        #     #print(cs[(I0+maxBint):(I0+maxFint)])
        #     if(np.any(Ifs[(I0+maxBint):(I0+maxFint)] != cs[(I0+maxBint):(I0+maxFint)])):
        #         sys.exit("using oldGIST with randomized integrator, not implemented")
        #     if(not np.any(Ifs[(I0+maxBint):(I0+maxFint)] != Ibs[(I0+maxBint):(I0+maxFint)])):
        #        qc = qProp
        #     else:
        #        print("GIST reject")
        # else:
        #     # no GIST accept/reject here - multinomially sampled new state
        #     # accepted with prob 1
        #     qc = qProp 
        qc = qProp    

        #print(Ifs[(I0+maxBint):(I0+maxFint)])
        #print(Ibs[(I0+maxBint):(I0+maxFint)])

        #-------------------------------------
        # store samples and diagnostics info
        #-------------------------------------

        if(maxBint<0 and maxFint>0):
            usedSteps = np.r_[maxBint:0,1:(maxFint+1)]
        elif(maxBint<0):
            usedSteps = np.r_[maxBint:0]
        else:
            usedSteps = np.r_[1:(maxFint+1)]
                

        orbitEnergyError = np.max(Hs[I0+usedSteps]) - np.min(Hs[I0+usedSteps])

        diagnostics[iterN-1,0] = L_
        diagnostics[iterN-1,1] = Ndoubl
        diagnostics[iterN-1,2] = orbitLen_
        diagnostics[iterN-1,3] = orbitLenSam_
        diagnostics[iterN-1,4] = maxFint
        diagnostics[iterN-1,5] = maxBint
        diagnostics[iterN-1,6] = nevalF
        diagnostics[iterN-1,7] = nevalB
        diagnostics[iterN-1,8] = np.min(Ifs[I0+usedSteps])
        diagnostics[iterN-1,9] = np.max(Ifs[I0+usedSteps])
        diagnostics[iterN-1,10] = np.min(lwts[I0+usedSteps])
        diagnostics[iterN-1,11] = np.max(lwts[I0+usedSteps])
        diagnostics[iterN-1,12] = 1.0*bothEndsPassive
        diagnostics[iterN-1,13] = 1.0*(lwtSumb < __logZero+1.0 or lwtSumf < __logZero+1.0)
        diagnostics[iterN-1,14] = np.mean(Ifs[I0+usedSteps]!=Ibs[I0+usedSteps])
        diagnostics[iterN-1,15] = H
        diagnostics[iterN-1,16] = np.mean(Ifs[I0+usedSteps]==0)
        diagnostics[iterN-1,17] = orbitEnergyError
        diagnostics[iterN-1,18] = delta
        diagnostics[iterN-1,19] = 1.0*stopCode
        
        
        samples[:,iterN] = generated(qc)
        
        #-------------------------------------
        # tuning parameter adaptation
        #-------------------------------------
        
        if(warmup):
            
            # tuning of local error threshold delta
            if(adaptDelta): energyErrorInfFacs[iterN-1] = orbitEnergyError/delta
            if(adaptDelta and iterN>10):
                delta = adaptDeltaTarget/np.quantile(energyErrorInfFacs[0:iterN],
                                                     adaptDeltaQuantile)
                
            
            # tuning of big step size H
            if(adaptH and igrConstQ.npush>10):
                H = ((delta)**(1.0/3.0))*np.exp(igrConstQ.quantile()) 
            
            
        

    # done main iteration loop



    


    if(recordOrbitStats):
        return(samples,diagnostics,orbitMin,orbitMax)
    else:
        return(samples,diagnostics)







