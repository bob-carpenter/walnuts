
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


alpha = 0.5
m = 1.0*np.ones(2)


def stdGauss(q,v=None):
    if(v is None):
        return -0.5*sum(q**2),-q 
    else:
        return -0.5*sum(q**2),-q,-v 
    
def funnel1(q,v=None):
    f = -(1.0/18.0)*q[0]**2 -0.5*q[1]**2*np.exp(-q[0]) - 0.5*q[0]
    g = np.array([-q[0]/9.0 + 0.5*q[1]**2*np.exp(-q[0]) - 0.5,
                  -q[1]*np.exp(-q[0])])
    if(v is None):
        return f,g
    else:
        H = np.zeros((2,2))
        H[0,0] = -1.0/9.0 - 0.5*q[1]**2*np.exp(-q[0])
        H[1,0] = q[1]*np.exp(-q[0])
        H[0,1] = H[1,0]
        H[1,1] = -np.exp(-q[0])
        return f,g,np.matmul(H,v)


def corrGauss(q,v=None):
    rho = 0.95
    f = -0.5*(q[0]**2 + q[1]**2 - 2.0*rho*q[0]*q[1])/(1.0-rho**2) 
    g = (1.0/(1.0-rho**2))*np.array([-q[0]+rho*q[1],-q[1]+rho*q[0]])
    H = np.zeros((2,2))
    H[0,0] = -1.0/(1.0-rho**2)
    H[0,1] = rho/(1.0-rho**2)
    H[1,0] = H[0,1]
    H[1,1] = H[0,0]
    if(v is None):
        return f,g
    else:
        return f,g,np.matmul(H,v)
    





class state:
    
    
    def __init__(self,
                 q=None,
                 p=None,
                 f=None,
                 g=None,
                 r=None,
                 L=None,
                 v=None,
                 Hr=None,
                 Hv=None,
                 Ham=None):
        self.q = q
        self.p = p
        self.f = f
        self.g = g
        self.r = r
        self.L = L
        self.v = v
        self.Hr = Hr
        self.Hv = Hv
        self.Ham = Ham
        
    
    def evalFirst(self,lpFun,q,p):
        
        self.q = q
        self.p = p
        [f,g] = lpFun(q)
        self.f = f
        self.g = g
        Gmat = alpha**2*np.outer(g, g) + np.diag(m)
        self.r = g/m
        self.L = 1.0 + alpha**2*np.dot(self.r,g)
        self.v = p/m - (alpha**2/self.L)*np.dot(self.r,p)*self.r
        [f,g,Hr] = lpFun(q,v=self.r)
        self.Hr = Hr
        [f,g,Hv] = lpFun(q,v=self.v)
        self.Hv = Hv
        self.Ham = -f + 0.5*np.log(self.L) + 0.5*np.dot(self.v,Gmat @ self.v)
    
    def momentumFlip(self):
        self.p = -self.p
        self.v = -self.v
        self.Hv = -self.Hv
        
def mongeIntAdapt(s,lpFun,Tmax):
    def ode(t,y):
        d = len(y)//2
        q = y[0:d]
        p = y[d:(2*d)]
        [f,g] = lpFun(q)
        r = g/m
        L = 1.0 + alpha**2*np.dot(r,g)
        v = p/m - (alpha**2/L)*np.dot(r,p)*r
        [_,_,Hr] = lpFun(q,v=r)
        phiGrad = -g + (alpha**2/L)*Hr
        [_,_,Hv] = lpFun(q,v=v)
        pForce = phiGrad - alpha**2*np.dot(v,g)*Hv
        return(np.concat((v,-pForce)))        
        
    
    out = sp.integrate.solve_ivp(ode,(0.0,Tmax),y0=np.concat((s.q,s.p)),
                                 rtol=1.0e-10,
                                 atol=1.0e-10)
    return(out)
    

    
# Lan et al Integrator for (q,p) (and not q,v)
def mongeInt(s,lpFun,h=0.3,nstep=1):
    
    q = s.q
    r = s.r
    g = s.g
    v = s.v
    L = s.L
    phiGrad = -s.g + (alpha**2/L)*s.Hr
    GlogDet0 = np.log(L) 
    Hv = s.Hv
    
    
    
    logJac = -GlogDet0
    
    qs = np.zeros((len(q),nstep+1))
    qs[:,0] = q
    
    for i in range(nstep):
        t1 = v - 0.5*h*(phiGrad/m - (alpha**2/L)*np.dot(r,phiGrad)*r)
        det0 = 1.0 + 0.5*h*(alpha**2/L)*np.dot(r,Hv)
        logJac -= np.log(det0)
        
        vh = t1 - 0.5*h*((alpha**2/L)/det0)*np.dot(Hv,t1)*r
        
        
        [_,_,Hvh0] = lpFun(q,v=vh) 
        
        det1 = 1.0 - 0.5*h*(alpha**2/L)*np.dot(r,Hvh0)
        logJac += np.log(det1)
        
        q = q + h*vh
        
        [f,g,Hvh1] = lpFun(q,v=vh)
        
        r = g/m
        L = 1.0 + alpha**2*np.dot(r,g)
        
        det2 = 1.0 + 0.5*h*(alpha**2/L)*np.dot(r,Hvh1)
        logJac -= np.log(det2)
        
        [_,_,Hr] = lpFun(q,v=r)
        phiGrad = -g + (alpha**2/L)*Hr
        GlogDet1 = np.log(L) 
        t1 = vh - 0.5*h*(phiGrad/m - (alpha**2/L)*np.dot(r,phiGrad)*r)
        
        v = t1 - 0.5*h*((alpha**2/L)/det2)*np.dot(Hvh1,t1)*r
        [_,_,Hv] = lpFun(q,v=v)
        
        det3 = 1.0 - 0.5*h*(alpha**2/L)*np.dot(r,Hv)
        logJac += np.log(det3)
        
        p = m*v + alpha**2*np.dot(g,v)*g
        Ginvp = p/m - (alpha**2/L)*np.dot(r,p)*r
        Ham = -f + 0.5*np.log(L) + 0.5*np.dot(p,Ginvp) 
        
        qs[:,i+1] = q
        
    logJac += GlogDet1
    plt.plot(qs[0,:],qs[1,:])
    out = state(q=q,p=p,f=f,g=g,r=r,L=L,v=v,Hr=Hr,Hv=Hv,Ham=Ham)
    
    print("Ham diff : " + str(Ham-s.Ham))
    print("log-Jacobian: " + str(logJac))
    return out,logJac
    
# Hamiltonian for extended phase space integrator    
def extHam(lpFun,q,p,qt,pt,omega):
    [f,g] = lpFun(q)
    [ft,gt] = lpFun(qt)
    L = 1.0 + alpha**2*np.dot(g,g)
    Lt = 1.0 + alpha**2*np.dot(gt,gt)
    
    
    Ginvpt = pt - (alpha**2/L)*np.dot(g,pt)*g
    HA = -f + 0.5*np.log(L) + 0.5*np.dot(pt,Ginvpt)
    
    Gtinvp = p - (alpha**2/Lt)*np.dot(gt,p)*gt
    HB = -ft + 0.5*np.log(Lt) + 0.5*np.dot(p,Gtinvp)
    
    Hamj = HA + HB + 0.5*omega*(np.dot(q-qt,q-qt) + np.dot(p-pt,p-pt))
    return(Hamj)

# Extended phase space integrator
# Note, not optimized with respect to gradient/Hessian evals
# Note, this presumes that m=I
def mongeEPSInt(lpFun,q,p,qt=None,pt=None,h=0.3,omega=100.0,nstep=1):
    
    if(qt is None):
        print("jittering")
        # Jittered copy
        qt = q + (h**2)*np.random.uniform(low=-1.0,high=1.0,size=len(q))
        pt = p + (h**2)*np.random.uniform(low=-1.0,high=1.0,size=len(q))
    
    [f,g] = lpFun(q)
    
    L = 1.0 + alpha**2*np.dot(g,g)
    
    Ginvp = p - (alpha**2/L)*np.dot(g,p)*g
    Ham0 = -f + 0.5*np.log(L) + 0.5*np.dot(p,Ginvp)
    
    Ham0j = extHam(lpFun,q,p,qt,pt,omega)
    
    wt1 = 0.5*np.cos(2.0*omega*h)
    wt2 = 0.5*np.sin(2.0*omega*h)
    
    
    for i in range(nstep):
        #print(qt[0])
        # PhiB: update q based on qt and p AND pt based on qt and p
        [ft,gt] = lpFun(qt)
        Lt = 1.0 + alpha**2*np.dot(gt,gt)
        tmp1 = alpha**2*np.dot(gt,p)/Lt
        q = q + 0.5*h*(p - tmp1*gt)
        [_,_,Htgt] = lpFun(qt,v=gt)
        [_,_,Htp] = lpFun(qt,v=p)
        pt = pt - 0.5*h*(-gt + (tmp1**2 + alpha**2/Lt)*Htgt - tmp1*Htp)
        
        #print("B")
        #print(qt[0])
        #print(Ham0j-extHam(lpFun,q,p,qt,pt,omega))
        
        # PhiA: pdate qt based on q and pt AND p based on q and pt
        [f,g] = lpFun(q)
        L = 1.0 + alpha**2*np.dot(g,g)
        tmp1 = alpha**2*np.dot(g,pt)/L
        qt = qt + 0.5*h*(pt - tmp1*g)
        [_,_,Hg] = lpFun(q,v=g)
        [_,_,Hpt] = lpFun(q,v=pt)
        p = p - 0.5*h*(-g + (tmp1**2 + alpha**2/L)*Hg - tmp1*Hpt)
        
        #print("A")
        #print(qt[0])
        #print(Ham0j-extHam(lpFun,q,p,qt,pt,omega))
        
        # PhiC:
        qbar = 0.5*(q+qt)
        pbar = 0.5*(p+pt)
        Delq = q-qt
        Delp = p-pt
        
        q = qbar + wt1*Delq + wt2*Delp
        qt = qbar - wt1*Delq - wt2*Delp
        p = pbar + wt1*Delp - wt2*Delq
        pt = pbar - wt1*Delp + wt2*Delq
        
        #print("C")
        #print(qt[0])
        #print(Ham0j-extHam(lpFun,q,p,qt,pt,omega))
        
        # PhiA: pdate qt based on q and pt AND p based on q and pt
        [f,g] = lpFun(q)
        L = 1.0 + alpha**2*np.dot(g,g)
        tmp1 = alpha**2*np.dot(g,pt)/L
        qt = qt + 0.5*h*(pt - tmp1*g)
        [_,_,Hg] = lpFun(q,v=g)
        [_,_,Hpt] = lpFun(q,v=pt)
        p = p - 0.5*h*(-g + (tmp1**2 + alpha**2/L)*Hg - tmp1*Hpt)
        
        
        #print("A")
        #print(qt[0])
        #print(Ham0j-extHam(lpFun,q,p,qt,pt,omega))
        
        # PhiB: update q based on qt and p AND pt based on qt and p
        [ft,gt] = lpFun(qt)
        Lt = 1.0 + alpha**2*np.dot(gt,gt)
        tmp1 = alpha**2*np.dot(gt,p)/Lt
        q = q + 0.5*h*(p - tmp1*gt)
        [_,_,Htgt] = lpFun(qt,v=gt)
        [_,_,Htp] = lpFun(qt,v=p)
        pt = pt - 0.5*h*(-gt + (tmp1**2 + alpha**2/Lt)*Htgt - tmp1*Htp)
    
        #print("B")
        #print(qt[0])
        #print(Ham0j-extHam(lpFun,q,p,qt,pt,omega))
    
        [f,g] = lpFun(q)
        L = 1.0 + alpha**2*np.dot(g,g)
        Ginvp = p - (alpha**2/L)*np.dot(g,p)*g
        Ham = -f + 0.5*np.log(L) + 0.5*np.dot(p,Ginvp)
        
        
        #print(np.max(np.abs(q-qt))/(h**2))
        #print(np.max(np.abs(p-pt))/(h**2))
        print(np.max(np.abs(q-qt))<h**2 and np.max(np.abs(p-pt))<h**2)
        
    print("joint")
    print(Ham0j-extHam(lpFun,q,p,qt,pt,omega))
    print("marginal")
    print(Ham0-Ham)
    print("rel dev: ")
    print(np.max(np.abs(q-qt))/(h**2))
    print(np.max(np.abs(p-pt))/(h**2))
    
    print("acceptprob")
    print(np.exp(min(0.0,Ham0-Ham))*(np.max(np.abs(q-qt))<h**2 and np.max(np.abs(p-pt))<h**2))
    print("phase")
    print(2.0*omega*h)
    return q,p,qt,pt



T = 5.0
nstep = 250
h = T/nstep
omega = 1.5

lp =  corrGauss # funnel1 # stdGauss
q0 = np.array([-2.0,-1.51])
p0 = np.array([-1.0,-1.50])

[q,p,qt,pt] = mongeEPSInt(lp,q0,p0,omega=omega,h=h,nstep=nstep)
# reversibility check
#[qb,pb,qtb,ptb] = mongeEPSInt(lp,q=q,p=-p,qt=qt,pt=-pt,omega=omega,h=h,nstep=nstep)

 
s0 = state()   
s0.evalFirst(lp,q0,p0)
s1,logJacf = mongeInt(s0,lp,h=h,nstep=nstep)

# reversibility check
#s1.momentumFlip()
#s0b,logJacb = mongeInt(s1,lp,h=h,nstep=nstep)
#s0b.momentumFlip()



#out = mongeIntAdapt(s0, corrGauss,0.1)
#print(out.y[:,len(out.y[1,:])-1])

