import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class relativisticKineticDistribution:
    
    def logpz(self,z,offset):
        return(-self.c*np.sqrt(self.cSq + np.exp(2.0*z)) + self.d*z+offset)
    
    
    
    def setup(self,d,c,ngrid=2048):
        print("setup routine")
        print("c: " + str(c))
        print("d: " + str(d))
        self.d = d
        self.c = c
        self.cSq = c**2
        self.ngrid = ngrid
        
        self.zcen = 0.5*np.log(0.5*(self.d/self.cSq)*(self.d + np.sqrt(self.d**2 + 4.0*(self.cSq)**2)))
        self.offset = -self.logpz(self.zcen,0.0)
        tmp = np.exp(2.0*self.zcen)
        
        numZero = -32.0
        llim = self.zcen - 1.0
        while(True):
            if(self.logpz(llim, self.offset) < numZero):
                break
            llim -= 1.0
        llim = sp.optimize.bisect(f=lambda x: self.logpz(x, self.offset) - numZero,a=llim,b=self.zcen)
        
        rlim = self.zcen + 1.0
        while(True):
            if(self.logpz(rlim, self.offset) < numZero):
                break
            rlim += 1.0
        rlim = sp.optimize.bisect(f=lambda x: self.logpz(x, self.offset) - numZero,a=self.zcen,b=rlim)
        
        self.grid = np.linspace(llim, rlim, num=ngrid)
        self.pz = np.exp(self.logpz(self.grid,self.offset))
        cs0 = np.cumsum(self.pz[0:(ngrid-1)])
        cs1 = np.cumsum(self.pz[1:ngrid])
        self.cdf = 0.5*(cs0+cs1)*(self.grid[1]-self.grid[0]) # cdf based on trapeziodal integration
        self.cdf = self.cdf/self.cdf[-1]
        self.grid = self.grid[1:ngrid]
        #plt.plot(self.cdf,self.grid)
        
    def checkCurrent(self,d,c,ngrid=2048):
        if(not (self.d==d and np.abs(self.c-c)<1.0e-14 and self.ngrid == ngrid)):
            self.setup(d,c,ngrid)
    
    def sample(self):
        r = np.exp(np.interp(np.random.uniform(), self.cdf, self.grid))
        dd = np.random.normal(size=self.d)
        return((r/np.linalg.norm(dd))*dd)

    def __init__(self):
        self.d = -1
        self.c = -1.0
        self.ngrid = -1
        
# single instance used for simulations
globalRelativisticKineticDistribution = relativisticKineticDistribution()
        
