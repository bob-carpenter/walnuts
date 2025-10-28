import numpy as np

# similar to C++ implementation in
# https://github.com/flatironinstitute/walnuts/blob/main/include/walnuts/dual_average.hpp

class dualAverage:
    
    
    
    def __init__(self, initPar,target,obsCountOffset=10,learnRate=0.05,decayRate=0.75):
        self.logEst = np.log(initPar)
        self.logEstAvg = self.logEst
        self.gradAvg = 0.0
        self.obsCount = 0.0
        self.logStepOffset= np.log(10.0) + np.log(initPar)
        self.target = target
        self.obsCountOffset = obsCountOffset
        self.learnRate = learnRate
        self.decayRate = decayRate
    
    def observe(self,targetDraw):
        
        self.obsCount += 1.0
        prop = 1.0/(self.obsCount + self.obsCountOffset)
        self.gradAvg = (1.0-prop)*self.gradAvg + prop*(self.target-targetDraw)
        self.logEst = self.logStepOffset - np.sqrt(self.obsCount)/self.learnRate*self.gradAvg
        
        prop2 = self.obsCount**(-self.decayRate)
        
        self.logEstAvg = prop2*self.logEst + (1.0-prop2)*self.logEstAvg
    
    
    def par(self):
        return(np.exp(self.logEstAvg))
        
    
    





