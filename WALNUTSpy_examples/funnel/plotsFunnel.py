
# NOTE: assumes main mainFunnel.py and mainFunnelNUTS.py has already been run


import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy as sp
from sklearn.neighbors import KNeighborsRegressor

nknn = 200
hnknn = np.round(nknn/2).astype(np.int64)
neigh = KNeighborsRegressor(n_neighbors=nknn)



def qqnormal(x,loc=0.0,scale=1.0,plot=False):
    ys = np.sort(x)
    n = len(x)
    ps = np.arange(start=1.0,stop=n+1.0)/(n+1.0)
    xs = sp.stats.norm.ppf(ps,loc=loc,scale=scale)
    if(plot):
        
        plt.plot(xs,ys,".b")
        plt.plot(np.array([xs[0],xs[-1]]),np.array([xs[0],xs[-1]]),"c")
        plt.xlabel("theoretical quantiles")
        plt.ylabel("sample quantiles")
        
    return(xs,ys)
 

samples = np.load("funnelSamples_FN.npy")
diagnostics = np.load("funnelDiagnostics_FN.npy")
omin = np.load("funnelOmin_FN.npy")
omax = np.load("funnelOmax_FN.npy")


samplesNUTS = np.load("funnelNUTSSamples_FN.npy")
diagNUTS = np.load("funnelNUTSDiagnostics_FN.npy")


subset = np.arange(start=0,stop=len(diagnostics[:,0]),dtype=np.int64,step=200 )

omegaGrid = np.arange(start=np.min(samples[0,:]),stop=np.max(samples[0,:]) ,step=0.01)

plt.figure()
plt.subplot(141)
plt.hist(samples[0,:],density=True,histtype='bar',edgecolor = 'Black',bins=28)
plt.plot(omegaGrid, sp.stats.norm.pdf(omegaGrid,loc=0.0,scale=3.0),"r--")
plt.xlabel("$\omega$")
plt.title("WALNUTS")

plt.subplot(142)
plt.hist(samplesNUTS[0,:],density=True,histtype='bar',edgecolor = 'Black',bins=21)
plt.plot(omegaGrid, sp.stats.norm.pdf(omegaGrid,loc=0.0,scale=3.0),"r--")
plt.xlabel("$\omega$")
plt.title("NUTS")

plt.subplot(143)
plt.plot(omax[0,subset],diagnostics[subset,1] + np.random.uniform(low=-0.25,high=0.25,size=len(subset) ),
         ".k",markersize=1)

neigh.fit(omax[0,:].reshape(-1,1),diagnostics[:,1])
maxgrid= np.arange(start=np.sort(omax[0,:])[hnknn],stop=np.sort(omax[0,:])[-hnknn],step=0.05)
plt.plot(maxgrid, neigh.predict(maxgrid.reshape(-1,1)),color="r")   


#smoothed = sm.nonparametric.lowess(exog=omax[0,subset], 
#                               endog=diagnostics[subset,1], frac=0.1)
#plt.plot(smoothed[:,0], smoothed[:,1],"r",linewidth=1)

#plt.legend(["# doublings (jittered, thinned)","LOWESS regression"],markerscale=10)
plt.xlabel("largest $\omega$ in orbit")
plt.ylabel("number of doublings in accepted orbit")

plt.subplot(144)
plt.plot(omin[0,subset],diagnostics[subset,9]+ np.random.uniform(low=-0.25,high=0.25,size=len(subset) ),
         ".k",markersize=1)

#xvals=np.arange(start=np.min(omin[0,:]),stop=np.max(omin[0,:]),step=0.05 )
#smoothed = sm.nonparametric.lowess(exog=omin[0,subset], 
#                               endog=diagnostics[subset,9], frac=0.1,
#                               xvals=xvals)
neigh.fit(omin[0,:].reshape(-1,1),diagnostics[:,9])
mingrid= np.arange(start=np.sort(omin[0,:])[hnknn],stop=np.sort(omin[0,:])[-hnknn],step=0.05)
plt.plot(mingrid, neigh.predict(mingrid.reshape(-1,1)),color="r")   

#plt.legend(["largest $\ell$ (jittered, thinned)","LOWESS regression"],markerscale=10)

plt.xlabel("smallest $\omega$ in orbit")
plt.ylabel("largest $\log_2(\ell)$ in orbit")


fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14  
fig_size[1] = 7 
plt.rcParams["figure.figsize"] = fig_size
plt.savefig("funnelHistDiag.pdf")




thresh = -3.0
plt.figure()
plt.subplot(121)
plt.plot(np.array([-14,thresh]),np.array([-14,thresh]),"r--",label='_nolegend_')
incr = np.floor(len(samples[0,:])/5).astype(np.int64)
for i in range(5):
    
    
    [xs,ys] = qqnormal(samples[0,(i*incr):((i+1)*incr)],scale=3.0)
    plt.plot(xs[xs<thresh],ys[xs<thresh],"k")
    
    
    [xs,ys] = qqnormal(samplesNUTS[0,(i*incr):((i+1)*incr)],scale=3.0)
    plt.plot(xs[xs<thresh],ys[xs<thresh],'0.8')
    if(i==0): plt.legend(["WALNUTS","NUTS"])
    
plt.axis([-9,thresh,-9,thresh])
plt.xlabel("$\omega$ theoretical quantiles")
plt.ylabel("$\omega$ MCMC output quantiles")
plt.title("QQ-plots from 200k iterations (only left hand tail)")

plt.subplot(122)
plt.plot(np.arange(1,1001),samples[0,0:1000],"k")
plt.plot(np.arange(1,1001),samplesNUTS[0,0:1000],'0.8')

plt.legend(["WALNUTS","NUTS"])
plt.xlabel("MCMC iteration #")
plt.ylabel("$\omega$")
plt.title("trace plots (first 1000 iterations only)")




fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14  
fig_size[1] = 7 
plt.rcParams["figure.figsize"] = fig_size
plt.savefig("funnel_qq.pdf")





print("relative cost: " ,sum(diagNUTS[:,6])/(sum(diagnostics[:,6])+sum(diagnostics[:,7])))
print("WALNUTS step size : ", diagnostics[0,15])
print("WALNUTS delta : ",diagnostics[0,18])
print("NUTS step size : ",diagNUTS[0,15])



