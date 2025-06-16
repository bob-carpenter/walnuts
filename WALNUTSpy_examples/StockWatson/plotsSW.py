import sys
if(sys.path[-1].find("WALNUTSpy")==-1):
    sys.path.append("../WALNUTS_gh/walnuts/WALNUTSpy")



import numpy as np
import MCMCutils as mu
import matplotlib.pyplot as plt
import arviz as az
import bridgestan as brs
import json

with open("swdata.json") as json_file:
    json_data = json.load(json_file)

yy = np.array(json_data["y"])

mdlDR = brs.StanModel("sw_dr.stan","swdata.json")
nms = mdlDR.param_names(include_tp=True)


mdlInnov = brs.StanModel("sw_innov.stan","swdata.json")
nmsi = mdlInnov.param_names(include_tp=True)

sw = np.load("sw_walnuts_dr_res.npy")
dw = np.load("sw_walnuts_dr_diag.npy")
sn = np.load("sw_nuts_dr_res.npy")
dn = np.load("sw_nuts_dr_diag.npy")

swi = np.load("sw_walnuts_innov2_res.npy")
dwi = np.load("sw_walnuts_innov2_diag.npy")
swir = np.load("sw_walnutsR2P_innov2_res.npy")
dwir = np.load("sw_walnutsR2P_innov2_diag.npy")
sni = np.load("sw_nuts_innov2_res.npy")
dni = np.load("sw_nuts_innov2_diag.npy")


plt.figure()
plt.subplot(1,2,1)
plt.hist((swir[nmsi.index("sigma"),:]),25,density=True)
plt.axis([0.1,0.6,0,10])
plt.subplot(1,2,2)
plt.hist((sni[nmsi.index("sigma"),:]),25,density=True)
plt.axis([0.1,0.6,0,10])


qstatwi = np.zeros((251,3))
qstatni = np.zeros((251,3))



for i in range(251):
    st = "z."+str(i+1)
    sam = swir[nmsi.index(st),1000:11000]
    qstatwi[i,:] = np.quantile(sam, q=(0.05,0.5,0.95))
    sam = sni[nmsi.index(st),1000:11000]
    qstatni[i,:] = np.quantile(sam, q=(0.05,0.5,0.95))
    
xstatwi = np.zeros((252,3))
xstatni = np.zeros((252,3))

for i in range(252):
    st = "x."+str(i+1)
    sam = swir[nmsi.index(st),1000:11000].flatten()
    xstatwi[i,:] = np.quantile(sam, q=(0.05,0.5,0.95))
    sam = sni[nmsi.index(st),1000:11000].flatten()
    xstatni[i,:] = np.quantile(sam, q=(0.05,0.5,0.95))
    
    
taustatwi = np.zeros((252,3))
taustatni = np.zeros((252,3))

for i in range(252):
    st = "tau."+str(i+1)
    sam = swir[nms.index(st),1000:11000].flatten()
    taustatwi[i,:] = np.quantile(sam, q=(0.05,0.5,0.95))
    sam = sni[nms.index(st),1000:11000].flatten()
    taustatni[i,:] = np.quantile(sam, q=(0.05,0.5,0.95))


obsstatwiu = np.zeros((252,3))
obsstatwil = np.zeros((252,3))
obsstatniu = np.zeros((252,3))
obsstatnil = np.zeros((252,3))

for i in range(252):
    stt = "tau."+str(i+1)
    stx = "x."+str(i+1)
    sam = swir[nms.index(stt),1000:11000] + 2*np.exp(0.5*swir[nms.index(stx),1000:11000])
    obsstatwiu[i,:] = np.quantile(sam, q=(0.05,0.5,0.95))
    sam = swir[nms.index(stt),1000:11000] - 2*np.exp(0.5*swir[nms.index(stx),1000:11000])
    obsstatwil[i,:] = np.quantile(sam, q=(0.05,0.5,0.95))
    sam = sni[nms.index(stt),1000:11000] + 2*np.exp(0.5*sni[nms.index(stx),1000:11000])
    obsstatniu[i,:] = np.quantile(sam, q=(0.05,0.5,0.95))
    sam = sni[nms.index(stt),1000:11000] - 2*np.exp(0.5*sni[nms.index(stx),1000:11000])
    obsstatnil[i,:] = np.quantile(sam, q=(0.05,0.5,0.95))


plt.figure()
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 14 
fig_size[1] = 7
#plt.rcParams["figure.figsize"] = fig_size

plt.subplot(3,1,1)

plt.plot(qstatwi[:,1])
plt.plot(qstatwi[:,0])
plt.plot(qstatwi[:,2])

plt.plot(qstatni[:,1],":")
plt.plot(qstatni[:,0],":")
plt.plot(qstatni[:,2],":")
plt.axis([0,252,-14,0])
plt.ylabel("$z_t$")

plt.subplot(3,1,2)
plt.plot(xstatwi[:,1])
plt.plot(xstatwi[:,0])
plt.plot(xstatwi[:,2])

plt.plot(xstatni[:,1],":")
plt.plot(xstatni[:,0],":")
plt.plot(xstatni[:,2],":")
plt.axis([0,252,-5,1])
plt.ylabel("$x_t$")

plt.subplot(3,1,3)
plt.plot(taustatwi[:,1])
plt.plot(obsstatwiu[:,1])
plt.plot(obsstatwil[:,1])
#plt.plot(taustatwi[:,0])
#plt.plot(taustatwi[:,2])

plt.plot(taustatni[:,1],":")
plt.plot(obsstatniu[:,1],":")
plt.plot(obsstatnil[:,1],":")


plt.plot(yy,".",markersize=3)
plt.axis([0,252,-3,4.0])
plt.ylabel("$\\tau_t$ and observations")



plt.savefig("swFactors.pdf",)

#plt.close()

plt.figure(3)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 7
fig_size[1] = 14
plt.rcParams["figure.figsize"] = fig_size



mis = 0.1*2**(-dwi[:,8])
mis_step = np.unique(mis)
mis_prev = np.zeros(len(mis_step))
i = 0
for s in mis_step:
    mis_prev[i] = sum(mis==s)/len(mis)
    i += 1

mas = 0.1*2**(-dwi[:,9])
mas_step = np.unique(mas)
mas_prev = np.zeros(len(mas_step))
i = 0
for s in mas_step:
    mas_prev[i] = sum(mas==s)/len(mas)
    i += 1
    

misr = 0.1*2**(-dwir[:,21])
misr_step = np.unique(misr)
misr_prev = np.zeros(len(misr_step))
i = 0
for s in misr_step:
    misr_prev[i] = sum(misr==s)/len(misr)
    i += 1

masr = 0.1*2**(-dwir[:,22])
masr_step = np.unique(masr)
masr_prev = np.zeros(len(masr_step))
i = 0
for s in masr_step:
    masr_prev[i] = sum(masr==s)/len(masr)
    i += 1



plt.subplot(1,3,1)


plt.semilogx(misr_step,misr_prev,"ks-",label="R2P largest step size of orbit")
plt.semilogx(masr_step,masr_prev,"k.-",label="R2P smallest step size of orbit")
plt.semilogx(mis_step,mis_prev,"bs-",label="D largest step size of orbit")
plt.semilogx(mas_step,mas_prev,"b.-",label="D smallest step size of orbit")
plt.semilogx(dni[0,15]*np.ones(2),np.array([-0.05,1.0]),"r",label="NUTS step size")
plt.semilogx((0.1/8.0)*np.ones(2),np.array([-0.05,1.0]),":",label="_nolabel_")



#plt.axis([0.0002,0.15/8,0.01,1])

plt.ylabel("proportion of orbits with step size")
plt.xlabel("step size")
plt.legend()

plt.subplot(1,3,2)
plt.semilogy(dni[1000:11000,17],"r",label="NUTS")
plt.semilogy(dwi[1000:11000,17],"b",label="WALNUTS D")
plt.semilogy(dwir[1000:11000,17],"k",label="WALNUTS R2P")

plt.xlabel("MCMC iteration #")
plt.ylabel("max orbit energy error")
plt.legend()


plt.subplot(1,3,3)
cen,hh,bins = mu.igrErrStatHistogram(dni[1000:11000],40)
plt.plot(cen,hh,"r",label="NUTS")
for b in bins:
    plt.plot(b*np.ones(2),np.array([-1,10]),'0.6',":",linewidth=0.3,label="_nolegend_")
cen,hh,bins = mu.igrErrStatHistogram(dwi[1000:11000],40)
plt.plot(cen,hh,"b",label="WALNUTS D")
cen,hh,bins = mu.igrErrStatHistogram(dwir[1000:11000],40)
plt.plot(cen,hh,"k",label="WALNUTS R2P")
plt.axis([-0.05,1.05,0,3])
plt.xlabel("standardized distance from current state")
plt.ylabel("density-scaled histogram of orbits")
plt.legend()


plt.savefig("swDiag.pdf",)

nnuts = sum(dni[1000:11000,6])
nwD = sum(dwi[1000:11000,6])+sum(dwi[1000:11000,7])
nwR = sum(dwir[1000:11000,6])+sum(dwir[1000:11000,7])
print("D cost ", nwD/nnuts)
print("R2P cost ", nwR/nnuts)

