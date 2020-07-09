#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

vz = 1000.0
t = 50.0
tr = 0.5*t
ma = 0.5/t
ma = (ma*vz)/(vz+ma*tr**2)
mu = vz+6*t
dels = 25.0
kf = np.sqrt(2*ma*mu)
delp = (dels*tr*kf)/(2*vz*mu)

nj = 751
nk = 501
j = 1.5*np.arange(0,nj,1)
k = 1.5*np.arange(0,nk,1)
J, K = np.meshgrid(j,k)
lsq = J**2/9 + K**2/3

na = 25
#a = np.array([np.exp(i/(10*np.pi)) for i in range(na)])
a = np.array([2*(75+2*i) for i in range(na)])
En = np.zeros((na,nj*nk))

for i in range(na):
  lf = 2*np.pi/(a[i]*kf)
  tmp = mu * np.sqrt(1 + lf**4*lsq**2 - lf**2*(2+(lf*delp/kf)**2)*lsq)
  En[i,:] = np.matrix.flatten(tmp)
  if i==20:
    plt.imshow(tmp)
    plt.colorbar()
    plt.show()
    plt.close()
  
#En[En>1100] = 0
# DOS
Emax = np.max(En)
Emin = 0
nE = 75
dE = (Emax-Emin)/(nE-1)
E = np.array([i*dE+Emin for i in range(nE)])
gE = np.zeros((na,nE))
for i in range(na):
  for j in range(nE-1):
    idx = np.where(np.logical_and(En[:,i]>E[j],En[:,i]<E[j+1]))[0]
    gE[i,j+1] = np.size(idx)

extent = [a[0],a[-1],Emin,Emax]
plt.imshow(gE.transpose(), extent=extent, origin='lower', aspect='auto')
plt.colorbar()
plt.show()
plt.close()
