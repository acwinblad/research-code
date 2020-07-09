#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
np.set_printoptions(linewidth=np.inf, precision=4)

eng_k = np.loadtxt('./data/eng-matrix.txt')
nphi = 100
phimax = 0.04

#Emax = np.max(energy)*0.05
#Emin = np.min(energy)*0.05
Emax = -3.99
Emin = -4.01
nE = 250
dE = (Emax-Emin)/(nE-1)
E = np.array([i*dE+Emin for i in range(nE)])
gE = np.zeros((nphi,nE))
for i in range(nphi):
  for j in range(nE-1):
    #idx = np.where(np.logical_and(energy[:,i]>E[j],energy[:,i]<E[j+1]))[0]
    idx = np.where(np.logical_and(eng_k[:,i]>E[j],eng_k[:,i]<E[j+1]))[0]
    gE[i,j+1] = np.size(idx)

#print(gE)
plt.figure(1000,figsize=(12,12))
#plt.title('k_y = %1.5f' % ky_a[l])
#  Extent = [0, 4*phimax, 2*Emin, 2*Emax]
Extent = [0, 0.5*phimax, Emin, Emax]
#plt.colorbar()
# include cmap='some_color' into the imshow function
#plt.imshow(gE.transpose(), origin='lower', extent=Extent, cmap='viridis', aspect=3/5)
#  plt.imshow(gE.transpose(), origin='lower', extent=Extent, cmap='viridis', aspect=3/5)
plt.imshow(gE.transpose(), cmap='viridis',interpolation='bicubic', origin='lower', extent=Extent, aspect=0.5)
plt.savefig('./figures/dos.pdf')
#plt.show(1000)
plt.close(1000)
