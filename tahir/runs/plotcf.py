#/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


def den_plot(r0mesh, denmesh):
    ''' Plot scalar field in real space '''
    plt.rcParams.update({'font.size': 18})
    r0num = np.int(r0mesh.shape[0]**0.5)
    X = r0mesh.T[0].reshape(r0num,r0num)
    Y = r0mesh.T[1].reshape(r0num,r0num)
    Z = denmesh.reshape(r0num,r0num)

    plt.figure(1000,figsize=(6,6))
    ax=plt.gca()
    ax.set_aspect('equal')
    plt.contourf(X,Y,Z, 30, cmap = 'hot')
    ax.set_facecolor('black')
    plt.colorbar(fraction=0.046, pad=0.04)
    xcap = 'x (a)'
    ycap = 'y (a)'
    plt.ylabel(ycap)
    plt.xlabel(xcap)
    outfile = 'denmap.pdf'
    plt.savefig(outfile,bbox_inches='tight')
    plt.close(1000)

eng_k = np.loadtxt('./data/eng-matrix.txt')

nphi = 50
phimax = 0.02
phi0 = np.linspace(0, phimax, nphi)

#Emax = np.max(energy)*0.05
#Emin = np.min(energy)*0.05
Emax = -3.995
Emin = -4.005
nE = 250
dE = (Emax-Emin)/(nE-1)
E = np.array([i*dE+Emin for i in range(nE)])
gE = np.zeros((nphi,nE))
for i in range(nphi):
  for j in range(nE-1):
    #idx = np.where(np.logical_and(energy[:,i]>E[j],energy[:,i]<E[j+1]))[0]
    idx = np.where(np.logical_and(eng_k[:,i]>E[j],eng_k[:,i]<E[j+1]))[0]
    gE[i,j+1] = np.size(idx)

plt.figure(1000,figsize=(6,6))
ax=plt.gca()
#ax.set_aspect('equal')
plt.contourf(phi0,E,gE.transpose(), 30, cmap = 'viridis')
ax.set_facecolor('black')
plt.colorbar(fraction=0.046, pad=0.04)
xcap = 'x (a)'
ycap = 'y (a)'
plt.ylabel(ycap)
plt.xlabel(xcap)
outfile = 'denmap.pdf'
plt.savefig(outfile,bbox_inches='tight')

