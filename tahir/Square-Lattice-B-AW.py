#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
np.set_printoptions(linewidth=np.inf, precision=4)

def hjjn(_n, _i, _j, _phi0, _kj):
  cs = np.cos(2*np.pi*_phi0*_n-_kj)
  if _i == _j:
    return 2*cs+2
  elif _i+1 == _j:
    #print(-sp.jv(_n,_phi0))
    return 0
  elif _i-1 == _j:
    #print(-(-1)**_n*sp.jv(_n,_phi0))
    return 0
  else:
    return 0

rc = 10
Ns = 2*rc+1

mc = 0
Nm = 2*mc+1

ka = 0.1

nphi = 100
phimax = 0.002
#phi0 = np.linspace(-phimax,phimax,nphi)
phi0 = np.linspace(0.0001,phimax,nphi)
energy = np.zeros((Ns, nphi)) #((nphi))

for k in range(nphi):
#  Hn = np.zeros([Ns, Ns], "complex")

#  for n in range(Nm):
    Hn = np.zeros([Ns, Ns])
    for i in range(Ns):
#      for j in range(Ns):
        kj = ka
        Hn[i,i] = hjjn(i, i, i, phi0[k], kj)
#        if k==0:
#          print(Hn[n,i,j])

    print(Hn)
  #print()
# here in energy second we need k but how ?
    energy[:,k] = np.linalg.eigvalsh(Hn)

xaxis = np.array([-1, 1])
plt.figure(figsize=(6,6))
plt.tick_params(
    axis='x',
    which='both',
    bottom='on',
    top='off',
    labelbottom='on')
plt.ylabel('$E(\phi_0)$', fontsize=12)
plt.xlabel('$\phi_0$', fontsize=12)
#plt.xlim(phi0[0], phi0[-1])
plt.xlim(phi0[0], phi0[-1])
#plt.ylim(np.min(energy),np.max(energy))
#plt.ylim(-4, 0)


for i in range(Nm):
  plt.plot(phi0, energy[i,:], ',', color = 'b', markersize = 1.6)
#  plt.plot(phi0, energy[i,:], color = 'b', markersize = .1)

#for i in range(Nm*Ns):
#  plt.plot(xaxis,[energy[i],energy[i]], 'g')

plt.tight_layout()

#plt.savefig('../../data/fig-spectral-flow.pdf')
plt.show()
plt.close()

#Emax = np.max(energy)*0.05
#Emin = np.min(energy)*0.05
Emax = 4.001
Emin = 3.990
nE = 100
dE = (Emax-Emin)/(nE-1)
E = np.array([i*dE+Emin for i in range(nE)])
gE = np.zeros((nphi,nE))
for i in range(nphi):
  for j in range(nE-1):
    idx = np.where(np.logical_and(energy[:,i]>E[j],energy[:,i]<E[j+1]))[0]
    gE[i,j+1] = np.size(idx)

#print(gE)
plt.figure(figsize=(10,10))
Extent = [0, 1*phimax, 1*Emin, 1*Emax]
#plt.colorbar()
# include cmap='some_color' into the imshow function
plt.imshow(gE.transpose(), origin='lower', extent=Extent, cmap='Greens', aspect=2/5)
#plt.savefig('../../data/fig-dos-spectral-flow.pdf')
plt.show()
plt.close()
