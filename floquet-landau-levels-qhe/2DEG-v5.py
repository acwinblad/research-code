#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
np.set_printoptions(linewidth=np.inf, precision=4)

def hjjn(_n, _i, _j, _phi0, _kj, _kya):
  cs = np.cos(_kj)
  if _i == _j:
    return -1*(sp.jv(_n,_phi0*cs)*np.exp(1.0j*_kya)+sp.jv(_n,-_phi0*cs)*np.exp(-1.0j*_kya)) * np.exp(1.0j*_n*np.pi/2)
  elif _i+1 == _j:
    #print(-sp.jv(_n,_phi0))
    return -1*sp.jv(_n,_phi0)
  elif _i-1 == _j:
    #print(-(-1)**_n*sp.jv(_n,_phi0))
    return -1*(-1)**_n*sp.jv(_n,_phi0)
  else:
    return 0

pointsFlag = True
dosFlag = False

rc = 12
Ns = 2*rc+1

mc = 7
Nm = 2*mc+1

nky = 40
kyMax = 0.05*np.pi
ky_a = np.linspace(0,kyMax,nky)
ka = 0.1
hw = 0.1
nphi = 100
phimax = 0.04
#phi0 = np.linspace(-phimax,phimax,nphi)
phi0 = np.linspace(0,phimax,nphi)
energy = np.zeros((Nm*Ns,nphi))

for l in range(nky):
  for k in range(nphi):
    Hn = np.zeros([Nm,Ns,Ns], "complex")

    for n in range(Nm):
      for i in range(Ns):
        for j in range(Ns):
          #kj = (j%Ns)*ka
          kj = (j-rc)*ka
          Hn[n,i,j] = hjjn(-n, i, j, phi0[k], kj, ky_a[l])
  #        if k==0:
  #          print(Hn[n,i,j])

    #print(Hn[0])
    #print()

    Qmn = np.zeros([Nm*Ns, Nm*Ns],"complex")

    for i in range(Nm):
      for j in range(Nm-i):
        midx = mc-j
        r1 = (i+j)*Ns 
        r2 = (i+j+1)*Ns
        c1 = j*Ns
        c2 = (j+1)*Ns
        if i == 0:
          Qmn[r1:r2,r1:r2] = Hn[i,:,:] - midx*hw*np.eye(Ns)
        else:
          Qmn[r1:r2,c1:c2] = Hn[i,:,:]

    #print(Qmn[0:Ns,0:Ns], '\n')
    #print(Qmn[0*Ns:3*Ns,0*Ns:3*Ns])

    energy[:,k] = np.linalg.eigvalsh(Qmn)
  
  if l == 0:
    eng_k = energy
  else:
    eng_k = np.vstack((eng_k,energy))
    
  #xaxis = np.array([-1, 1])
  #plt.figure(figsize=(15,15))
  #plt.tick_params(
  #    axis='x',
  #    which='both',
  #    bottom='on',
  #    top='off',
  #    labelbottom='on')
  #plt.ylabel('$E(\phi_0)$', fontsize=12)
  #plt.xlabel('$\phi_0$', fontsize=12)
  ##plt.xlim(phi0[0], phi0[-1])
  #plt.xlim(phi0[0], phi0[-1])
  ##plt.ylim(np.min(energy),np.max(energy))
  ##plt.ylim(-4, 0)


if pointsFlag==True:
  for i in range(Nm*Ns):
    plt.plot(phi0, energy[i,:], ',', color = 'b', markersize = 1.6)
  #  plt.plot(phi0, energy[i,:], color = 'b', markersize = .1)

  #for i in range(Nm*Ns):
  #  plt.plot(xaxis,[energy[i],energy[i]], 'g')

  plt.tight_layout()

  #plt.savefig('../../data/fig-spectral-flow.pdf')
  plt.show()
  plt.close()

if dosFlag==True:
  #Emax = np.max(energy)*0.05
  #Emin = np.min(energy)*0.05
  Emax = -0.015
  Emin = -0.03
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
  plt.figure(figsize=(12,12))
  #plt.title('k_y = %1.5f' % ky_a[l])
#  Extent = [0, 4*phimax, 2*Emin, 2*Emax]
  Extent = [0, 0.5*phimax, Emin, Emax]
  #plt.colorbar()
  # include cmap='some_color' into the imshow function
  #plt.imshow(gE.transpose(), origin='lower', extent=Extent, cmap='viridis', aspect=3/5)
#  plt.imshow(gE.transpose(), origin='lower', extent=Extent, cmap='viridis', aspect=3/5)
  plt.imshow(gE.transpose(), cmap='viridis',interpolation='bicubic', origin='lower', extent=Extent, aspect=0.5)
  #plt.savefig('../../data/fig-dos-spectral-flow.pdf')
  plt.show()
  plt.close()
