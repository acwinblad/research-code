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

# variable values
# mc [5-10]
mc = 0
Nm = 2*mc+1

# rc [10-15]
rc = 3
Ns = 2*rc+1

# static values
nky = 1
kyMax = 0.005*np.pi
ky_a = np.linspace(0,kyMax,nky)
ka = 0.1
hw = 0.1
nphi = 50
phimax = 0.02
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
    print(Qmn)

    # eigenvalue for k_y
    energy[:,k] = np.linalg.eigvalsh(Qmn)

  # eigenvalue as a function of k_y
  if l == 0:
    eng_k = energy
  else:
    eng_k = np.vstack((eng_k,energy))

np.savetxt('./data/eng-matrix.txt', eng_k, fmt='%1.8f')
