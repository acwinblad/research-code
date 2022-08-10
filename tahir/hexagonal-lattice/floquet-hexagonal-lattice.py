#!/usr/bin/python3

import numpy as np
import scipy.special as sp
import os
import glob

np.set_printoptions(linewidth=np.inf, precision=2)

files = glob.glob('./data/eig*')
for f in files:
  os.remove(f)


# build global arrays for fourier transform
tau = np.linspace(0, 2*np.pi, 51)
s1 = np.sin(tau)
c2 = np.cos(2*tau)

# number of modes
mc = 3
Nm = 2*mc+1
marr = np.arange(0, Nm, 1) - mc

# radius
rc = 15
Ns = 2*rc+1
rarr = np.arange(0, Ns, 1) - rc

# constants
hbar = 6.582E-16 # 6.582 * 10^-16 eV*s
c = 2.998E* # 2.998 * 10^8 m/s
m_e = 0.51E6 / c**2 # 0.51 MeV/c^2
ec = 1.602E-19 # C

# incoming light
hw = 191E-3 # meV
ka = 0.1
a = 100E-9 # nm
t = hbar**2 / (2 * a**2 * m_e)

E = 2E8 # V/m
alpha = ka / (8*np.pi*hw**2)
nphi = 100
phimin = 0.0
phimax = 0.15 * 1e-1
phiE = np.array( [ (phimin + i/nphi)**(1/3) for i in range(nphi) ] ) * phimax

print(t)
print(alpha)
print(phimax)
print(alpha*phimax**2)
print('B =', )

jjdifx = np.array([np.sqrt(3)/2, 0, np.sqrt(3)/2])
jjavgx = np.array([np.sqrt(3)/4, np.sqrt(3)/2, 3*np.sqrt(3)/4])
jjdify = np.array([-1/2, -1, -1/2])
jp1difx = jjdifx
jp1avgx = jjavgx
jp1dify = np.array([1/2, -1, 1/2])
jm1difx = jjdifx
jm1avgx = jjavgx
jm1dify = np.array([-1/2, 1, -1/2])

def hblock(_p, _n):
  hnblock = np.zeros( [4*Ns, 4*Ns], "complex" )

  for ridx, rval in enumerate(rarr):
    tmp = np.outer([s1,],[jjdifx,]) + 0.5*np.outer([c2,],[jjdify*np.sin(ka*(rval*np.sqrt(3)+jjavgx)),])
    tmp = np.exp(-1.0j * ( _p*tmp + _n*np.outer([tau,],[np.ones(3),]) ) )
    tmp = -np.trapz(tmp, x=tau, axis=0) / (2*np.pi)
    hnblock[4*ridx:4*(ridx+1),4*ridx:4*(ridx+1)] = np.diag(tmp, k=1) + np.diag(tmp.conjugate(), k=-1)

  for ridx, rval in enumerate(rarr[1:]):
    tmp = np.outer([s1,],[jp1difx,]) + 0.5*np.outer([c2,],[jp1dify*np.sin(ka*((2*rval+1)*np.sqrt(3)/2+jp1avgx)),])
    tmp = np.exp(-1.0j * ( _p*tmp + _n*np.outer([tau,],[np.ones(3),]) ) )
    tmp = -np.trapz(tmp, x=tau, axis=0) / (2*np.pi)

    Hjp1 = np.reshape( [0,0,0,0, tmp[0],0,0,0,  0,0,0,0, tmp[1],0,tmp[2],0], (4,4) )
    hnblock[4*ridx:4*(ridx+1), 4*(ridx+1):4*(ridx+2)] = Hjp1
    hnblock[4*(ridx+1):4*(ridx+2), 4*ridx:4*(ridx+1)] = Hjp1.conj().T

#    tmp = -np.outer([s1,],[jm1difx,]) + 0.5*np.outer([c2,],[jm1dify*np.sin(ka*((2*rval-1)*np.sqrt(3)/2+jm1avgx)),])
#    tmp = np.exp(-1.0j * ( _p*tmp + _n*np.outer([tau,],[np.ones(3),]) ) )
#    tmp = -np.trapz(tmp, x=tau, axis=0) / (2*np.pi)
#    hnblock[4*(ridx+1):4*(ridx+2), 4*ridx:4*(ridx+1)] = np.transpose( np.reshape( [0,0,0,0, tmp[0],0,0,0,  0,0,0,0, tmp[1],0,tmp[2],0], (4,4) ) )

  return hnblock

## Calculating Floquet data set
#energy = np.zeros( (Nm*Ns, nphi) )
#for pidx, pval in enumerate(phiE):
#  Qmn = np.zeros( [4*Nm*Ns, 4*Nm*Ns], "complex" )
#  for midx, mval in enumerate(marr):
#    Hn = hblock(pval, midx)

energy = np.zeros( (4*Nm*Ns, nphi) )
for phiidx, phival in enumerate(phiE):
  # Build H_ijn matrices as a 3D matrix
  Hn = np.zeros( [Nm, 4*Ns, 4*Ns], "complex")
  Qmn = np.zeros( [4*Nm*Ns, 4*Nm*Ns], "complex" )

  for nidx, nval in enumerate(marr):
    Hn[nidx, :,:] = hblock(phival, -nidx)

  for i in range(Nm):
    for j in range(Nm-i):
      r1 = (i+j)*4*Ns
      r2 = (i+j+1)*4*Ns
      c1 = j*4*Ns
      c2 = (j+1)*4*Ns
      if i ==0:
        Qmn[r1:r2, r1:r2] = Hn[i,:,:] - marr[j]*hw*np.eye(4*Ns)
      else:
        Qmn[r1:r2, c1:c2] = Hn[i,:,:]

  energy[:,phiidx], states = np.linalg.eigh(Qmn)

  np.savetxt('./data/eigenstate-phi-%03i.txt' % (phiidx), states, fmt = '%1.8f')

np.savetxt('./data/eng-matrix.txt', energy, fmt = '%1.8f')

np.savetxt('./data/config-floquet.txt', [rc, mc, alpha, phimin, phimax, nphi])
