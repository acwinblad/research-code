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
mc = 5
Nm = 2*mc+1
marr = np.arange(0, Nm, 1) - mc

# radius
rc = 5
Ns = 2*rc+1
rarr = np.arange(0, Ns, 1) - rc

# constants
hbar = 6.582E-16 # 6.582 * 10^-16 eV*s
c = 2.998E8 # 2.998 * 10^8 m/s
ec = 1.602E-19 # C
vf = 1E6 # fermi velocity 10^6 m/s

# effective mass of electron
m_e = 0.51E6 / c**2 # 0.51 MeV/c^2
m = 1.0 * m_e

# incoming light and wavenumber
hw = 191E-3 # meV
d = 100E-9
K = 2*np.pi/d

# lattice constant
a = 0.142E-9 # nm

k = 0
ka = k*a

# hopping parameters in eV
#t = ( 2 * hbar * vf ) / ( 3 * a )
t = 2.8
print(t)
Emax = 3E7 # V/m
phimax = Emax*a/hw
C = 1 - ( vf * hbar * phimax ) / ( a * hw )
B = ( K * hbar**3 * vf**2 * phimax**2 ) / ( 4 * a**3 * hw**2 * C )
alpha = ( 3*np.sqrt(3) * K * hbar**2 * vf**2 ) / ( 8 * a * hw**2 * C )
phimin = 0.0
nphi = 100
phi0 = np.array( [ (phimin + i/nphi)**(1/1) for i in range(nphi) ] ) * phimax
phi0 = np.linspace(phimin,phimax,nphi)

print('m =', m)
print('K =', K)
print('ka= ', ka)
print('t= ', t)
print('B= ', B)
print('alpha= ', alpha)
print('phi= ', phimax)
print('phi_b= ', alpha*phimax**3)

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
    tmp = np.outer([s1,],[jjdifx,]) + 0.5*np.outer([c2,],[jjdify*np.sin(K*a*(rval*np.sqrt(3)+jjavgx)),])
    tmp = np.exp(-1.0j * ( _p*tmp + _n*np.outer([tau,],[np.ones(3),]) ) )
    tmp = -t*np.trapz(tmp, x=tau, axis=0) / (2*np.pi)
    hnblock[4*ridx:4*(ridx+1),4*ridx:4*(ridx+1)] = np.diag(tmp, k=1) + np.diag(tmp.conjugate(), k=-1)

  for ridx, rval in enumerate(rarr[1:]):
    tmp = np.outer([s1,],[jp1difx,]) + 0.5*np.outer([c2,],[jp1dify*np.sin(K*a*((2*rval+1)*np.sqrt(3)/2+jp1avgx)),])
    tmp = np.exp(-1.0j * ( _p*tmp + _n*np.outer([tau,],[np.ones(3),]) ) )
    tmp = -t*np.trapz(tmp, x=tau, axis=0) / (2*np.pi)

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
for phiidx, phival in enumerate(phi0):
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

np.savetxt('./data/config-floquet.txt', [rc, mc, t, alpha, phimin, phimax, nphi])
