#!/usr/bin/python3

import numpy as np
from pfapack import pfaffian as pf
import matplotlib.pyplot as plt

pi = np.pi

t = 1.0
delta = t
mu = -2.50*t
a = 1
w = 3
nA = 4*90
nphi = nA

A0 = 0*pi
Af = 2*pi / (np.sqrt(3)*a)
Af = 1.0*pi

d1 = a
d2 = a/2
d3 = -a/2

th1 = 0
th2 = pi/3
th3 = 2*pi/3

def ph1(_A, _ang):
  Ax = -_A * np.sin(_ang)
  return Ax * a

def ph2(_A, _ang):
  Ax = -_A * np.sin(_ang)
  Ay = _A * np.cos(_ang)
  return -Ax * a / 2 - Ay * np.sqrt(3) * a / 2

def ph3(_A, _ang):
  Ax = -_A * np.sin(_ang)
  Ay = _A * np.cos(_ang)
  return Ax * a / 2 - Ay * np.sqrt(3) * a / 2

def epsdiag0(_mu, _k, _A, _ang):
  return -2 * t * np.cos( ( _k * a + ph1(_A,_ang) ) ) - _mu

def epsdiag1(_k, _A, _ang):
  return -t * ( np.exp(-1.0j * ph2(_A, _ang) ) + np.exp(1.0j * ( _k * a - ph3(_A,_ang) ) ) )

def epsdiagm1(_k, _A, _ang):
  return -t * ( np.exp(1.0j * ph2(_A, _ang) ) + np.exp(-1.0j * ( _k * a - ph3(_A,_ang) ) ) )

def deldiag0(_k):
  return 2.0j * delta * np.sin(_k * a) * np.exp(1.0j * th1)

def deldiag1(_k):
  return -delta * ( np.exp(1.0j * th2) + np.exp(1.0j * (_k * a + th3)) )

def deldiagm1(_k):
  return delta * ( np.exp(1.0j * th2) + np.exp(1.0j * (-_k * a + th3)) )

Avalues = np.linspace(A0,Af,nA)
phivalues = np.linspace(0,pi,nphi)
kvalues = [0, pi/a]
H0 = np.zeros((2,2*w,2*w), dtype='complex')
W0 = np.zeros((2,2*w,2*w), dtype='complex')

U = np.sqrt(0.5) * np.matrix([[1,1],[-1.0j,1.0j]])
U = np.kron(U, np.identity(w))
MN0 = np.zeros((2, nA, nphi))

for l, phivals in enumerate(phivalues):
  for j, avals in enumerate(Avalues):
    for k, kvals in enumerate(kvalues):
      h11 = epsdiag0(mu, kvals, avals, phivals)*np.diag(np.ones(w),k=0) + epsdiag1(kvals, avals, phivals)*np.diag(np.ones(w-1),k=1) + epsdiagm1(kvals, avals, phivals)*np.diag(np.ones(w-1),k=-1)
      h22 = epsdiag0(mu, -kvals, avals, phivals)*np.diag(np.ones(w),k=0) + epsdiag1(-kvals, avals, phivals)*np.diag(np.ones(w-1),k=1) + epsdiagm1(-kvals, avals, phivals)*np.diag(np.ones(w-1),k=-1)
      h12 = deldiag0(kvals)*np.diag(np.ones(w),k=0) + deldiag1(kvals)*np.diag(np.ones(w-1),k=1) + deldiagm1(kvals)*np.diag(np.ones(w-1),k=-1)
      #h21 = deldiag0(-kvals)*np.diag(np.ones(w),k=0) + deldiag1(-kvals)*np.diag(np.ones(w-1),k=1) + deldiagm1(-kvals)*np.diag(np.ones(w-1),k=-1)
      H0[k,0:w,0:w] = h11
      H0[k,0:w,w:2*w] = h12
      H0[k,w:2*w,0:w] = h12.T.conj()
      H0[k,w:2*w,w:2*w] = -h22.T

      W0[k,:,:] = -1.0j * U * H0[k,:,:] * np.conjugate(np.transpose(U))
      #W0[k,:,:] = (W0[k,:,:] - W0[k,:,:].T)/2

      MN0[k,j,l] = np.sign(np.real(pf.pfaffian(W0[k,:,:])))

#plt.figure()
TP = MN0[0,:,:]*MN0[1,:,:]
#TP = (TP+0.8)**3
absmu = abs(mu)
mustring = f"{absmu:1.4f}"
mustring = mustring.replace('.','_')
if(mu >= 0):
  mustring = 'p' + mustring
else:
  mustring = 'n' + mustring
np.savetxt('./data/majorana-number-w-%01i-mu-%s.txt' % (w, mustring) , np.flipud(TP), fmt='%1.6f')

