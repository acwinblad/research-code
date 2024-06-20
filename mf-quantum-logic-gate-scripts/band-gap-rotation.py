#!/usr/bin/python3

import numpy as np
from pfapack import pfaffian as pf
import matplotlib.pyplot as plt

pi = np.pi

t = 1.0
delta = t
a = 1
w = 3
nA = 4*45
nvarphi = nA
Af = 4*pi / (np.sqrt(3)*a)
A0 = 0*pi
Af = 1*pi
mu = -2.5*t

d1 = a
d2 = a/2
d3 = -a/2

th1 = 0
th2 = pi/3
th3 = 2*pi/3

def ph1(_A, _ang):
  Ax = -_A * np.sin(_ang)
  return -Ax * a

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
varphivalues = np.linspace(0,pi,nvarphi)
nk = 90 # Must be even
kvalues = np.linspace(0,pi/a,nk+1)
H0 = np.zeros((2*w,2*w), dtype='complex')
evk = np.zeros((nk+1))
bg = np.zeros((nA,nvarphi))


for j, avals in enumerate(Avalues):
  for l, varphivals in enumerate(varphivalues):
    for k, kvals in enumerate(kvalues):
      h11 = epsdiag0(mu, kvals, avals, varphivals)*np.diag(np.ones(w),k=0) + epsdiag1(kvals, avals, varphivals)*np.diag(np.ones(w-1),k=1) + epsdiagm1(kvals, avals, varphivals)*np.diag(np.ones(w-1),k=-1)
      h22 = epsdiag0(mu, -kvals, avals, varphivals)*np.diag(np.ones(w),k=0) + epsdiag1(-kvals, avals, varphivals)*np.diag(np.ones(w-1),k=1) + epsdiagm1(-kvals, avals, varphivals)*np.diag(np.ones(w-1),k=-1)
      h12 = deldiag0(kvals)*np.diag(np.ones(w),k=0) + deldiag1(kvals)*np.diag(np.ones(w-1),k=1) + deldiagm1(kvals)*np.diag(np.ones(w-1),k=-1)
      #h21 = deldiag0(-kvals)*np.diag(np.ones(w),k=0) + deldiag1(-kvals)*np.diag(np.ones(w-1),k=1) + deldiagm1(-kvals)*np.diag(np.ones(w-1),k=-1)
      H0[0:w,0:w] = h11
      H0[0:w,w:2*w] = h12
      #H0[w:2*w,0:w] = h12.T.conj()
      H0[w:2*w,w:2*w] = -h22.T

      eng = np.linalg.eigvalsh(H0, UPLO = 'U')
      evk[k] = np.min(np.abs(eng))
    bg[j,l] = np.min(evk)

absmu = abs(mu)
mustring = f"{absmu:1.4f}"
mustring = mustring.replace('.','_')
if(mu >= 0):
  mustring = 'p' + mustring
else:
  mustring = 'n' + mustring

np.savetxt('./data/band-gap-rotation-w-%01i-mu-%s.txt' % (w, mustring) , np.flipud(bg), fmt='%1.8f')

