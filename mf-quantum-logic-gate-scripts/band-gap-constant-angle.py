#!/usr/bin/python3

import numpy as np
from pfapack import pfaffian as pf
import matplotlib.pyplot as plt

pi = np.pi

t = 1.0
delta = t
a = 1
w = 3
nA = 1*15
nmu = nA
Af = 4*pi / (np.sqrt(3)*a)
Af = 1*pi
mui = -3.5*t
muf = -2.0*t

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

Avalues = np.linspace(0,Af,nA)
muvalues = np.linspace(mui,muf,nmu)
nk = 250 # Must be even
kvalues = np.linspace(-pi/a,pi/a,nk+1)
H0 = np.zeros((2*w,2*w), dtype='complex')
evk = np.zeros((nk+1))
bg = np.zeros((nmu,nA))
ang0 = 2*pi/6


plt.figure()
for j, avals in enumerate(Avalues):
  for l, muvals in enumerate(muvalues):
    for k, kvals in enumerate(kvalues):
      h11 = epsdiag0(muvals, kvals, avals, ang0)*np.diag(np.ones(w),k=0) + epsdiag1(kvals, avals, ang0)*np.diag(np.ones(w-1),k=1) + epsdiagm1(kvals, avals, ang0)*np.diag(np.ones(w-1),k=-1)
      h22 = epsdiag0(muvals, -kvals, avals, ang0)*np.diag(np.ones(w),k=0) + epsdiag1(-kvals, avals, ang0)*np.diag(np.ones(w-1),k=1) + epsdiagm1(-kvals, avals, ang0)*np.diag(np.ones(w-1),k=-1)
      h12 = deldiag0(kvals)*np.diag(np.ones(w),k=0) + deldiag1(kvals)*np.diag(np.ones(w-1),k=1) + deldiagm1(kvals)*np.diag(np.ones(w-1),k=-1)
      #h21 = deldiag0(-kvals)*np.diag(np.ones(w),k=0) + deldiag1(-kvals)*np.diag(np.ones(w-1),k=1) + deldiagm1(-kvals)*np.diag(np.ones(w-1),k=-1)
      H0[0:w,0:w] = h11
      H0[0:w,w:2*w] = h12
      H0[w:2*w,0:w] = h12.T.conj()
      H0[w:2*w,w:2*w] = -h22.T

      eng = np.linalg.eigvalsh(H0, UPLO = 'U')
      evk[k] = np.min(np.abs(eng))
      #evk[k] = eng[w]
    bg[l,j] = np.min(evk)
    #plt.plot(evk)

#plt.show()
plt.close()
np.savetxt('./data/band-gap-constant-angle-w-%01i.txt' % (w) , np.flipud(bg), fmt='%1.8f')

