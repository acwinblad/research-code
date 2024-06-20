#!/usr/bin/python3

import numpy as np
from pfapack import pfaffian as pf
import matplotlib.pyplot as plt

pi = np.pi

t = 1.0
delta = t
a = 1
w = 3
nA = 4*90
nmu = nA
Af = 4*pi / (np.sqrt(3)*a)
#Af = 4*pi
mui = -5
muf = 3

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
muvalues = np.linspace(muf,mui,nmu)
kvalues = [0, pi/a]
H0 = np.zeros((2,2*w,2*w), dtype='complex')
W0 = np.zeros((2,2*w,2*w), dtype='complex')
H1pi3 = np.zeros((2,2*w,2*w), dtype='complex')
W1pi3 = np.zeros((2,2*w,2*w), dtype='complex')
ang0 = 0
ang1 = 1*pi/3

U = np.sqrt(0.5) * np.matrix([[1,1],[-1.0j,1.0j]])
U = np.kron(U, np.identity(w))
MN0 = np.zeros((2, nmu, nA))
MN1pi3 = np.zeros((2, nmu, nA))

for j, avals in enumerate(Avalues):
  for l, muvals in enumerate(muvalues):
    for k, kvals in enumerate(kvalues):
      h11 = epsdiag0(muvals, kvals, avals, ang0)*np.diag(np.ones(w),k=0) + epsdiag1(kvals, avals, ang0)*np.diag(np.ones(w-1),k=1) + epsdiagm1(kvals, avals, ang0)*np.diag(np.ones(w-1),k=-1)
      h22 = epsdiag0(muvals, -kvals, avals, ang0)*np.diag(np.ones(w),k=0) + epsdiag1(-kvals, avals, ang0)*np.diag(np.ones(w-1),k=1) + epsdiagm1(-kvals, avals, ang0)*np.diag(np.ones(w-1),k=-1)
      h12 = deldiag0(kvals)*np.diag(np.ones(w),k=0) + deldiag1(kvals)*np.diag(np.ones(w-1),k=1) + deldiagm1(kvals)*np.diag(np.ones(w-1),k=-1)
      h21 = deldiag0(-kvals)*np.diag(np.ones(w),k=0) + deldiag1(-kvals)*np.diag(np.ones(w-1),k=1) + deldiagm1(-kvals)*np.diag(np.ones(w-1),k=-1)
      H0[k,0:w,0:w] = h11
      H0[k,0:w,w:2*w] = h12
      H0[k,w:2*w,0:w] = h12.T.conj()
      H0[k,w:2*w,w:2*w] = -h22.T


      g11 = epsdiag0(muvals, kvals, avals, ang1)*np.diag(np.ones(w),k=0) + epsdiag1(kvals, avals, ang1)*np.diag(np.ones(w-1),k=1) + epsdiagm1(kvals, avals, ang1)*np.diag(np.ones(w-1),k=-1)
      g22 = epsdiag0(muvals, -kvals, avals, ang1)*np.diag(np.ones(w),k=0) + epsdiag1(-kvals, avals, ang1)*np.diag(np.ones(w-1),k=1) + epsdiagm1(-kvals, avals, ang1)*np.diag(np.ones(w-1),k=-1)
      H1pi3[k,0:w,0:w] = g11
      H1pi3[k,0:w,w:2*w] = h12
      H1pi3[k,w:2*w,0:w] = h12.T.conj()
      H1pi3[k,w:2*w,w:2*w] = -g22.T

      W0[k,:,:] = -1.0j * U * H0[k,:,:] * np.conjugate(np.transpose(U))
      W1pi3[k,:,:] = -1.0j * U * H1pi3[k,:,:] * np.conjugate(np.transpose(U))
      #W0[k,:,:] = (W0[k,:,:] - W0[k,:,:].T)/2
      #W1pi3[k,:,:] = (W1pi3[k,:,:] - W1pi3[k,:,:].T)/2

      MN0[k,l,j] = np.sign(np.real(pf.pfaffian(W0[k,:,:])))
      MN1pi3[k,l,j] = np.sign(np.real(pf.pfaffian(W1pi3[k,:,:])))

MN1 = MN0[0,:,:]*MN0[1,:,:]
MN2 = MN1pi3[0,:,:]*MN1pi3[1,:,:]
bpb = -(MN1-1)//2
bpt = -(MN2-1)//2
BP = 1*bpb+1*bpt*2
MN1 = (MN1+1.1)**3
MN2 = (MN1pi3+1.1)**3
TP = MN1+MN2
np.savetxt('./data/majorana-number-1pi3-w-%01i.txt' % (w) , BP, fmt='%1d')

