#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
pi = np.pi

# Flags
pbc = True
corners = True

# Define parameters
t = 1
delta = t
mu = 1.1*t
a = 1
nl = 24
n = 3*(nl+1)

# Set vector potential field strengths
nA = 1
A0 = 0
Af = 2.25
A = np.linspace(Af,Af,nA)

# Set vector potential rotation angles
nvphi = 1*15
vphi0 = 0
vphif = pi
vphi = np.linspace(vphi0, vphif, nvphi)

H = np.zeros((2*n,2*n), dtype = 'complex')
lc = np.ones(nl)
brc = np.ones(nl+1)
wc = np.ones(n)

phil = 2*pi/3
phib = 0
phir = -2*pi/3
phi1 = pi/3
phi2 = -pi/3
phi3 = 0

dlc = delta*np.exp(-1.0j*-2*pi/3) * lc
dbc = delta * brc
drc = delta*np.exp(-1.0j*-pi/3) * brc
darr = np.append(dlc, np.append(dbc, drc))
h12 = np.diag(darr, k=1)

if(pbc):
  h12[0:-1] = -delta*np.exp(-1.0j*-2*pi/3)
if(corners):
  h12[nl-1,nl+1] = delta*np.exp(-1.0j*-pi/3)
  h12[2*nl,2*nl+2] = delta*np.exp(-1.0j*pi/3)
  h12[3*nl+1,0] = delta*np.exp(-1.0j*pi/3)
h12 -= h12.T

nE = 1*4
eva = np.zeros((2*nE,nvphi))
wf = np.zeros((n,nvphi))

for j, avals in enumerate(A):
  for l, vphivals in enumerate(vphi):
    tlc = -t*np.exp(1.0j * -avals * np.sin(phil + vphivals)) * lc
    tbc = -t*np.exp(1.0j * -avals * np.sin(phib + vphivals)) * brc
    trc = -t*np.exp(1.0j * -avals * np.sin(phir + vphivals)) * brc
    tarr = np.append(tlc, np.append(tbc, trc))
    h11 = -mu*np.diag(wc,k=0) + np.diag(tarr,k=1)
    if(pbc):
      h11[0:-1] = -t*np.exp(-1.0j * -avals * np.sin(phir + vphivals))
    if(corners):
      h11[nl-1,nl+1]   = -t*np.exp(1.0j * -avals * np.sin(phi1 + vphivals))
      h11[2*nl,2*nl+2] = -t*np.exp(1.0j * -avals * np.sin(phi2 + vphivals))
      h11[3*nl+1,0]    = -t*np.exp(1.0j * -avals * np.sin(phi3 + vphivals))

    H[0:n,0:n] = h11
    H[0:n,n:2*n] = h12
    H[n:2*n,n:2*n] = -h11.conj()

    eng, vec = np.linalg.eigh(H, UPLO = 'U')
    eva[:,l] = eng[n-nE:n+nE]
    vec = np.real(np.multiply(vec, vec.conj()))
    wf[:,l] = 1*(vec[0:n,n] + vec[n:2*n,n]) + 1*(vec[0:n,n-1] + vec[n:2*n,n-1])


plt.figure()
for i in range(2*nE):
  plt.plot(vphi, eva[i,:], 'C0')
plt.show()
x = np.linspace(0,n,n)
for i in range(nvphi):
  plt.plot(x+i,0*wf[:,i]+0.1*i, 'k:')
  plt.plot(x+i,wf[:,i]+0.1*i)
plt.show()
plt.close()


