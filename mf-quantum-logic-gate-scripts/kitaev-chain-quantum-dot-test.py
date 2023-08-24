#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

# Initialize parameters

t = 1
delta = t
mul = 0.0*t

# Quantum dots potential

V = 2.5*t

# Prepare tuning of chemical potential on right segment

nmu = 15
mui = -3.0*t
muf = -1.0*t
muarr = np.linspace(mui,muf,nmu)

# Size of left kitaev chain, quantum dot, and right kitaev chain
nl = 0
nqd = 30
ndx = 0
nr = 50 * 2**2
n = nl + nqd + nr

# Initialize hopping and pairing potentials

hbdg = np.zeros((2*n,2*n))
hbdg[0:n,0:n] = -t * np.diag(np.ones(n-1),k=1)
hbdg[n:2*n,n:2*n] = t * np.diag(np.ones(n-1),k=1)

for j in range(nl-1):
  l=j+1
  hbdg[j, n+l] = delta
  hbdg[l, n+j] = -delta

p2 = nl + nqd
for j in range(ndx+nr-1):
  l=j+1
  hbdg[p2-ndx + j, p2-ndx + n + l] = delta
  hbdg[p2-ndx + l, p2-ndx + n + j] = -delta

# Prepare empty arrays for energy spectral flow and wavefunctions

engarr = np.zeros((2*n,nmu))
vec1arr = np.zeros((n,nmu))
vec2arr = np.zeros((n,nmu))
vec3arr = np.zeros((n,nmu))
vec4arr = np.zeros((n,nmu))

# Fill out initial chemical potentials and potential wells for qd
for j in range(nl):
  hbdg[j,j] = -mul
  hbdg[n + j,n + j] = mul

#print(hbdg[0:n,0:n])
#print(hbdg[0:n,n:2*n])

for j, val in enumerate(muarr):
  mugrad = np.linspace(mul,val,nqd)
  for l in range(nqd):
    muavg = (val+val)/2
    #hbdg[nl + l,nl + l] = -mugrad[l] + V
    #hbdg[n + nl + l,n + nl + l] = mugrad[l] - V
    hbdg[nl + l,nl + l] = -muavg + V
    hbdg[n + nl + l,n + nl + l] = muavg - V
  for l in range(nr):
    hbdg[nl + nqd + l,nl + nqd + l] = -val
    hbdg[n + nl + nqd + l,n + nl + nqd + l] = val

  eng, vec = np.linalg.eigh(hbdg, UPLO = 'U')

  # Store the energy and four lowest wavefunctions

  engarr[:,j] = eng
  vec = np.real(np.multiply(vec, np.conj(vec)))
  vec1arr[:,j] = vec[0:n,n-1] + vec[n:2*n,n-1]
  vec2arr[:,j] = vec[0:n,n] + vec[n:2*n,n]
  vec3arr[:,j] = vec[0:n,n+1] + vec[n:2*n,n+1]
  vec4arr[:,j] = vec[0:n,n+1] + vec[n:2*n,n+1]

# Plot the energy spectral flow

plt.figure()

nE = 15
for i in range(2*nE):
  plt.plot(muarr,engarr[n-nE+i,:])
plt.ylim(-0.2,0.2)
plt.show()
#plt.savefig('./data/figures/kitaev-chain-quantum-dot-band-gap-nr-'+str(nr)+'.pdf')
plt.close()

# Plot the four lowest energy wavefunctions

plt.figure()
for l, val in enumerate(muarr):
  lx = l * n * 0.05
  ly = 0.13*l
  wf = vec1arr[:,l] + ly
  wf2 = vec2arr[:,l] + ly
  wf3 = vec3arr[:,l] + ly
  wf4 = vec4arr[:,l] + ly
  xarr = np.linspace(0,n,n) + lx
  vb1x = nl + lx
  vb1x = [vb1x, vb1x]
  vb2x = nl + nqd + lx
  vb2x = [vb2x, vb2x]
  vby = [ly-0.02, ly+0.08]
  plt.plot(vb1x,vby, 'k--', linewidth=1.0, zorder=-l)
  plt.plot(vb2x,vby, 'k--', linewidth=1.0, zorder=-l)
  plt.plot(xarr,ly*np.ones(n), 'k--', linewidth=1.0, zorder=-l)
  plt.plot(xarr+3,wf4, 'C3', lw=0.6, zorder = -l)
  plt.plot(xarr+2,wf3, 'C2', lw=0.6, zorder = -l)
  plt.plot(xarr+1,wf2, 'C1', lw=0.6, zorder = -l)
  plt.plot(xarr,wf, 'C0',    lw=0.6, zorder = -l)

#plt.show()
plt.savefig('./data/figures/kitaev-chain-quantum-dot-wavefunction-nr-' + str(nr) + '.pdf')
plt.close()
