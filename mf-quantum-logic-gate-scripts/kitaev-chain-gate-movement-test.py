#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt


t = 1
delta = t

n0 = 50
n1 = 100
ng = 1
nR = ng*n1
n = n0+nR

hbdg = np.zeros((2*n,2*n))
hbdg[0:n,0:n] = -t * np.diag(np.ones(n-1),k=1)
hbdg[n:2*n,n:2*n] = t * np.diag(np.ones(n-1),k=1)
hbdg[0:n,n:2*n] = delta * (np.diag(np.ones(n-1),k=1) - np.diag(np.ones(n-1),k=-1))

#for j in range(n0):
#  hbdg[j,j] = -mu0
#  hbdg[n+j,n+j] = mu0

nmu = 5
mui = 2.4*t
muf = 1.6*t
mu0 = 1.6*t
muarr = np.linspace(mui,muf,nmu)

engarr = np.zeros((2*n,nmu*ng))
vec1arr = np.zeros((n,nmu*ng))
vec2arr = np.zeros((n,nmu*ng))
vec3arr = np.zeros((n,nmu*ng))
vec4arr = np.zeros((n,nmu*ng))

for j in range(n0):
  hbdg[j,j] = -mu0
  hbdg[n+j,n+j] = mu0

for j in range(nR):
  hbdg[n0+j,n0+j] = -mui
  hbdg[n+n0+j,n+n0+j] = mui

for j in range(ng):
  for l, val in enumerate(muarr):
    for k in range(n1):
      hbdg[n0+j*n1+k, n0+j*n1+k] = -val
      hbdg[n+n0+j*n1+k, n+n0+j*n1+k] = val

    eng, vec = np.linalg.eigh(hbdg, UPLO = 'U')
      #engarr[:, j*nmu + l] = eng
    engarr[:, j*nmu + l] = eng
    vec = np.real(np.multiply(vec, np.conj(vec)))
    vec1arr[:,j*nmu + l] = vec[0:n,n-1] + vec[n:2*n,n-1]
    vec2arr[:,j*nmu + l] = vec[0:n,n] + vec[n:2*n,n]
    vec3arr[:,j*nmu + l] = vec[0:n,n+1] + vec[n:2*n,n+1]
    vec4arr[:,j*nmu + l] = vec[0:n,n+1] + vec[n:2*n,n+1]

plt.figure()
for j in range(ng):
  ji = j*nmu
  jf = (j+1)*nmu
  x = muarr + j*0.5
  plt.plot(x,engarr[n-1,ji:jf])
  plt.plot(x,engarr[n-2,ji:jf])
  plt.plot(x,engarr[n-3,ji:jf])
  plt.plot(x,engarr[n-4,ji:jf])
plt.ylim(-0.001,0.001)
plt.show()
plt.savefig('./kitaev-chain-gate-movement-band-gap.pdf')
plt.close()

plt.figure()
for j in range(np.max(1,ng//2)):
  for l, val in enumerate(muarr):
    ll = j*nmu + l
    wf = vec1arr[:,ll]+vec2arr[:,ll]+0.13*ll
    wf2 = vec3arr[:,ll]+vec4arr[:,ll]+0.13*ll
    xarr = np.linspace(0,n,n) + ll*n*0.05
    plt.plot(xarr,0.13*ll*np.ones(n), 'k--', linewidth=1.0, zorder=-ll)
    plt.plot(xarr,wf, zorder = -ll)
    plt.plot(xarr,wf2, zorder = -ll-1)

#plt.show()
plt.savefig('./kitaev-chain-gate-movement-wavefunction.pdf')
plt.close()
