#!/usr/bin/python3

import numpy as np

PBC=True
t = 1
#delta = t*1.1
delta = t
mu = 1.4

a = 1
nr = 60
n = 3*nr
B = -3*np.pi / (a**2 * (nr+1))
#B = 0.0
B=np.pi/25
B = 0.156*np.pi


x = np.array([a*(i - (nr-1)/2) for i in range(nr)])
xx = (a**2 + 2*x[:-1]) / 2
phase = np.ones(nr)
phase = np.append(phase, np.exp(1.0j*B*xx))
phase = np.append(phase, np.ones(nr))

delarr = delta*np.ones(nr)
delarr = np.append(delarr,delta*np.ones(nr-1))
delarr = np.append(delarr,delta*(np.ones(nr)))

bdg = np.zeros((2*n,2*n), dtype='complex')
bdg[0:n, 0:n] = -mu*np.eye(n) - t*np.diag(phase,k=-1)
bdg[n:2*n, n:2*n] = mu*np.eye(n) + t*np.conjugate(np.diag(phase, k=-1))
bdg[n:2*n, 0:n] = np.diag(delarr, k=1) - np.diag(delarr,k=-1)

# Now n-1 to 0 lattice point (PBC)
if(PBC):
  bdg[n-1,0]   = -t
  bdg[2*n-1,n] = t
  bdg[2*n-1,0]   = delta
  bdg[n,n-1] = -delta

np.savetxt('./data/bdg.txt', bdg, fmt='%1.1f')
eng, vec = np.linalg.eigh(bdg)
#vec = vec[0:n,0:n]
#vec = np.real( np.matmul(vec.conj().T,vec) )
vec = np.real(np.multiply(vec,np.conj(vec)))
eng = np.real(eng)
np.savetxt('./data/double-chain-energy.txt', eng, fmt='%1.8e')
np.savetxt('./data/double-chain-states.txt', vec, fmt='%1.8e')
