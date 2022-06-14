#!/usr/bin/python3

import numpy as np

t = 1
#delta = t*1.1
delta = t
mu = -2.5

a = 1
nr = 50
n = 3*(nr-1)

bdg = np.zeros((2*n,2*n), dtype='complex')
pm = np.exp(-1.0j*np.pi/2)
pp = np.exp(+1.0j*np.pi/2)

for i in range(nr-1):
  bdg[i,i] = -mu
  bdg[i+n,i+n] = +mu
  bdg[i+1,i] = -t*pm
  bdg[i,i+1] = -t*pp
  bdg[i+1+n,i+n] = t*pp
  bdg[i+n,i+1+n] = t*pm
  bdg[i+1,i+n]  = delta
  bdg[i+n,i+1]  = delta
  bdg[i+1+n,i] = -delta
  bdg[i,i+1+n] = -delta

for i in range(nr-1):
  j = i+nr-1
  bdg[j,j] = -mu
  bdg[j+n,j+n] = +mu
  bdg[j+1,j] = -t*pm
  bdg[j,j+1] = -t*pp
  bdg[j+1+n,j+n] = t*pp
  bdg[j+n,j+1+n] = t*pm
  bdg[j+1,j+n] = delta
  bdg[j+n,j+1] = delta
  bdg[j+1+n,j] = -delta
  bdg[j,j+1+n] = -delta

for i in range(nr-2):
  j = i+2*(nr-1)
  bdg[j,j] = -mu
  bdg[j+n,j+n] = +mu
  bdg[j+1,j] = -t*pm
  bdg[j,j+1] = -t*pp
  bdg[j+1+n,j+n] = t*pp
  bdg[j+n,j+1+n] = t*pm
  bdg[j+1+n,j+n] = delta
  bdg[j+n,j+1+n] = delta
  bdg[j+1+n,j]  = -delta
  bdg[j,j+1+n]  = -delta

# Now n-1 to 0 lattice point (PBC)
bdg[n-1,n-1] = -mu
bdg[2*n-1,2*n-1] = +mu
bdg[0,n-1]   = -t*pp
bdg[n-1,0]   = -t*pm
bdg[n,2*n-1] = t*pm
bdg[2*n-1,n] = t*pp
bdg[0,2*n-1]   = delta*pp
bdg[2*n-1,0]   = delta*pm
bdg[n,n-1] = -delta*pm
bdg[n-1,n] = -delta*pp

np.savetxt('./data/bdg.txt', bdg, fmt='%1.1f')
eng, vec = np.linalg.eigh(bdg)
#vec = vec[0:n,0:n]
#vec = np.real( np.matmul(vec.conj().T,vec) )
vec = np.real(np.multiply(vec,np.conj(vec)))
eng = np.real(eng)
np.savetxt('./data/kitaev-triangle-chain-energy.txt', eng, fmt='%1.8e')
np.savetxt('./data/kitaev-triangle-chain-states.txt', vec, fmt='%1.8e')
