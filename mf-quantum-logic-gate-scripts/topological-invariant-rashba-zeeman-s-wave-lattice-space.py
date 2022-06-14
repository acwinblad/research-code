#!/usr/bin/python3

import numpy as np

t = 1
#delta = t*1.1
delta = t
alpha = t/2
mu = -1000
Z = 1000

a = 1
nr = 50
#n = 3*(nr-1)
n = 2*nr

bdg = np.zeros((4*n,4*n), dtype='complex')
pm = np.exp(-1.0j*np.pi/2)
pp = np.exp(+1.0j*np.pi/2)

# Only need to fill the lower triangle of the matrix
for i in range(nr-1):
  bdg[i,i] = -mu+Z
  bdg[i+n,i+n] = -mu-Z
  bdg[i+2*n,i+2*n] = mu-Z
  bdg[i+3*n,i+3*n] = mu+Z
  bdg[i+1,i] = -t*pp
  bdg[i+1+n,i+n] = -t*pp
  bdg[i+1+2*n,i+2*n] = t*pp
  bdg[i+1+3*n,i+3*n] = t*pp
  bdg[i+1+n,i] = alpha*pp
  bdg[i+n,i+1] = -alpha*pm
  bdg[i+3*n,i+1+2*n] = alpha*pp
  bdg[i+1+3*n,i+2*n] = -alpha*pm
  bdg[i+n,i] = delta
  bdg[i+2*n,i+n] = delta

for j in range(nr-1):
  i = j+nr
  bdg[i,i] = -mu+Z
  bdg[i+n,i+n] = -mu-Z
  bdg[i+2*n,i+2*n] = mu-Z
  bdg[i+3*n,i+3*n] = mu+Z
  bdg[i+1,i] = -t
  bdg[i+1+n,i+n] = -t
  bdg[i+1+2*n,i+2*n] = t
  bdg[i+1+3*n,i+3*n] = t
  bdg[i+1+n,i] = alpha
  bdg[i+n,i+1] = -alpha
  bdg[i+3*n,i+1+2*n] = alpha
  bdg[i+1+3*n,i+2*n] = -alpha
  bdg[i+n,i] = delta
  bdg[i+2*n,i+n] = delta

# Energy at n-1 pt
bdg[n-1,n-1] = -mu+Z
bdg[2*n-1,2*n-1] = -mu-Z
bdg[3*n-1,3*n-1] = mu-Z
bdg[4*n-1,4*n-1] = mu+Z
bdg[2*n-1,n-1] = delta
bdg[3*n-1,2*n-1] = delta

# PBC here
#bdg[n-1,0] = t
#bdg[2*n-1,n] = t
#bdg[3*n-1,2*n] = -t
#bdg[4*n-1,3*n] = -t
#bdg[i,n] = -alpha
#bdg[0,2*n-1] = alpha
#bdg[2*n,4*n-1] = -alpha
#bdg[3*n-1,3*n] = alpha

np.savetxt('./data/bdg.txt', bdg, fmt='%1.1f')
eng, vec = np.linalg.eigh(bdg)
#vec = vec[0:n,0:n]
#vec = np.real( np.matmul(vec.conj().T,vec) )
vec = np.real(np.multiply(vec,np.conj(vec)))
eng = np.real(eng)
np.savetxt('./data/kitaev-triangle-chain-energy.txt', eng, fmt='%1.8e')
np.savetxt('./data/kitaev-triangle-chain-states.txt', vec, fmt='%1.8e')
