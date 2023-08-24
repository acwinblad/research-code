#!/usr/bin/python3

import numpy as np

t = 1
delta = t
mu = 0.4
n = 20
A = 2.0*np.pi/3
theta = 1*np.pi/3

h1 = -mu*np.eye(n) - t*(np.exp(-1.0j*A)*np.diag(np.ones(n-1),k=-1)+np.exp(1.0j*A)*np.diag(np.ones(n-1),k=1))
h2 = delta*np.exp(-1.0j*theta)*(np.diag(np.ones(n-1),1)-np.diag(np.ones(n-1),-1))
ht = np.hstack((h1,0*h2))
hb = np.hstack((h2,-np.conjugate(h1)))
h = np.vstack((ht,hb))

eng, vec = np.linalg.eigh(h)
print(eng[n-1:n+1])
vec = np.real(np.multiply(vec,np.conj(vec)))
np.savetxt('./data/kitaev-peierls-chain-energy.txt', eng, fmt = '%1.2e')
np.savetxt('./data/kitaev-peierls-chain-states.txt', vec, fmt = '%1.2e')

