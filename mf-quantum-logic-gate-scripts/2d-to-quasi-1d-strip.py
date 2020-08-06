#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

plotFlag = False

npw = 200
nsw = int(1*1.5*npw)
n = 2*nsw+npw
# create zero matrix
bdg = np.zeros((4*n,4*n))

# material values
delta = 10.0    # superconducting order parameter
t = 10.0    # hopping strength
alpha = 0.25*t  # Rashba-SOC strength
Vz = 50.   # Zeeman field strength along z-axis

# since v will change as a function of x we will make it into a matrix.
v = np.zeros(n)
# transform the s-wave into an effective p-wave region.
v[n//2-npw//2:n//2+npw//2] = Vz
# allow a gaussian decay into the s-wave region
sigma = npw/8.
g = np.array([ Vz*np.exp(-(i-nsw)**2/sigma**2) for i in range(nsw)])
v[0:nsw] = g
v[n//2+npw//2:n] = np.flipud(g)
np.savetxt('./data/2d-1d-zeeman-dist.txt', v)
v = np.diag(v)
# mu = Vz+Const*t, eps = 2*t-mu
eps = 2*t-1*v-0*Vz-0.01*t

np.savetxt('./data/2d-1d-config.txt', [npw, n], fmt='%i')

# since it is hermitian we only need to fill the lower triangle
# it is built by going down the 0th block diagonal then the rext diagonal after, repeat
# since we know the bdg has 16 section but only 8 to fill that are static we write them explicitly

bdg[0:n, 0:n] = eps * np.eye(n) + v - t * ( np.eye(n,k=-1) + 0 * np.eye(n,k=-n) )
bdg[n:2*n, n:2*n] = eps * np.eye(n) - v - t * ( np.eye(n,k=-1) + 0 * np.eye(n,k=-n) )
bdg[2*n:3*n, 2*n:3*n] = -eps * np.eye(n) - v + t * ( np.eye(n,k=-1) + 0 * np.eye(n,k=-n) )
bdg[3*n:4*n, 3*n:4*n] = -eps * np.eye(n) + v + t * ( np.eye(n,k=-1) + 0 * np.eye(n,k=-n) )

bdg[1*n:2*n, 0:n] = alpha * ( np.eye(n,k=-1) - np.eye(n,k=1) - 0 * ( np.eye(n,k=-n) - np.eye(n,k=n) ) )
bdg[2*n:3*n, n:2*n] = -delta * np.eye(n)
bdg[3*n:4*n, 2*n:3*n] = -alpha * ( np.eye(n,k=-1) - np.eye(n,k=1) - 0 * ( np.eye(n,k=-n) - np.eye(n,k=n) ) )
bdg[3*n:4*n, 0:n] = delta * np.eye(n)

#print(bdg)
#print()

eng, states = np.linalg.eigh(bdg)
#states = states[0:n, :]
#idx = eng.argsort()[::-1]
#eng = eng[idx]
#states = states[:, idx]
#print(eng[4*(nsw+npw//2)-1:4*(nsw+npw//2)+1])
#print(eng[4*(nsw+8):4*(nsw+npw-8)])
np.savetxt('./data/2d-1d-eigenvalues.txt', eng)
prob = np.multiply(states,np.conj(states))
np.savetxt('./data/2d-1d-states.txt', prob)
