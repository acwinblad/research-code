#!/usr/bin/python
#
# bdg solver lattice p-wave triangle
# Created by: Aidan Winblad
# 09/07/2017
#
# This code solves the eigenvalue problem of the bdg matrix for a lattice p-wave equilateral triangle
# We should get an array of energy values. We will then wrap the energy values into a 
# energy density histogram. yada yada yada
#

import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d


# initialize lattice parameters
delta = 1.0
#thop = 10*np.abs(delta)
t = 10.
mu = 6*t
a = 1.
#nr = np.int(thop/np.abs(delta))
nr = 25
n = nr*(nr+1)/2

bdg = np.zeros((2*n,2*n),dtype='complex')

# create equilateral triangular lattice mesh
siteCoord = np.zeros((n,2))
latticeCtr = 0
for i in range(nr):
  for j in range(i+1):
    siteCoord[latticeCtr,0] = a*(j-i/2.)
    siteCoord[latticeCtr,1] = -i*a*np.sqrt(3)/2.
    latticeCtr += 1

# fill in bdg hamiltonian 
for i in range(n):
  for j in range(n-i):
    dx = siteCoord[i+j,0]-siteCoord[i,0]
    dy = siteCoord[i+j,1]-siteCoord[i,1]
    d = np.sqrt(dx**2 + dy**2)
    if d < 1e-5:
      # this is the current lattice site
      bdg[i,i] = (-mu+6*t)/2.
      bdg[i+n,i+n] = -bdg[i,i]
    elif np.abs(d-a) < 1e-5:
      # this is a nearest neighbor, fill in the hopping and superconductivity terms in the correct element
      phaseAngle = np.arctan(dy/dx)
      if dx<0:
        phaseAngle+= np.pi
      #print "phase angle", phaseAngle
      bdg[i,i+j] = -t
      bdg[i+n,i+j+n] = t
      bdg[i,i+j+n] = delta*np.exp(1j*phaseAngle)
      bdg[i+j,i+n] = -delta*np.exp(1.0j*phaseAngle)

#np.savetxt('../data/bdg-matrix-triangle.txt', bdg, fmt='%1.0e')

# solve the eigen-value problem to calculate the energy and wavefunctions
bdg += np.conj(np.transpose(bdg))
energy, states = np.linalg.eigh(bdg)
# normalize the eigenstates, Energies and Nomalize Eigenstates should all be real, typecast to real
idx = energy.argsort()[::-1]
energy = np.real(energy[idx])
states = states[:,idx]
states = np.real(np.multiply(states,np.conj(states)))
np.savetxt('../../data/tbm-lattice-pwave-energy-triangle.txt', energy, fmt='%1.8f')
np.savetxt('../../data/tbm-lattice-pwave-states-triangle.txt', states, fmt='%1.32f')
np.savetxt('../../data/tbm-lattice-pwave-coordinates-triangle.txt', siteCoord, fmt='%1.32f')

xaxis = np.array([-1,1])

plt.figure(figsize=(2,4))
plt.tick_params(
    axis='x',
    which='both',
    bottom='off',
    top='off',
    labelbottom='off')
plt.ylabel('$\epsilon (t)$', fontsize=12)
plt.xlim(-1.2,1.2)
for i in range(2*n):
  plt.plot(xaxis,[energy[i],energy[i]], 'b', linewidth=0.25)
plt.tight_layout()
plt.show()
