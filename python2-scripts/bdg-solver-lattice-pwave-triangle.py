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
delta = 1.0+0j
#thop = 10*np.abs(delta)
thop = 1
mu = 6*thop
a = 1.
#nr = np.int(thop/np.abs(delta))
nr = 40
n = nr*(nr+1)/2

bdg = np.zeros((2*n,2*n),dtype='complex')

# create equilateral triangular lattice mesh
siteCoord = np.zeros((n,2))
latticeCtr = 0
for i in xrange(nr):
  for j in xrange(i+1):
    siteCoord[latticeCtr,0] = a*(j-i/2.)
    siteCoord[latticeCtr,1] = -i*a*np.sqrt(3)/2.
    latticeCtr += 1

# fill in bdg hamiltonian
for i in xrange(n):
  for j in xrange(n-i):
    dx = siteCoord[i+j,0]-siteCoord[i,0]
    dy = siteCoord[i+j,1]-siteCoord[i,1]
    d = np.sqrt(dx**2 + dy**2)
    if d < 1e-5:
      # this is the current lattice site
      bdg[i,i] = -mu+6*thop
      bdg[i+n,i+n] = -bdg[i,i]
    elif np.abs(d-a) < 1e-5:
      # this is a nearest neighbor, fill in the hopping and superconductivity terms in the correct element
      phaseAngle = np.arctan(dy/dx)
      if dx<0:
        phaseAngle+= np.pi
      phaseAngle=0
      #print "phase angle", phaseAngle
      bdg[i+j,i] = -thop
      bdg[i,i+j] = -thop
      bdg[i+j+n,i+n] = thop
      bdg[i+n,i+j+n] = thop
      bdg[i+j,i+n] = delta*np.exp(1j*phaseAngle)
      bdg[i,i+j+n] = -delta*np.exp(1j*phaseAngle)
      bdg[i+n,i+j] = np.conj(bdg[i+j,i+n])
      bdg[i+j+n,i] = np.conj(bdg[i,i+j+n])

#np.savetxt('../data/bdg-matrix-triangle.txt', bdg, fmt='%1.0e')

# solve the eigen-value problem to calculate the energy and wavefunctions
bdg = bdg +np.conj(np.transpose(bdg))
energy, states = np.linalg.eigh(bdg)
# normalize the eigenstates, Energies and Nomalize Eigenstates should all be real, typecast to real
idx = energy.argsort()[::-1]
energy = np.real(energy[idx])
states = states[:,idx]
states = np.real(np.multiply(states,np.conj(states)))
np.savetxt('../../data/bdg-energy-triangle.txt', energy, fmt='%1.8f')
np.savetxt('../../data/bdg-states-triangle.txt', states, fmt='%1.32f')
np.savetxt('../../data/bdg-coordinates-triangle.txt', siteCoord, fmt='%1.32f')
