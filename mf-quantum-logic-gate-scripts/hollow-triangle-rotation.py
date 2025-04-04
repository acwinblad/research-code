#!/usr/bin/python3

import hollow_triangle_module as htm
import numpy as np
import shapely as sh
from shapely import ops
import shapely.geometry as geo
import descartes as ds
import matplotlib.pyplot as plt
import imageio
import os

plotLattice = False
plotVectorField = False
plotWavefunction = False
plotSpectral = True

# Define parameters
t = 1
delta = t
mu = 1.60*t
a = 1
nr = 80

vecPotFunc = 'step-function'
#vecPotFunc = 'linear'
vecPotFunc = 'constant'
#vecPotFunc = 'tanh'

if(vecPotFunc == ''):
  print('pick a vector potential type from the following: step-function, tanh, linear, or constant')

A0 = 0.75
Amin = 1.10
Amax = 1.250
Amin = 0.77
Amax = 0.830
#Amin = 2.35
#Amax = 2.35



# The inner boundary is dependant on the width we want and bounded by the outer boundary.
# Define the width first then we can determine the the number of rows of the inner boundary.
width = 3
width *= a
# Build the hollow triangle lattice
hollowtri, innertri = htm.build_hollow_triangle(a, nr, width)

# Populate what the directory hierarchy will be
latticePlotPath, filepath = htm.create_directory_path('rotation-' + vecPotFunc, mu, nr, width)

# Find lattice point coordinates inside the hollow triangle, used for nearest-neighbor and saved later for plotting (maybe)
coords = htm.hollow_triangle_coords(a, nr, hollowtri)

# Initialize BdG Hamiltonian
n = np.size(coords[:,0])
bdg = np.zeros((2*n,2*n), dtype='complex')
nnlist, nnphaseFtr, nnphiParams = htm.nearest_neighbor_list(a, coords)

# Loop through the varying angles of t for the vector potential
nE = 1*4 # must be even?

if(plotSpectral):
  nt = 4*60
  tf = 3*np.pi/3
  tvals = np.linspace(0,tf,nt+1)
  Avals = np.linspace(Amax,Amin,nt//6, endpoint=False)
  Avals = np.append(Avals, np.linspace(Amin,Amax,nt//6))
  Avals = np.append(Avals,Avals)
  Avals = np.append(Avals,Avals)
  Avals = np.append(Avals,Amax)

if(plotWavefunction):
  nt = 1*6
  tf = 3*np.pi/3
  tvals = np.linspace(0,tf,nt+1)
  Avals = np.linspace(Amax,Amin,nt//2, endpoint=False)
  #Avals = np.append(Avals, np.linspace(Amin,Amax,nt//2+1))
  Avals = np.append(Avals, np.linspace(Amin,Amax,nt//6))
  Avals = np.append(Avals,Avals)
  Avals = np.append(Avals,Avals)
  Avals = np.append(Avals,Amax)

evt = np.zeros((2*nE,nt+1))
evtt = np.zeros((nt+1))
wf = np.zeros((n, nt+1))


# the order parameter will not change so we only need to initialize it once
for j in range(len(nnlist)):
  for nnl, l in enumerate(nnlist[j]):
    bdg[l, n+j] = delta*nnphaseFtr[j][nnl]
    bdg[j, n+l] = -bdg[l, n+j]

bdg[0:n, 0:n] = -mu*np.eye(n)
bdg[n:2*n, n:2*n] = mu*np.eye(n)

for k, angle in enumerate(tvals):
  # Construct the BdG Hamiltonian for varying vector potential rotation angles
  for j in range(len(nnlist)):
    for nnl, l in enumerate(nnlist[j]):
      phiftr = htm.calc_phi(a, coords[j,0], coords[l,0], coords[j,1], coords[l,1], nnphiParams[j][nnl][0], nnphiParams[j][nnl][1], angle, vecPotFunc)
      bdg[j, l] = -t * np.exp(1.0j * phiftr * Avals[k])
      bdg[n+j, n+l] = -np.conjugate(bdg[j, l])

  # Solve the eigenvalue problem for energies only
  eng, vec = np.linalg.eigh(bdg, UPLO = 'U')
  evt[:,k] = eng[n-nE:n+nE]
  evtt[k] = eng[n]
  vec = np.real(np.multiply(vec, np.conj(vec)))
  vvt = vec[:,n-nE:n+nE]
  wf[:,k] = vec[0:n,n] + vec[n:2*n,n] + vec[0:n,n-1] + vec[n:2*n,n-1]

  ## Save data
  #np.savetxt('./data/hollow-triangle-energy.txt', eng, fmt='%1.8e')

if(plotWavefunction):
  htm.plot_hollow_triangle_wavefunction_circles(a, width, nr, coords, tvals, evtt, wf, filepath)
if(plotSpectral):
  htm.plot_hollow_triangle_rotation_spectral_flow(mu, nr, A0, width, nE, tvals, evt, filepath
      )

