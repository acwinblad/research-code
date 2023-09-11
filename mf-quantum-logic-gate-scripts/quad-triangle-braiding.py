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

plotWavefunction = False
plotSpectral = False
plotWavefunction = True
#plotSpectral = True

# Define parameters
t = 1
delta = t
mu = 1.6*t
a = 1
nr = 50

vecPotFunc = 'step-function'
#vecPotFunc = 'linear'
vecPotFunc = 'constant'
#vecPotFunc = 'tanh'

if(vecPotFunc=='step-function'):
  A0 = 2 * np.pi / (3*np.sqrt(3) * a)
  A0 = 2.75 / a
elif(vecPotFunc=='tanh'):
  A0 = 2 * np.pi / (3 * np.sqrt(3) * a)
  A0 = 2.75 / a
elif(vecPotFunc=='linear'):
  A0 = 8 * np.pi / (3 * np.sqrt(3) * a**2 * (2 * nr - 3) )
elif(vecPotFunc=='constant'):
  A0 = 4 * np.pi / (np.sqrt(3) * a) / 2
  A0 = 2.75
else:
  print('pick a vector potential type from the following: step-function, tanh, linear, or constant')


# The inner boundary is dependant on the width we want and bounded by the outer boundary.
# Define the width first then we can determine the the number of rows of the inner boundary.
width = 1
width *= a
# Build the hollow triangle lattice
hollowtri, innertri = htm.build_hollow_triangle(a, nr, width)

# Populate what the directory hierarchy will be
latticePlotPath, filepath = htm.create_directory_path('braiding-' + vecPotFunc, mu, nr, width)

# Find lattice point coordinates inside the hollow triangle, used for nearest-neighbor and saved later for plotting (maybe)
coords = htm.hollow_triangle_coords(a, nr, hollowtri)

# Initialize BdG Hamiltonian
n = np.size(coords[:,0])
bdgl = np.zeros((2*n,2*n), dtype='complex')
nnlist, nnphaseFtr, nnphiParams = htm.nearest_neighbor_list(a, coords)

# Loop through the varying angles of t for the vector potential
nE = 1*6 # must be even?
nt = 1*5
ti = 1*np.pi/6
tf = 1*np.pi/3
tvals = np.linspace(ti,tf,nt)
evt = np.zeros((2*nE, nt))
evtt = np.zeros((nt))
vgst = np.zeros((4, 8*n, nt))


bdgl[0:n, 0:n] = -mu*np.eye(n)
bdgl[n:2*n, n:2*n] = mu*np.eye(n)

# the order parameter will not change so we only need to initialize it once
for j in range(len(nnlist)):
  for nnl, l in enumerate(nnlist[j]):
    bdgl[l, n+j] = delta*nnphaseFtr[j][nnl]
    bdgl[j, n+l] = -bdgl[l, n+j]

for j in range(len(nnlist)):
  for nnl, l in enumerate(nnlist[j]):
    phiftr = htm.calc_phi(a, coords[j,0], coords[l,0], coords[j,1], coords[l,1], nnphiParams[j][nnl][0], nnphiParams[j][nnl][1], 0, vecPotFunc)
    bdgl[j, l] = -t * phiftr**A0
    bdgl[n+j, n+l] = -np.conjugate(bdgl[j, l])

zmat = np.zeros((2*n,2*n))
#bdg = np.block( [ [bdgl,zmat], [zmat,bdgl] ] )
bdg = np.block( [ [bdgl,zmat,zmat,zmat], [zmat,bdgl,zmat,zmat], [zmat,zmat,bdgl,zmat], [zmat,zmat,zmat,bdgl] ] )

# Link the hopping between the four triangles
bdg[n-1, 3*n-nr] = -t
bdg[2*n-1, 4*n-nr] = t
bdg[3*n-1, 5*n-nr] = -t
bdg[4*n-1, 6*n-nr] = t
bdg[0, 7*n-nr] = -t
bdg[n, 8*n-nr] = t
bdg[2*n, 7*n-1] = -t
bdg[3*n, 8*n-1] = t

# Link the pairing between the four triangles
bdg[n-1, 4*n-nr] = -delta
bdg[2*n-1, 3*n-nr] = delta
bdg[3*n-1, 6*n-nr] = -delta
bdg[4*n-1, 5*n-nr] = delta
bdg[n, 7*n-nr] = -delta
bdg[0, 8*n-nr] = delta
bdg[2*n, 8*n-1] = -delta
bdg[3*n, 7*n-1] = delta

for k, angle in enumerate(tvals):
  # Construct the BdG Hamiltonian for varying vector potential rotation angles
  for j in range(len(nnlist)):
    for nnl, l in enumerate(nnlist[j]):
      phiftr = htm.calc_phi(a, coords[j,0], coords[l,0], coords[j,1], coords[l,1], nnphiParams[j][nnl][0], nnphiParams[j][nnl][1], angle, vecPotFunc)
      bdg[2*n+j, 2*n+l] = -t * phiftr**A0
      bdg[3*n+j, 3*n+l] = -np.conjugate(bdg[2*n+j, 2*n+l])

  # Solve the eigenvalue problem for energies only
  #eng = np.linalg.eigvalsh(bdg, UPLO = 'U')
  eng, vec = np.linalg.eigh(bdg, UPLO = 'U')
  evt[:,k] = eng[4*n-nE:4*n+nE]
  evtt[k] = eng[4*n]
  vec = np.real(np.multiply(vec, np.conj(vec)))
  for i in range(4):
    vgst[i,:,k] = vec[:,4*n-2+i]

if(plotWavefunction):
  htm.plot_quad_hollow_triangle_wavefunction_circles(a, width, nr, coords, tvals, evtt, vgst, filepath)
if(plotSpectral):
  htm.plot_hollow_triangle_rotation_spectral_flow(mu, nr, A0, width, nE, tvals, evt, filepath
      )

