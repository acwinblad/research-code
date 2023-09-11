#!/usr/bin/python3

import hollow_triangle_module as htm
import numpy as np
import shapely as sh
from shapely import ops
import shapely.geometry as geo
import descartes as ds
import matplotlib.pyplot as plt

plotLattice = False
plotVectorField = False
plotWavefunction = True
plotSpectral = True

# Define parameters
t = 1
delta = t
mu = 1.6*t
a = 1
nr = 100

vecPotFunc = 'step-function'
#vecPotFunc = 'linear'
vecPotFunc = 'constant'
#vecPotFunc = 'tanh'
if(vecPotFunc=='step-function'):
  A0 = 6 * np.pi / (3 * np.sqrt(3) * a)
  #A0 = 2*np.pi / a /2
  #A0 = 2*np.pi / (np.sqrt(3) * a) / 2
elif(vecPotFunc=='linear'):
  A0 = 8 * np.pi / (3 * np.sqrt(3) * a**2 * (2 * nr - 3) )
elif(vecPotFunc=='constant'):
  A0 = 6 * np.pi / (3 * np.sqrt(3) * a)
  A0 = 2.55 / a
elif(vecPotFunc=='tanh'):
  A0 = 2 * np.pi / (3 * np.sqrt(3) * a)
  A0 = 6 * np.pi / (3 * np.sqrt(3) * a)
else:
  print('pick a vector potential type from the following: step-function, linear, tanh')


# The inner boundary is dependant on the width we want and bounded by the outer boundary.
# Define the width first then we can determine the the number of rows of the inner boundary.
width = 1
width *= a
# Build the hollow triangle lattice
hollowtri, innertri = htm.build_hollow_triangle(a, nr, width)

# Populate what the directory hierarchy will be
latticePlotPath, filepath = htm.create_directory_path('varying-mu-' + vecPotFunc, mu, nr, width)

# Test to see if the hollow triangle is being constructed as wanted
if(plotLattice):
  htm.plot_hollow_triangle_lattice(a, nr, hollowtri, innertri, latticePlotPath)

# Find lattice point coordinates inside the hollow triangle, used for nearest-neighbor and saved later for plotting (maybe)
coords = htm.hollow_triangle_coords(a, nr, hollowtri)
# if the width is 1 we want to clone the interior points, we use this later for plotting. Also get the boolean array of their locations wrt to the original coordinates matrix.
if(width == a):
  clonedCoords, clonedCoordsBool = htm.clone_width_one_interior_points(a, coords)

# Plot the Vector field potential for a given setup
if(plotVectorField):
  htm.plot_vector_potential(coords, latticePlotPath)

# Now that we have both coordinates and a hollow triangle polygon we want to make a centroids mask to be used later for plotting the wavefunction. returns a masked triang
if(width != a):
  triang = htm.create_centroids_mask(a, coords, innertri)
else:
  innertri = htm.shifted_innertri(a, nr)
  newCoords = np.vstack([coords,clonedCoords])
  triang = htm.create_centroids_mask(a, newCoords, innertri)

# Loop through the varying values of B for the vector potential
nE = 2*2 # must be even?
nk = 100
nmu = 45
mui = mu
muf = 2.4
muarr = np.linspace(mui,muf,nmu)
engvmu = np.zeros((2*nE, nmu))

# Initialize BdG Hamiltonian
n = np.size(coords[:,0])
bdgl = np.zeros((2*n,2*n), dtype='complex')
bdgr = np.zeros((2*n,2*n), dtype='complex')
nnlist, nnphaseFtr, nnphiParams = htm.nearest_neighbor_list(a, coords)

# the order parameter will not change so we only need to initialize it once
for j in range(len(nnlist)):
  for nnl, l in enumerate(nnlist[j]):
    bdgl[l, n+j] = delta*nnphaseFtr[j][nnl]
    bdgl[j, n+l] = -bdgl[l, n+j]

for j in range(len(nnlist)):
  for nnl, l in enumerate(nnlist[j]):
    phiftr = htm.calc_phi(a, coords[j,0], coords[l,0], coords[j,1], coords[l,1], nnphiParams[j][nnl][0], nnphiParams[j][nnl][1], 0, vecPotFunc)
    bdgl[j, l] = -t * phiftr**A0
    bdgl[j+n, l+n] = -np.conjugate(bdgl[j, l])

bdgr = bdgl

for j in range(n):
  bdgl[j, j] = -mu
  bdgl[n+j, n+j] = mu

# combine the matrices and make sure they are connected at end points
bdg = np.block([[bdgl, np.zeros((2*n,2*n))],[np.zeros((2*n,2*n)), bdgr]])
bdg[n-1, 3*n-nr] = -t
bdg[2*n-1, 4*n-nr] = t
bdg[n-1, 4*n-nr] = -delta
bdg[2*n-1, 3*n-nr] = delta

for k, val in enumerate(muarr):
  # Construct the BdG Hamiltonian for varying mu strengths on right triangle
  for j in range(n):
    bdg[2*n+j, 2*n+j] = -val
    bdg[3*n+j, 3*n+j] = val


  # Solve the eigenvalue problem for energies only
  eng = np.linalg.eigvalsh(bdg, UPLO = 'U')
  engvmu[:,k] = eng[2*n-nE:2*n+nE]

  # Let's only plot the wavefunction of the states if we have a MF or MF-like state for a given vector potential strength

if(plotSpectral):
  htm.plot_hollow_triangle_spectral_flow(mu, nr, A0, width, nE, muarr, engvmu, filepath
      )
