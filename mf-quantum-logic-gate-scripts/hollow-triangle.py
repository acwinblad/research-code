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
nr = 50

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
elif(vecPotFunc=='tanh'):
  A0 = 2 * np.pi / (3 * np.sqrt(3) * a)
  A0 = 6 * np.pi / (3 * np.sqrt(3) * a)
else:
  print('pick a vector potential type from the following: step-function, linear, tanh')


# The inner boundary is dependant on the width we want and bounded by the outer boundary.
# Define the width first then we can determine the the number of rows of the inner boundary.
width = 3
width *= a
# Build the hollow triangle lattice
hollowtri, innertri = htm.build_hollow_triangle(a, nr, width)

# Populate what the directory hierarchy will be
latticePlotPath, filepath = htm.create_directory_path('increasing-' + vecPotFunc, mu, nr, width)

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

# Initialize BdG Hamiltonian
n = np.size(coords[:,0])
bdg = np.zeros((2*n,2*n), dtype='complex')
nnlist, nnphaseFtr, nnphiParams = htm.nearest_neighbor_list(a, coords)

# the order parameter will not change so we only need to initialize it once
for j in range(len(nnlist)):
  for nnl, l in enumerate(nnlist[j]):
    bdg[l, n+j] = delta*nnphaseFtr[j][nnl]
    bdg[j, n+l] = -bdg[l, n+j]

bdg[0:n, 0:n] = -mu*np.eye(n)
bdg[n:2*n, n:2*n] = mu*np.eye(n)

# Loop through the varying values of B for the vector potential
nE = 2*2 # must be even?
nk = 45
dA = A0 / nk
Amults = int(2)
Amax = Amults * A0
if(Amults>1):
  nkk = Amults*nk+1
  avals = np.linspace(0,Amax,nkk)
  eva = np.zeros((2*nE,nkk))
  evaa = np.zeros((nkk))
  vgs0a = np.zeros((2*n, nkk))
  vgs1a = np.zeros((2*n, nkk))
else:
  avals = np.linspace(0,A0,nk)
  eva = np.zeros((2*nE,nk))
  evaa = np.zeros((nk))
  vgs0a = np.zeros((2*n, nk))
  vgs1a = np.zeros((2*n, nk))

for k,values in enumerate(avals):
  # Construct the BdG Hamiltonian for varying vector potential strengths bvalues
  for j in range(len(nnlist)):
    for nnl, l in enumerate(nnlist[j]):
      phiftr = htm.calc_phi(a, coords[j,0], coords[l,0], coords[j,1], coords[l,1], nnphiParams[j][nnl][0], nnphiParams[j][nnl][1], 0, vecPotFunc)
      bdg[j, l] = -t * phiftr**values
      bdg[j+n, l+n] = -np.conjugate(bdg[j, l])

  # Solve the eigenvalue problem for energies only
  eng, vec = np.linalg.eigh(bdg, UPLO = 'U')
  eva[:,k] = eng[n-nE:n+nE]
  evaa[k] = eng[n]
  vec = np.real(np.multiply(vec, np.conj(vec)))
  vgs0a[:,k] = vec[:,n]
  vgs1a[:,k] = vec[:,n+1]

  # Let's only plot the wavefunction of the states if we have a MF or MF-like state for a given vector potential strength

#np.savetxt('./data/hollow-triangle-bdg-copy.txt', bdg, fmt='%1.2e')
  ## Save data
  #np.savetxt('./data/hollow-triangle-energy.txt', eng, fmt='%1.8e')

htm.plot_hollow_triangle_wavefunction_circles(a, width, nr, coords, avals, evaa, vgs0a, vgs1a, filepath)
if(plotSpectral):
  htm.plot_hollow_triangle_spectral_flow(mu, nr, A0, width, nE, avals, eva, filepath)
