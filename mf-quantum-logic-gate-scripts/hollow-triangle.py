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
nr = 25

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
width = 1
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
dA = A0 / nk
Amults = int(2)
Amax = Amults * A0
if(Amults>1):
  avals = np.linspace(0,Amax,Amults*nk+1)
  eva = np.zeros((2*nE,Amults*nk+1))
else:
  avals = np.linspace(0,A0,nk)
  eva = np.zeros((2*nE,nk))

# Initialize BdG Hamiltonian
n = np.size(coords[:,0])
bdg = np.zeros((2*n,2*n), dtype='complex')
nnlist, nnphaseFtr, nnphiParams = htm.nearest_neighbor_list(a, coords)

#bdg[n-1, n-nr] = -t
#bdg[2*n-1, 2*n-nr] = t
#bdg[2*n-1, n-nr] = delta
#bdg[2*n-nr, n-1] = -delta

# the order parameter will not change so we only need to initialize it once
for j in range(len(nnlist)):
  for nnl, l in enumerate(nnlist[j]):
    bdg[l, n+j] = delta*nnphaseFtr[j][nnl]
    bdg[j, n+l] = -bdg[l, n+j]

#bdg[2+n,1] = 0
#bdg[1+n,2] = 0
#bdg[2*n-2,n-nr-1] = 0
#bdg[2*n-nr-1, n-2] = 0
#bdg[2*n-nr+1, n-nr-2] = 0
#bdg[2*n-nr-2, n-nr+1] = 0

bdg[0:n, 0:n] = -mu*np.eye(n)
#bdg[n-nr:n, n-nr:n] = -100*t*np.eye(nr)
#bdg[0:n-nr:2,0:n-nr:2] = -1000*mu*np.eye(nr-1)
#bdg[0:3, 0:3] = -1000*mu*np.eye(3)

bdg[n:2*n, n:2*n] = mu*np.eye(n)
#bdg[2*n-nr:2*n, 2*n-nr:2*n] = -bdg[n-nr:n, n-nr:n]
#bdg[n:2*n-nr:2,n:2*n-nr:2] = 1000*mu*np.eye(nr-1)
#bdg[n:n+3, n:n+3] = 1000*mu*np.eye(3)

for k,values in enumerate(avals):
  # Construct the BdG Hamiltonian for varying vector potential strengths bvalues
  for j in range(len(nnlist)):
    for nnl, l in enumerate(nnlist[j]):
      phiftr = htm.calc_phi(a, coords[j,0], coords[l,0], coords[j,1], coords[l,1], nnphiParams[j][nnl][0], nnphiParams[j][nnl][1], 0, vecPotFunc)
      bdg[j, l] = -t * phiftr**values
      bdg[j+n, l+n] = -np.conjugate(bdg[j, l])

  #bdg[2,1] = 0
  #bdg[2+n,1+n] = 0
  #bdg[n-2,n-nr-1] = 0
  #bdg[2*n-2,2*n-nr-1] = 0
  #bdg[n-nr+1,n-nr-2] = 0
  #bdg[2*n-nr+1,2*n-nr-2] = 0

  # Solve the eigenvalue problem for energies only
  eng, vec = np.linalg.eigh(bdg/2, UPLO = 'U')
  eva[:,k] = eng[n-nE:n+nE]
  vec = np.real(np.multiply(vec, np.conj(vec)))
  vva = vec[:,n-nE:n+nE]

  # Let's only plot the wavefunction of the states if we have a MF or MF-like state for a given vector potential strength
  if(abs(eva[nE,k]) < 1E-8):
    if(width != 1):
      if(values % (A0 / 2) == 0):
        htm.plot_hollow_triangle_wavefunction(a, width, innertri, coords, triang, nE, values, eva[:,k], vva, filepath)
    else:
      # copy the eigenstate elements to the correct locations!
      # split
      v0 = vva[0:n,:]
      v00 = vva[n:2*n,:]
      # append in the cloned coordinates eigenstate values
      v1 = np.vstack([v0,v0[clonedCoordsBool,:]])
      v2 = np.vstack([v00,v00[clonedCoordsBool,:]])
      # recombine the two sets to get the intended state vector
      newvec = np.vstack([v1,v2])
      #htm.plot_hollow_triangle_wavefunction(a, width, innertri, np.vstack([coords,clonedCoords]), triang, nE, values, eva[:,k], newvec, filepath)
      #htm.plot_hollow_triangle_wavefunction_circles(a, width, innertri, coords, triang, nE, values, eva[:,k], vva, filepath)

#np.savetxt('./data/hollow-triangle-bdg-copy.txt', bdg, fmt='%1.2e')
  ## Save data
  #np.savetxt('./data/hollow-triangle-energy.txt', eng, fmt='%1.8e')

if(plotSpectral):
  htm.plot_hollow_triangle_spectral_flow(mu, nr, A0, width, nE, avals, eva, filepath)
