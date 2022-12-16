#!/usr/bin/python3

import hollow_triangle_module as htm
import numpy as np
import shapely as sh
from shapely import ops
import shapely.geometry as geo
import descartes as ds
import matplotlib.pyplot as plt
from pfapack import pfaffian as pf

plotLattice = False

# Define parameters
PI = np.pi
t = 1.0
delta = t
nmu = 30
muvals = np.linspace(0,2*t,nmu+1)/2
a = 1
nr = 25

vecPotFunc = 'step-function'
#vecPotFunc = 'linear'
#vecPotFunc = 'tanh'
if(vecPotFunc=='step-function'):
  B0 = 4 * PI / (3 * np.sqrt(3) * a)
  B0 = 2.3*PI / a
elif(vecPotFunc=='linear'):
  B0 = 8 * PI / (3 * np.sqrt(3) * a**2 * (2 * nr - 3) )
elif(vecPotFunc=='tanh'):
  print('tanh function not setup yet')
else:
  print('pick a vector potential type from the following: step-function, linear, tanh')

# Create B values for the vector potential strength
Bmax = B0
nB = 2*nmu
Bvals = np.linspace(0,Bmax,nB+1)

# The inner boundary is dependant on the width we want and bounded by the outer boundary.
# Define the width first then we can determine the the number of rows of the inner boundary.
width = 1

# Build the hollow triangle lattice
hollowtri, innertri = htm.build_hollow_triangle(a, nr, width)

# Test to see if the hollow triangle is being constructed as wanted
if(plotLattice):
  htm.plot_hollow_triangle_lattice(a, nr, hollowtri, innertri, latticePlotPath)

# Find lattice point coordinates inside the hollow triangle, used for nearest-neighbor and saved later for plotting (maybe)
coords = htm.hollow_triangle_coords(a, nr, hollowtri)

# if the width is 1 we want to clone the interior points, we use this later for plotting. Also get the boolean array of their locations wrt to the original coordinates matrix.
if(width == 1):
  clonedCoords, clonedCoordsBool = htm.clone_width_one_interior_points(a, coords)

# Now that we have both coordinates and a hollow triangle polygon we want to make a centroids mask to be used later for plotting the wavefunction. returns a masked triang
if(width != 1):
  triang = htm.create_centroids_mask(a, coords, innertri)
else:
  innertri = htm.shifted_innertri(a, nr)
  newCoords = np.vstack([coords,clonedCoords])
  triang = htm.create_centroids_mask(a, newCoords, innertri)

# Initiate 'empty' BdG matrix and add PBC's for bottom two triangle vertices
n = np.size(coords[:,0])
bdg = np.zeros((2*n,2*n), dtype='complex')
nnlist, nnphaseFtr, nnphiFtr = htm.nearest_neighbor_list(a, coords, vecPotFunc)

# Set the PBC to be between the bottom two vertices
bdg[n-1,n-nr] = -t
bdg[2*n-1, 2*n-nr] = t
bdg[2*n-1, n-nr] = delta
bdg[2*n-nr, n-1] = -delta

# The order parameter will not change so we only need to initialize it once in the BdG Hamiltonian
for i in range(len(nnlist)):
  for j, nn in enumerate(nnlist[i]):
    bdg[i+n, nn] = delta*nnphaseFtr[i][j]
    bdg[nn+n, i] = -bdg[i+n, nn]

# These lines turn off the order parameter between the inner links at the vertices of the triangle
#bdg[2+n, 1] = 0
#bdg[1+n, 2] = 0
#bdg[2*n-2, n-nr-1] = 0
#bdg[2*n-nr-1, n-2] = 0
#bdg[2*n-nr+1, n-nr-2] = 0
#bdg[2*n-nr-2, n-nr+1] = 0


# Create tranformation matrix to Majorana basis
U = np.sqrt(0.5) * np.matrix([[1,1],[-1.0j,1.0j]])
U = np.kron(U, np.identity(n))
majNum = np.zeros((nmu+1, nB+1))

for k, bvalues in enumerate(Bvals):
  # Calculate and place hopping terms in the BdG Hamiltonian
  for i in range(len(nnlist)):
    for j, nn in enumerate(nnlist[i]):
      bdg[nn, i] = -t * nnphiFtr[i][j]**bvalues
      bdg[nn+n, i+n] = -np.conjugate(bdg[nn, i])
  # These lines turn off the hopping between the inner links at the vertices of the triangle
  #bdg[2,1] = 0
  #bdg[2+n,1+n] = 0
  #bdg[n-2,n-nr-1] = 0
  #bdg[2*n-2,2*n-nr-1] = 0
  #bdg[n-nr+1,n-nr-2] = 0
  #bdg[2*n-nr+1,2*n-nr-2] = 0

  for l, muvalues in enumerate(muvals):
    # Calculate and place the chemical potential terms in the Bdg Hamiltonian (along the diagonal)
    for i in range(n):
      bdg[i+n, i+n] = muvalues
      bdg[i,i] = -bdg[i+n,i+n]

    # Solve the eigenvalue problem for energies only
    # Need to add h.c. to correct the chemical potential and use later for the Majorana Number calc
    tmp = bdg + np.conjugate(np.transpose(bdg))
    eng = np.linalg.eigvalsh(tmp)

    # Calculate the Majorana number for each mu and vector potential strength B
    A = -1.0j * U * tmp * np.conjugate(np.transpose(U))
    majNum[l,k] = np.sign(np.real(pf.pfaffian(A)))*np.abs(eng[n]-eng[n-1])

np.savetxt('./data/majorana-number-hollow-triangle.txt', majNum)
print('Finished')
