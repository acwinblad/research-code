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
mu = 0.0*t
a = 1
nr = 26

vecPotFunc = 'step-function'
#vecPotFunc = 'linear'
#vecPotFunc = 'constant'
#vecPotFunc = 'tanh'
if(vecPotFunc=='step-function'):
  A0 = 6 * np.pi / (3 * np.sqrt(3) * a)
  #A0 = 2*np.pi / a /2
  A0 = 2*np.pi / (3 * np.sqrt(3) * a)
elif(vecPotFunc=='linear'):
  A0 = 8 * np.pi / (3 * np.sqrt(3) * a**2 * (2 * nr - 3) )
elif(vecPotFunc=='constant'):
  A0 = 6 * np.pi / (3 * np.sqrt(3) * a)
elif(vecPotFunc=='tanh'):
  A0 = 2 * np.pi / (3 * np.sqrt(3) * a)
else:
  print('pick a vector potential type from the following: step-function, linear, tanh')


# The inner boundary is dependant on the width we want and bounded by the outer boundary.
# Define the width first then we can determine the the number of rows of the inner boundary.
width = 0
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

# Construct the BdG Hamiltonian for varying vector potential strengths avalues
for j in range(len(nnlist)):
  for nnl, l in enumerate(nnlist[j]):
    phiftr = htm.calc_phi(a, coords[j,0], coords[l,0], coords[j,1], coords[l,1], nnphiParams[j][nnl][0], nnphiParams[j][nnl][1], 0, vecPotFunc)
    bdg[j, l] = -t * phiftr**A0
    bdg[j+n, l+n] = -np.conjugate(bdg[j, l])

# Solve the eigenvalue problem for energies only
eng, vec = np.linalg.eigh(bdg, UPLO = 'U')
vec = np.real(np.multiply(vec, np.conj(vec)))
vgs0a = vec[:,n]
vgs1a = vec[:,n+1]


if(plotSpectral):
  plt.figure()
  plt.xlim(0,1)
  plt.ylabel('Energy',fontsize=12)
  plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
  plt.ylim(-0.2,0.2)
  xarr = [0,1]
  for i in range(2*n):
    plt.plot(xarr,[eng[i],eng[i]], 'C0')
  plt.tight_layout()
  plt.savefig('./data/figures/full-triangle-spectral-flow.pdf')
  #plt.show()
  plt.close()


