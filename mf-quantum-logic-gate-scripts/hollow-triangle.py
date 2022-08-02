#!/usr/bin/python3

import hollow_triangle_module as htm
import numpy as np
import shapely as sh
from shapely import ops
import shapely.geometry as geo
import descartes as ds
import matplotlib.pyplot as plt

plotLattice = False
plotVectorField = True
plotWavefunction = True
plotSpectral = True

# Define parameters
t = 1
delta = t
muarr = np.linspace(-6*t,6*t,1001)
mu = muarr[235]
mu = t*0
a = 1
outernr = 25
outerlen = a*(outernr-1)

#vecPotFunc = 'step-function'
vecPotFunc = 'linear'
#vecPotFunc = 'tanh'
if(vecPotFunc=='step-function'):
  B0 = 2 * np.pi / (3 * np.sqrt(3) * a)
elif(vecPotFunc=='linear'):
  B0 = 8 * np.pi / (3 * np.sqrt(3) * a**2 * (2 * outernr - 3) )
elif(vecPotFunc=='tanh'):
  print('tanh function not setup yet')
else:
  print('pick a vector potential type from the following: step-function, linear, tanh')


# The inner boundary is dependant on the width we want and bounded by the outer boundary.
# Define the width first then we can determine the the number of rows of the inner boundary.
width = 0
width *= a
yp = np.sqrt(3)/2
# Build the hollow triangle lattice
hollowtri, innertri = htm.build_hollow_triangle(a, outernr, outerlen, width, yp)

# Populate what the directory hierarchy will be
latticePlotPath, filepath = htm.create_directory_path(vecPotFunc, mu, outernr, width)

# Test to see if the hollow triangle is being constructed as wanted
if(plotLattice):
  htm.plot_hollow_triangle_lattice(a, yp, outernr, hollowtri, innertri, latticePlotPath)

# Find lattice point coordinates inside the hollow triangle, used for nearest-neighbor and saved later for plotting (maybe)
coords = htm.hollow_triangle_coords(a, yp, outernr, hollowtri)
# if the width is 1 we want to clone the interior points, we use this later for plotting. Also get the boolean array of their locations wrt to the original coordinates matrix.
if(width == 1):
  clonedCoords, clonedCoordsBool = htm.clone_width_one_interior_points(a, yp, coords)

# Plot the Vector field potential for a given setup
if(plotVectorField):
  htm.plot_vector_potential(coords, latticePlotPath)

# Now that we have both coordinates and a hollow triangle polygon we want to make a centroids mask to be used later for plotting the wavefunction. returns a masked triang
if(width != 1):
  triang = htm.create_centroids_mask(a, coords, innertri)
else:
  innertri = htm.shifted_innertri(a, yp, outerlen)
  newCoords = np.vstack([coords,clonedCoords])
  triang = htm.create_centroids_mask(a, newCoords, innertri)

# Loop through the varying values of B for the vector potential
nE = 6 # must be even?
nk = 10
dB = B0 / nk
Bmults = int(3)
Bmax = Bmults * B0
if(Bmults>1):
  bvals = np.linspace(0,Bmax,Bmults*nk+1)
  evb = np.zeros((2*nE,Bmults*nk+1))
else:
  bvals = np.linspace(0,B0,nk)
  evb = np.zeros((2*nE,nk))

for k,values in enumerate(bvals):
  # Construct the BdG Hamiltonian
  n = np.size(coords[:,0])
  bdg = np.zeros((2*n,2*n), dtype='complex')
  for i in range(n):
    for j in range(n-i):
      dx = coords[i+j,0] - coords[i,0]
      dy = coords[i+j,1] - coords[i,1]
      d = np.sqrt(dx**2 + dy**2)

      # Self site
      if d<1e-5:
        bdg[i,i] = -mu
        bdg[i+n,i+n] = -bdg[i,i]

      # Nearest-neighbor
      elif np.abs(d-a)<1e-5:
        phase = np.arctan(dy/dx)
        if dx<0:
          phase += np.pi
#        phi = - (values/2) * (dy/dx) * (coords[i+j,0]**2 - coords[i,0]**2)
        if(vecPotFunc=='step-function'):
          # not quite right
          phi = values * dy * np.sign(coords[i,0])
        elif(vecPotFunc=='linear'):
          phi = - (values/2) * (dy/dx) * (coords[i+j,0]**2 - coords[i,0]**2)
        bdg[i+j,i] = -t*np.exp(1.0j*phi)
        bdg[i+j+n,i+n] = t*np.exp(-1.0j*phi)
        bdg[i+n,i+j] = delta*np.exp(-1.0j*phase)
        bdg[i+j+n,i] = -delta*np.exp(-1.0j*phase)
      # Else nothing

  # Solve the eigenvalue problem for energies only
  eng, vec = np.linalg.eigh(bdg)
  evb[:,k] = eng[n-nE:n+nE]
  vec = np.real(np.multiply(vec, np.conj(vec)))
  vvb = vec[:,n-nE:n+nE]

  # Let's only plot the wavefunction of the states if we have a MF or MF-like state for a given vector potential strength
  if(abs(evb[nE,k]) < 1E-5):
    if(width != 1):
      htm.plot_hollow_triangle_wavefunction(a, width, innertri, coords, triang, nE, values, evb[:,k], vvb, filepath)
    else:
      # copy the eigenstate elements to the correct locations!
      # split
      v0 = vvb[0:n,:]
      v00 = vvb[n:2*n,:]
      # append in the cloned coordinates eigenstate values
      v1 = np.vstack([v0,v0[clonedCoordsBool,:]])
      v2 = np.vstack([v00,v00[clonedCoordsBool,:]])
      # recombine the two sets to get the intended state vector
      newvec = np.vstack([v1,v2])
      htm.plot_hollow_triangle_wavefunction(a, width, innertri, np.vstack([coords,clonedCoords]), triang, nE, values, evb[:,k], newvec, filepath)

  ## Save data
  #np.savetxt('./data/hollow-triangle-energy.txt', eng, fmt='%1.8e')

if(plotSpectral):
  htm.plot_hollow_triangle_spectral_flow(mu, outernr, B0, width, Bmax, nE, bvals, evb, filepath
      )
