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
# For a triangular chain n = 3*(n_r-1)
n = 3*(nr-1)
# Need a width for the filename path
width = 1
width *= a
yp = np.sqrt(3)*a/2

vecPotFunc = 'step-function'
#vecPotFunc = 'linear'
#vecPotFunc = 'constant'
#vecPotFunc = 'tanh'
if(vecPotFunc=='step-function'):
  A0 = 2 * np.pi / (3*np.sqrt(3) * a)
  #A0 = 3.0 / a
elif(vecPotFunc=='tanh'):
  A0 = 2 * np.pi / (3 * np.sqrt(3) * a)
  A0 = 3.0 / a
elif(vecPotFunc=='linear'):
  A0 = 8 * np.pi / (3 * np.sqrt(3) * a**2 * (2 * nr - 3) )
elif(vecPotFunc=='constant'):
  A0 = 4 * np.pi / (np.sqrt(3) * a) / 2
else:
  print('pick a vector potential type from the following: step-function, tanh, linear, or constant')

# Populate what the directory hierarchy will be
latticePlotPath, filepath = htm.create_directory_path(vecPotFunc+'-rotation', mu, nr, width)

# Create a coordinate list so we can rotate the vector potentail field
coords = np.empty((1,2))
# Start bottom left and move right
for i in range(nr-1):
  xi = -(nr-1)*a/2 + i*a
  yi = -yp*(nr-1)/3
  coords = np.append(coords, np.array([[xi,yi]]), axis=0)
# Move from bottom right to top middle
for i in range(nr-1):
  xi = (nr-1)*a/2 -i*a/2
  yi = -yp*(nr-1)/3 + yp*i
  coords = np.append(coords, np.array([[xi,yi]]), axis=0)
# Move from top middle to bottom left
for i in range(nr-1):
  xi = -i*a/2
  yi = 2*yp*(nr-1)/3 - yp*i
  coords = np.append(coords, np.array([[xi,yi]]), axis=0)

coords = np.delete(coords,0,0)
dxdy = np.diff(coords,axis=0)
dxdy = -np.append(dxdy, [coords[0,:]-coords[-1,:]], axis=0)

# Loop through the varying angles of t for the vector potential
nE = 1*2 # must be even?
nt = 30
tf = np.pi
tvals = np.linspace(0,tf,nt+1)
evt = np.zeros((2*nE,nt+1))

# Initialize BdG Hamiltonian
n = np.size(coords[:,0])
bdg = np.zeros((2*n,2*n), dtype='complex')

# Chemical potential will remain constant and is only along the diagonal
bdg[0:n, 0:n] = -mu*np.eye(n)
bdg[n:2*n, n:2*n] = mu*np.eye(n)

# the order parameter will not change so we only need to initialize it once
for j in range(n-1):
  l = (j+1) % n
  thetaftr = htm.calc_phase_factor(dxdy[j,0],dxdy[j,1])
  bdg[j, n+l] = delta*thetaftr
  bdg[l, n+j] = -bdg[j, n+l]
thetaftr = htm.calc_phase_factor(dxdy[-1,0],dxdy[-1,1])
bdg[0, 2*n-1] = delta*thetaftr
bdg[n-1, n] = -bdg[0, 2*n-1]

for k, angle in enumerate(tvals):
  # Construct the BdG Hamiltonian for varying vector potential rotation angles
  for j in range(n-1):
    l = (j+1) % n
    phi = htm.calc_phi(a, coords[j,0], coords[l,0], coords[j,1], coords[l,1], dxdy[j,0], dxdy[j,1], angle, vecPotFunc)
    bdg[j, l] = -t*np.exp(1.0j*A0*phi)
    bdg[n+j, n+l] = -np.conjugate(bdg[j, l])
  phi = htm.calc_phi(a, coords[-1,0], coords[0,0], coords[-1,1], coords[0,1], dxdy[-1,0], dxdy[-1,1], angle, vecPotFunc)
  bdg[0, n-1] = -t*np.exp(1.0j*A0*phi)
  bdg[n, 2*n-1] = -np.conjugate(bdg[0, n-1])

  # Solve the eigenvalue problem for energies only
  eng, vec = np.linalg.eigh(bdg/2, UPLO = 'U')
  evt[:,k] = eng[n-nE:n+nE]
  vec = np.real(np.multiply(vec, np.conj(vec)))
  vvt = vec[:,n-nE:n+nE]

  if(angle % (np.pi/6) <= 1e-5):
    htm.plot_chain_triangle_wavefunction(a, coords, nE, angle, evt[:,k], vvt, filepath)


  ## Save data
  #np.savetxt('./data/hollow-triangle-energy.txt', eng, fmt='%1.8e')
  #np.savetxt('./data/triangular-chain-bdg-copy.txt', bdg, fmt='%0.1f')

if(plotSpectral):
  htm.plot_hollow_triangle_spectral_flow(mu, nr, A0, width, tf, nE, tvals, evt, filepath
      )
