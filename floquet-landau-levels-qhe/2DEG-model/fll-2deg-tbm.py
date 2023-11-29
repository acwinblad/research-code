#!/usr/bin/python3

import numpy as np
import scipy.special as sp
import os
import glob

files = glob.glob('./data/eig*')
for f in files:
  os.remove(f)

# Constants
hbar = 6.582E-16 # 6.582 * 10^-16 [eV*s]
c = 2.998E8 # 2.998 * 10^8 [m/s]
q = 1.602E-19 # [C]
m_e = 0.51E6 / c**2 # [0.51 MeV/c^2]
pi = np.pi

# System parameters
a = 0.56E-9 # GaAs/AlGaAs: a = 0.56nm
a = 10e-9 # [m]
m = 0.067 * m_e # GaAs/AlGaAs: m = 0.067m_e
m = 1.0 * m_e # [MeV/c^2]
h = hbar**2 / (2 * m * a**2) # hopping value for square lattice
print(h)
k = 0 # Momentum space momentum value
ka = k*a

# Laser parameters
hw = 191e-3 # [191 meV]
Emax = 3e7 # [V/m]
d = 100E-9 # [m] Spatial period of the electric field of laser in x direction, make sure d>>a and not necessarily integer multiple
K = 2*pi/d # Spatial wavenumber of laser light in x direction

# Matrix cutoffs
# Number of modes
mc = 10
Nm = 2*mc+1
nhw = np.arange(-mc,mc+1,1)*hw

# Size of system in x direction
rc = 5
xm = rc*a
Nr = 2*rc+1
xj = np.linspace(-xm, xm, Nr)
Kxj = K*xj

# Range of Electric field strength and
phimin = 0
phimax = 1e8 # unitless
phimax = Emax*a/hw # unitless
nphi = 100
phi0 = np.linspace(phimin,phimax,nphi)

# Create empty arrays
energy = np.zeros( (Nm*Nr, nphi) )
#energy = np.zeros( (Nr, nphi) )
H = np.zeros( (Nm,Nr,Nr) )

# For loop over the phi0 terms
for i, phi in enumerate(phi0):

  # Calculate individual block matrices for given phi0 and construct Q matrix
  # Clear the Q matrix since we will be adding to the matrix
  Q = np.zeros( (Nm*Nr,Nm*Nr) )
  for n in range(Nm):
    H[n] = -2*h*np.diag(sp.jv(n,phi*np.cos(Kxj)),k=0)*np.cos(ka-n*pi/2) - h*np.diag(sp.jv(n,phi*np.ones(Nr-1)),k=1) - h*(-1)**n * np.diag(sp.jv(n,phi*np.ones(Nr-1)),k=-1)
    # Quick way to fill the Q matrix is to use kronecker functions for the block diagonals (and subdiagonals H_1, H_2, ...)
    Q += np.kron(np.diag(np.ones(Nm-n),k=-n),H[n])

  # Quickly apply the modes along the diagonal of Q
  Q += np.kron(np.diag(nhw,k=0),np.eye(Nr))

  #eng = np.linalg.eigvalsh(Q, UPLO = 'L')
  eng, vec = np.linalg.eigh(Q, UPLO = 'L')
  energy[:,i] = eng
  #energy[:,i] = eng[mc*Nr:(mc+1)*Nr]
  np.savetxt('./data/eigenstate-phi-%03i.txt' % (i), vec, fmt = '%1.8f')


np.savetxt('./data/eng-matrix.txt', energy, fmt='%1.8f')

np.savetxt('./data/config.txt', [rc, mc, h, phimin, phimax, nphi])
