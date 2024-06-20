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
#a = 1e-9 # [m]
m = 0.067 * m_e # GaAs/AlGaAs: m = 0.067m_e
m = 1.0 * m_e # [MeV/c^2]
h = hbar**2 / (2 * m * a**2) # hopping value for square lattice
print(h)
k = 0 # Momentum space momentum value
ka = k*a

# Laser parameters
Bmax = 1*pi*hbar/a**2 # [T=Vs/m^2]

# Matrix cutoffs
# Size of system in x direction
rc = 35
xm = rc*a
Nr = 2*rc+1
xj = np.linspace(-xm, xm, Nr)

# Range of Electric field strength and
phimin = Bmax/hbar/25
phimax = 1e8 # unitless
phimax = Bmax/hbar/20 # unitless
nphi = 1*100
phi0 = np.linspace(phimin,phimax,nphi)

# Create empty arrays
energy = np.zeros( (Nr, nphi) )
#energy = np.zeros( (Nr, nphi) )
Adotdl = a*xj

# For loop over the phi0 terms
for i, phi in enumerate(phi0):

  H = -2*h*np.diag(np.cos(ka-phi*Adotdl), k=0) - h*np.diag(np.ones(Nr-1),k=-1)

  #eng = np.linalg.eigvalsh(Q, UPLO = 'L')
  eng, vec = np.linalg.eigh(H, UPLO = 'L')
  vec = np.real(np.multiply(vec, vec.conj()))
  energy[:,i] = eng
  #energy[:,i] = eng[mc*Nr:(mc+1)*Nr]
  np.savetxt('./data/eigenstate-phi-%03i.txt' % (i), vec, fmt = '%1.8f')


np.savetxt('./data/eng-matrix.txt', energy, fmt='%1.8f')

np.savetxt('./data/config.txt', [rc, 0, h, phimin, phimax, nphi])
