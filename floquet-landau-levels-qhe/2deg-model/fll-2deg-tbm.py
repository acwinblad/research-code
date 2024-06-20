#!/usr/bin/python3

import numpy as np
import scipy.special as sp
import os
import glob

files = glob.glob('./data/eig*')
for f in files:
  os.remove(f)

filepath = './data/'
bbproj = np.loadtxt('./data/bottom-band-projector.txt')

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
t = hbar**2 / (2 * m * a**2) # hopping value for square lattice
#print(t)
k = 0 # Momentum space momentum value
ka = k*a

# Laser parameters
#hw = 191E-3 # [191 meV]
hw = 10.00*t # [eV]
#d = 100E-9 # [m] Spatial period of the electric field of laser in x direction, make sure d>>a and not necessarily integer multiple
d = 1000*a # [m]
K = 2*pi/d # Spatial wavenumber of laser light in x direction

# Matrix cutoffs
# Number of modes
mc = 2
Nm = 2*mc+1
nhw = np.arange(-mc,mc+1,1)*hw

# Size of system in x direction
rc = 25
xm = rc*a
Nr = 2*rc+1
xj = np.linspace(-xm, xm, Nr)
Kxj = K*xj

m0 = (mc+0)*Nr
mf = (mc+1)*Nr

# Range of Electric field strength
Efmin = 0.0E0 # [V/m]
Efmin = 2.25 * hw / a # [V/m]
Efmin = 0. * hw / a # [V/m]
Efmax = 1.0E9 # [V/m]
Efmax = 4.0 * hw / a # [V/m]
nEf = 100
Efrange = np.linspace(Efmin,Efmax,nEf)

# Create empty arrays
energy = np.zeros( (Nm*Nr, nEf) )
H = np.zeros( (Nm,Nr,Nr) )

# For loop over the Efrange terms
for i, Ef in enumerate(Efrange):

  # Calculate individual block matrices for given Ef and construct Q matrix
  # Clear the Q matrix since we will be adding to the matrix
  Q = np.zeros( (Nm*Nr,Nm*Nr) )
  for n in range(Nm):
    H[n] = -2*t*np.diag(sp.jv(n,(Ef*a/hw)*np.sin(Kxj)),k=0)*np.cos(ka-n*pi/2) - t * (-1)**n * np.diag(sp.jv(n,(Ef*a/hw)*np.ones(Nr-1)),k=1) - t * np.diag(sp.jv(n,(Ef*a/hw)*np.ones(Nr-1)), k=-1)
    # Quick way to fill the Q matrix is to use kronecker functions for the block diagonals (and subdiagonals H_1, H_2, ...)
    Q += np.kron(np.diag(np.ones(Nm-n),k=n),H[n])

  # Quickly apply the modes along the diagonal of Q
  Q += np.kron(np.diag(nhw,k=0),np.eye(Nr))

  #eng = np.linalg.eigvalsh(Q, UPLO = 'L')
  eng, vec = np.linalg.eigh(Q, UPLO = 'U')
  energy[:,i] = eng
  #energy[:,i] = eng[mc*Nr:(mc+1)*Nr]

  state = np.real(np.multiply( vec[m0:mf,:], vec[m0:mf,:].conj()))
  np.savetxt('./data/eigenstate-Ef-%03i.txt' % (i), vec[m0:mf,:], fmt = '%1.8f')
  np.savetxt('./data/eigenstate-full-Ef-%03i.txt' % (i), state, fmt = '%1.8f')

  vecm0 = vec[m0:mf,:]
  tmp = np.real(np.matmul(vecm0.conj().T, bbproj))
  #tmp = np.real(np.matmul(vecm0.conj().T, np.eye(Nr)))
  tmp = np.real(np.matmul(tmp, vecm0))
  weight = np.diag(tmp,k=0)
  np.savetxt('./data/bottom-band-weight-Ef-%03i.txt' % (i), weight, fmt = '%1.8f')


np.savetxt('./data/eng-matrix.txt', energy, fmt='%1.8f')

np.savetxt('./data/config.txt', [rc, mc, t, Efmin, Efmax, nEf])
