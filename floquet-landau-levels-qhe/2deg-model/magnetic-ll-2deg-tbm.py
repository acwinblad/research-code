#!/usr/bin/python3

import numpy as np
import scipy.special as sp
import os
import glob

files = glob.glob('./data/magnetic-eig*')
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
t = hbar**2 / (2 * m * a**2) # hopping value for square lattice
print(t)

k = 0 # Momentum space momentum value
ka = k*a
dE = 0.10
ecutoff = t*np.sqrt(12*dE)-4*t

# Laser parameters
#hw = 191E-3 # [191 meV]
hw = 100*t # [eV]
w = hw/hbar
#d = 100E-9 # [m] Spatial period of the electric field of oblique laser in x direction, make sure d>>a and not necessarily integer multiple
d = 1000*a # [m]
K = 2*pi/d # Spatial wavenumber of obilque light in x-direction

# Matrix cutoffs
# Size of system in x direction
rc = 25
xm = rc*a
Nr = 2*rc+1
xj = np.linspace(-xm, xm, Nr)

# Range of Electric field strength and
nE = 100
Efmin = 0 # [V/m]
Efmax = 1.0E9 # [V/m]
Efmax = 1E4 * hw / a
Efrange = np.linspace(Efmin,Efmax,nE)
alpha = K**2*hbar**2 / (m * hw**3) # [s/V]
beta = (K*hbar**3/hw**3)**2 / (16 * m**3)

# Create empty arrays
energy = np.zeros( (Nr, nE) )
#energy = np.zeros( (Nr, nE) )
#Adotdl = (2*xj+a)*a/2 # [m^2]
Adotdl = a*xj # [m^2]

# For loop over the Efrange terms
for i, Ef in enumerate(Efrange):

  #H = -2*t*np.diag(np.cos(ka - alpha * Ef**2 * Adotdl), k=0) - t*np.diag(np.ones(Nr-1), k=-1)
  B = np.sqrt( (Ef * K / w)**2/2 + (9/12)*(Ef**2*K**2/(m*w**3))**2 )
  B1 = Ef*K / (np.sqrt(2)*w)
  print(B, B1)
  H = -2*t*np.diag(np.cos(ka - B * Adotdl), k=0) - t*np.diag(np.ones(Nr-1), k=-1)
  #H[-1,0] = -t* np.exp(1.0j*alpha*Ef**2 * Adotdl[-1]/hbar)

  eng, vec = np.linalg.eigh(H, UPLO = 'L')
  #energy[:,i] = eng
  #energy[:,i] = eng - 0*beta*Ef**4
  energy[:,i] = eng + 1 * Ef**2 / (4*m*w**2) - 0 * (9*Ef**4*K**4) / (48*m**3*w**6)
  #energy[:,i] = eng[mc*Nr:(mc+1)*Nr]

  if(i==0):
    ecutidx = np.where(eng <= ecutoff)[0]
    ecutarr = np.zeros(Nr, dtype = 'int')
    ecutarr[ecutidx] = 1
    lowEngProj = vec[:,ecutarr]
    lowEngProj = np.zeros((Nr,Nr))
    for j, jv in enumerate(ecutidx):
      lowEngProj += np.real(np.outer(vec[:,j],vec[:,j]))

    np.savetxt('./data/bottom-band-projector.txt', lowEngProj, fmt = '%1.8f')
  np.savetxt('./data/magnetic-eigenstate-Ef-%03i.txt' % (i), vec, fmt = '%1.8f')


np.savetxt('./data/magnetic-eng-matrix.txt', energy, fmt='%1.8f')

np.savetxt('./data/magnetic-config.txt', [rc, 0, t, Efmin, Efmax, nE])
