#!/usr/bin/python3

import numpy as np
import os
import glob

files = glob.glob('./data/eig*')
for f in files:
  os.remove(f)

# Constants
hbar = 6.582E-16 # 6.582 * 10^-16 [eV*s]
c = 2.998E8 # 2.998 * 10^8 [m/s]
q = 1.602E-19 # [C]
m_e = 0.51E6 / c**2 # [MeV/c^2]
pi = np.pi

# System parameters
a = 0.142E-9 # [m] lattice constant for graphene
m = m_e # [MeV/c^2]
v_f = 1E6 # [m/s] fermi velocity for graphene
h = 2*v_f*hbar/(3*a) # [eV] hopping energy for graphene based on formula? This value is close to the 2.8 eV
h = 2.8 # [eV] hopping energy for graphene is 2.8 eV
k = 0.0*pi/(3*a) # [m^-1] Momentum space wavenumber Probably shouldn't be zero for graphene
ka = k*a

# Magnetic parameters
Bmax = 13500 # T

# Size of system in x direction
rc = 6
xm = rc*a
Nr = 2*rc+1
xj = np.linspace(-xm, xm, Nr)

phimin = 0
phimax = Bmax*a**2/hbar # unitless

nphi = 100
phi0 = np.linspace(phimin,phimax,nphi)
#Ka = np.linspace(-k*a,k*a,nphi)
phi0 = np.array([ (phimin + i/nphi)**(1/1) for i in range(nphi) ] ) * phimax
energy = np.zeros( (4*Nr, nphi) )
#energy = np.zeros( (4*Nr, nphi) )
H = np.zeros( (4*Nr,4*Nr) , dtype='complex')

# Coordinates for respective hopping terms
dx = (np.sqrt(3)/2) * np.array([1,0,1,1,1,0])
dy = (a/2) * np.array([1,2,1,-1,-1,2])
xavg = (np.sqrt(3)*a/4) * np.array([1,2,3,3,5,4])
Adotdl = np.zeros( (6, Nr) )
for i in range(6):
  Adotdl[i] = (xj+xavg[i]) * dy[i] / a**2

# Compute the indices for H_n and form the block matrix
def create_H_n(_phi, _ka):
  hjla1b1 = -h * np.exp(1.0j * _phi * Adotdl[0])
  hjlb1a2 = -h * np.exp(1.0j * _phi * Adotdl[1])
  hjla2b2 = -h * np.exp(1.0j * _phi * Adotdl[2])

  hjp1lb1a1   = -h * np.exp(1.0j * _phi * Adotdl[3])
  hjp1lb2a2   = -h * np.exp(1.0j * _phi * Adotdl[4])
  hjp1lp1b2a1 = -h * np.exp(1.0j * _phi * Adotdl[5] - 3*_ka)

  Hb = np.zeros((4*Nr,4*Nr), dtype='complex')

  for xidx, xval in enumerate(xj):
    Hb[4*xidx:4*(xidx+1),4*xidx:4*(xidx+1)] = np.array([[0,hjla1b1[xidx],0,0],[0,0,hjlb1a2[xidx],0],[0,0,0,hjla2b2[xidx]],[0,0,0,0]])
  for xidx, xval in enumerate(xj[:-1]):
    Hb[4*xidx:4*(xidx+1),4*(xidx+1):4*(xidx+2)] = np.array([[0,0,0,0],[hjp1lb1a1[xidx],0,0,0],[0,0,0,0],[hjp1lp1b2a1[xidx],0,hjp1lb2a2[xidx],0]])

  Hb = Hb + Hb.conj().T
  return Hb

# For loop over the phi0 terms
for i, phi in enumerate(phi0):
#for i, Kaval in enumerate(Ka):

  # To construct the H_n matrices we must numerically solve the time domain Fourier transform for each matrix index
  H = create_H_n(phi, ka)
  #H = create_H_n(0, Kaval)

  eng, vec = np.linalg.eigh(H, UPLO = 'L')
  energy[:,i] = eng
  #energy[:,i] = eng[(mc+0)*4*Nr:(mc+1)*4*Nr]
  np.savetxt('./data/eigenstate-phi-%03i.txt' % (i), vec, fmt = '%1.8f')


np.savetxt('./data/eng-matrix.txt', energy, fmt='%1.8f')

np.savetxt('./data/config.txt', [rc, 0, h, phimin, phimax, nphi])
