#!/usr/bin/python3

import numpy as np
import os
import glob
import pathlib

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

# Laser parameters
hw = 191e-3 # [meV]
Emax = 2e9 # V/m
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

filepath = './data/rc-%02i-mc-%02i/' % (rc, mc)
pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)

phimin = 0
#phimax = 1e9 # unitless
phimax = Emax*a/hw # unitless

nphi = 100
phi0 = np.linspace(phimin,phimax,nphi)
energy = np.zeros( (Nm*4*Nr, nphi) )
#energy = np.zeros( (4*Nr, nphi) )
H = np.zeros( (Nm,4*Nr,4*Nr) , dtype='complex')

# Build global arrays for time domain fourier transform
tau = np.linspace(0, 2*pi, 51)
s1 = np.sin(tau)
c2 = np.cos(2*tau)

# Coordinates for respective hopping terms
dx = (np.sqrt(3)/2) * np.array([1,0,1,1,1,0])
dy = (1/2) * np.array([1,2,1,-1,-1,2])
xavg = (np.sqrt(3)*a/4) * np.array([1,2,3,3,5,4])
Adotdl = np.zeros( (6, Nr, np.size(tau)) )
for i in range(6):
  Adotdl[i] = -dx[i]*s1 + 0.5 * np.outer(dy[i]*np.sin(K*(xj+xavg[i])), c2)

# Compute the indices for H_n and form the block matrix
def create_H_n(_n, _phi, _ka):
  hjla1b1 = -h* np.trapz(np.exp(1.0j * (_phi * Adotdl[0] - _n*tau) ), x=tau, axis=1) / (2*pi)
  hjlb1a2 = -h* np.trapz(np.exp(1.0j * (_phi * Adotdl[1] - _n*tau) ), x=tau, axis=1) / (2*pi)
  hjla2b2 = -h* np.trapz(np.exp(1.0j * (_phi * Adotdl[2] - _n*tau) ), x=tau, axis=1) / (2*pi)

  hjp1lb1a1 = -h* np.trapz(np.exp(1.0j * (_phi * Adotdl[3] - _n*tau) ), x=tau, axis=1) / (2*pi)
  hjp1lb2a2 = -h* np.trapz(np.exp(1.0j * (_phi * Adotdl[4] - _n*tau) ), x=tau, axis=1) / (2*pi)
  hjp1lp1b2a1 = -h* np.trapz(np.exp(1.0j * (_phi * Adotdl[5] - 3*_ka - _n*tau) ), x=tau, axis=1) / (2*pi)

  Hb = np.zeros((4*Nr,4*Nr), dtype='complex')

  for xidx, xval in enumerate(xj):
    Hb[4*xidx:4*(xidx+1),4*xidx:4*(xidx+1)] = np.array([[0,hjla1b1[xidx],0,0],[0,0,hjlb1a2[xidx],0],[0,0,0,hjla2b2[xidx]],[0,0,0,0]])
  for xidx, xval in enumerate(xj[:-1]):
    Hb[4*xidx:4*(xidx+1),4*(xidx+1):4*(xidx+2)] = np.array([[0,0,0,0],[hjp1lb1a1[xidx],0,0,0],[0,0,0,0],[hjp1lp1b2a1[xidx],0,hjp1lb2a2[xidx],0]])

  Hb = Hb + Hb.conj().T
  return Hb


# For loop over the phi0 terms
for i, phi in enumerate(phi0):

  # Calculate individual block matrices for given phi0 and construct Q matrix
  # Clear the Q matrix since we will be adding to the matrix
  Q = np.zeros( (Nm*4*Nr,Nm*4*Nr) , dtype = 'complex')
  for n in range(Nm):
    # To construct the H_n matrices we must numerically solve the time domain Fourier transform for each matrix index
    H[n] = create_H_n(n, phi, ka)

    # Quick way to fill the Q matrix is to use kronecker functions for the block diagonals (and subdiagonals H_1, H_2, ...)
    Q += np.kron(np.diag(np.ones(Nm-n),k=-n),H[n])

  # Quickly apply the modes along the diagonal of Q
  Q += np.kron(np.diag(nhw,k=0),np.eye(4*Nr))

  eng, vec = np.linalg.eigh(Q, UPLO = 'L')
  energy[:,i] = eng
  #energy[:,i] = eng[(mc+0)*4*Nr:(mc+1)*4*Nr]
  np.savetxt(filepath+'eigenstate-phi-%03i.txt' % (i), vec, fmt = '%1.8f')


np.savetxt(filepath+'eng-matrix.txt', energy, fmt='%1.8f')

np.savetxt(filepath+'config.txt', [rc, mc, h, phimin, phimax, nphi])
np.savetxt('./data/config.txt', [rc, mc, h, phimin, phimax, nphi])
