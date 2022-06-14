#!/usr/bin/python3

import numpy as np
np.set_printoptions(linewidth=np.inf, precision=4)


# Define pauli matrices for quick building of Hamiltonian
s0 = np.eye(2)
sx = np.array([[0,1.],[1.,0]])
sy = -1.0j*np.array([[0,1],[-1,0]])
sz = np.array([[1,0],[0,-1]])

# Define parameters
hbar = 1.
mass = 1.
g = 1.
mu = 1.
lamb = 0.1
delta = 0.01
jx = 1*1.231
jy = 1*2.963
jz = 1*10e+1*2.5
A = 1.

# Define the k-space to be integrated over (0,kmax) and (0,2pi)
kmax = 2*np.pi
nk = 250
Kr = np.linspace(0.,kmax,nk)
dk = kmax/nk

nphi = 1*360
Phi = np.linspace(0,2*np.pi,nphi, endpoint=False)
dphi = 2*np.pi/nphi

# Zeeman and Magnetization matrices
delta = delta * np.kron(sy,sy)
j = jx*np.kron(sz,sx) + jy*np.kron(s0,sy) + jz*np.kron(sz,sz)
mx = g*np.kron(sz,sx)
my = g*np.kron(s0,sy)
mz = g*np.kron(sz,sz)

# magnetization averages to be summed up
magx = 0
magy = 0
magz = 0
# To compute <m> we need to solve the eigensystem of h0 to calculate the
# correction eigenvector required in the integral (summation)
for i, phi in enumerate(Phi):
  for j, kr in enumerate(Kr):
    eps = ( (hbar*kr)**2 / (2*mass) - mu ) * np.kron(sz,s0)
    ly = lamb*kr*np.sin(phi) * np.kron(s0,sx)
    lx = lamb*kr*np.cos(phi) * np.kron(sz,sy)

    h0 = eps + ly - lx - delta + jz
    hA = -A * ( hbar*kr*np.cos(phi)/mass * np.kron(s0,s0) - lamb/hbar * np.kron(s0,sy) )

    eng, vec = np.linalg.eigh(h0)

    # calculate correction terms
    dpsi = np.zeros((4,4), dtype='complex')
    for l in range(4):
      for m in range(4):
        if l==m or eng[l]==eng[m]:
          dpsi[:,l] +=0
        else:
          dpsi[:,l] += np.matmul( vec[:,m].conj().T, np.matmul( hA, vec[:,l] ) ) * vec[:,m] / ( eng[l] - eng[m] )

    # calculate the magnetization for each energy and sum them into the corresponding <m_i>
    # for the fermi-dirac distribution we set T=0, leaving us with a factor of 1 or 0, two values (negative) are below the Fermi energy, these contribute while the other two (positive) do not.
    for l in range(4):
      if eng[l] < 0:
        magx += kr*np.real( np.sum( np.matmul( vec[:,l].conj().T, np.matmul( mx, dpsi[:,l] ) ) ) )
        magy += kr*np.real( np.sum( np.matmul( vec[:,l].conj().T, np.matmul( my, dpsi[:,l] ) ) ) )
        magz += kr*np.real( np.sum( np.matmul( vec[:,l].conj().T, np.matmul( mz, dpsi[:,l] ) ) ) )

magx *= dk*dphi/(2*np.pi)**2
magy *= dk*dphi/(2*np.pi)**2
magz *= dk*dphi/(2*np.pi)**2
print(magx)
print(magy)
print(magz)
print()
