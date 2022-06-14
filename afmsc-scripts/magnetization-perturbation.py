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
g = 0.763
mu = 1.3
lamb = 2.1
delta = 1.45
j = 4.321
A = 0.0001

# Define the k-space to be integrated over (0,Kmax) and (0,2pi)
kr = 10*np.pi
nk = 100
k = np.linspace(0.,kr,nk)
nphi = 4*360
phi = np.linspace(0,2*np.pi,nphi)
K, Phi = np.meshgrid(k, phi)
dphi = 2*np.pi/nphi
dk = kr/nk

# Construct Hamiltonians in k-space
eps = np.kron( (hbar*K)**2 / (2*mass) - mu , np.kron(sz,s0) )
ly = np.kron( lamb*K*np.sin(Phi), np.kron(s0,sx) )
lx = np.kron( lamb*K*np.cos(Phi), np.kron(sz,sy) )
delta = np.kron( np.ones( (nphi, nk) ), delta*np.kron(sy,sy) )
jz = np.kron( np.ones( (nphi, nk) ), j*np.kron(sz,sz) )

hA = -A * ( np.kron( hbar*K*np.cos(Phi)/mass, np.kron(s0,s0) ) - lamb/hbar * np.kron( np.ones( (nphi,nk) ), np.kron(s0,sy) ) )
h0 = eps + ly - lx - delta + jz

# save these for later when calculating <m>
mx = g*np.kron(sz,sx)
my = g*np.kron(s0,sy)
mz = g*np.kron(sz,sz)
magx = 0
magy = 0
magz = 0

# To compute <m> we need to solve the eigensystem of h0 to calculate the
# correction eigenvector required in the integral (summation)
for i in range(nphi):
  I = 4*i
  for j in range(nk):
    J = 4*j
    eng, vec = np.linalg.eigh(h0[I:I+4,J:J+4])

    # calculate correction terms
    dpsi = np.zeros((4,4), dtype='complex')
    for l in range(4):
      for m in range(4):
        if l==m or eng[l]==eng[m]:
          dpsi[:,l] +=0
        else:
          dpsi[:,l] += np.matmul( vec[:,m].conj().T, np.matmul( hA[I:I+4,J:J+4], vec[:,l] ) ) * vec[:,m] / ( eng[l] - eng[m] )

    # calculate the magnetization for each energy and sum them into the corresponding <m_i>
    # for the fermi-dirac distribution we set T=0, leaving us with a factor of 1 or 0, two values (negative) are below the Fermi energy, these contribute while the other two (positive) do not.
    for l in range(4):
      if eng[l] < 0:
        magx += K[0,j]*dk*dphi*2*np.real( np.sum( np.matmul( vec[:,l].conj().T, np.matmul( mx, dpsi[:,l] ) ) ) )
        magy += K[0,j]*dk*dphi*2*np.real( np.sum( np.matmul( vec[:,l].conj().T, np.matmul( my, dpsi[:,l] ) ) ) )
        magz += K[0,j]*dk*dphi*2*np.real( np.sum( np.matmul( vec[:,l].conj().T, np.matmul( mz, dpsi[:,l] ) ) ) )

print(magx)
print(magy)
print(magz)
print()
