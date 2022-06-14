#!/usr/bin/python3

import numpy as np
np.set_printoptions(linewidth=np.inf, precision=4)


# Define pauli matrices for quick building of Hamiltonian
s0 = np.eye(2)
sx = np.array([[0,1.],[1.,0]])
sy = -1.0j*np.array([[0,1],[1,0]])
sz = np.array([[1,0],[0,-1]])


# Define parameters
hbar = 1.
mass = 1.
g = 0.763
mu = 1.3
lamb = 2.1
delta = 1.45
j = 4.321
A = 0.01


# Define the k-space to be integrated over (0,Kmax) and (0,2pi)
kr = 100.
nk = 1
k = np.linspace(0.,kr,nk)
nphi = 1
phi = np.linspace(0,2*np.pi,nphi)
K, Phi = np.meshgrid(k, phi)

# Construct Hamiltonians in k-space
eps = np.kron( (hbar*K)**2 / (2*mass) - mu , np.kron(sz,s0) )
lk = np.kron( lamb*np.sqrt(K), np.kron(sz,sz) )

hA = -A * ( np.kron( hbar*K*np.cos(Phi)/mass, np.kron(s0,s0) ) - lamb/hbar * np.kron( np.ones( (nphi,nk) ), np.kron(s0,sy) ) )

h0 = eps + lk

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
    Delta = np.array( [ [0, 0, -1.0j*np.exp(1.0j*Phi[i,0])*delta, 0],
                        [0, 0, 0, 1.0j*np.exp(1.0j*Phi[i,0])*delta],
                        [-1.0j*np.exp(-1.0j*Phi[i,0])*delta, 0, 0, 0],
                        [0, 1.0j*np.exp(-1.0j*Phi[i,0])*delta, 0, 0] ] , dtype='complex')

    eng, vec = np.linalg.eigh(h0[I:I+4,J:J+4] + Delta)
    print(eng)
    print(vec)
    print()
#    print(np.matmul(vec.conj().T, vec ) )

    # calculate correction terms
    dpsi = np.zeros((4,4), dtype='complex')
    for l in range(4):
      for m in range(4):
        if l==m or eng[l]==eng[m]:
          dpsi[:,l] += 0
        else:
          dpsi[:,l] += np.matmul( vec[:,m].conj().T, np.matmul( hA[I:I+4,J:J+4], vec[:,l] ) ) * vec[:,m] / ( eng[l] - eng[m] )

    print(dpsi)
    print()
    psicorrected = vec+dpsi
    print(psicorrected)
    print(np.matmul(psicorrected.conj().T, psicorrected ) )
