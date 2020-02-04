#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
np.set_printoptions(linewidth=np.inf, precision=6)

# Define radius to be used and determine the dimensions of the Bloch matrices for each mode
rc = 2
Ns = 2*rc+1

# Determine the number of modes to look at
mc = 2
Nn = 2*mc+1

# initialize parameters
ka = 0.01
hw = 2.

# phi_0 will be a free variable and be used for a for loop potential?
nphi = 10
phi0 = np.linspace(-1,1,nphi)
energy = np.zeros((Nn*Ns,nphi))

for l in range(nphi):
  # build up the matrix
  Jd = np.zeros(Nn*Ns)

  for i in range(Nn*Ns):
      # since 'n' is used frequently we will define it here
      # in python3 integer divide is taken care of by using '//' instead of '/'
      n = mc - i//Ns

      if n%2 == 1:
          # if 'n' is odd the Bessel funtions will cancel each other out leaving just
          Jd[i] = n*hw

      else:
          # else 'n' is even and the Bessel function combine, after using identities, to give $2 (-1)^n J_|n| ( phi_0 * |cos(kaj)| )
          # for kj we need to mod the integer first then multiply the Ka term in, otherwise it will mod Ka*i and give the wrong value
          kj = (i%Ns)*ka
          cs = abs(np.cos(kj))
          #Jd[i] = n*hw - 2*sp.jv(n,phi0*cs)*np.exp(1.0j*n*np.pi/2)
          Jd[i] = n*hw - 2*(-1)**n*sp.jv(n,phi0[l]*cs)

  #print(Jd)

  Ju = np.zeros(Nn)
  Jl = np.zeros(Nn)

  for i in range(Nn):
      n = mc - i
      Ju[i] = -sp.jv(n,phi0[l])
      Jl[i] = -(-1)**n*sp.jv(n,phi0[l])

  #print(Ju)
  #print(Jl)

  Q = np.diag(Jd)

  for i in range(Nn):
      j = 3*i
      Q[j,j+1] = Ju[i]
      Q[j+1,j+2] = Ju[i]
      Q[j+1,j] = Jl[i]
      Q[j+2,j+1] = Jl[i]

  #print(Q)

  # solve eigenvalue problem
  energy[:,l] = np.linalg.eigvals(Q)

  # print eigen-energies
  #print(energy)


xaxis = np.array([-1, 1])
plt.figure(figsize=(6,4))
plt.tick_params(
    axis='x',
    which='both',
    bottom='off',
    top='off',
    labelbottom='off')
plt.ylabel('$Energy$', fontsize=12)
plt.xlabel('$\phi_0$', fontsize=12)
#plt.xlim(-10.2, 10.2)
#plt.ylim(np.min(energy)-1, np.max(energy)+1)
#plt.ylim(-6, 20)


for i in range(nphi):
  plt.plot(phi0, energy[i,:], '.', color = 'r', markersize = 1.8)

plt.tight_layout()



plt.show()
