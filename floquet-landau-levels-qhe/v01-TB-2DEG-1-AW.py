#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
np.set_printoptions(linewidth=np.inf, precision=4)

# Define radius to be used and determine the dimensions of the Bloch matrices for each mode
rc = 10
Ns = 2*rc+1

# Determine the number of modes to look at
mc = 5
Nn = 2*mc+1

# initialize parameters
ka = 0.1
hw = 1*np.pi

# phi_0 will be a free variable and be used for a for loop potential?
nphi = 100
phimax = 4
phi0 = np.linspace(-phimax,phimax,nphi)
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

  Jl = np.zeros(Nn)

  for i in range(Nn):
      n = mc - i
      Jl[i] = -(-1)**n*sp.jv(n,phi0[l])

  Q = np.diag(Jd)

  for i in range(Nn*Ns-1):
    ll = (i+1)%Ns
    mm = i//Ns
    if ll == 0:
      Q[i+1,i] = 0
    else:
      Q[i+1,i] = Jl[mm]


  # solve eigenvalue problem
  energy[:,l] = np.linalg.eigvalsh(Q)

  # print eigen-energies
print(energy)


#xaxis = np.array([-1, 1])
plt.figure(figsize=(4,4))
plt.tick_params(
    axis='x',
    which='both',
    bottom='off',
    top='off',
    labelbottom='off')
plt.ylabel('$Energy$', fontsize=12)
plt.xlabel('$\phi_0$', fontsize=12)
plt.xlim(phi0[0], phi0[-1])
plt.ylim(np.min(energy),np.max(energy))


for i in range(Nn*Ns):
  plt.plot(phi0, energy[i,:], '.', color = 'g', markersize = 1)
  #plt.plot(phi0, energy[i,:], color = 'b', markersize = .1)

plt.tight_layout()

plt.show()
