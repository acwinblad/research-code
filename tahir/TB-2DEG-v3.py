#!/usr/bin/python3
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp
np.set_printoptions(linewidth=np.inf, precision=4)

# Define radius to be used and determine the dimensions of the Bloch matrices for each mode
rc = 1
Ns = 2*rc+1

# Determine the number of modes to look at
mc = 1
Nm = 2*mc+1

# initialize parameters
#ka = 0.1
#hw = 4*np.pi

# phi_0 will be a free variable and be used for a for loop potential?
nphi = 1
phimax = 40
phi0 = np.linspace(0,phimax,nphi)
energy = np.zeros((Nm*Ns,nphi))

for l in range(nphi):
  # build up the matrix
  hjj = np.zeros(Nm*Ns, "complex")
  hjl = np.zeros(Nm*(Ns-1), "complex")

  for i in range(Nm*Ns):
      # since 'n' is used frequently we will define it here
      # in python3 integer divide is taken care of by using '//' instead of '/'
      n = mc - i//Nm

      # for kj we need to mod the integer first then multiply the Ka term in, otherwise it will mod Ka*i and give the wrong value
      kj = (i%Ns)*ka
      cs = abs(np.cos(kj))
      hjj[i] = -2.*sp.jv(n,phi0*cs)*np.exp(1.0j*n*np.pi/2)
      while(i<Nm*Ns-1):
        

  for i in range(Nm):
      n = mc - i
      Jl[i] = -(-1)**n*sp.jv(n,phi0[l])

  Q = np.diag(Jd)

  for i in range(Nm*Ns-1):
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


xaxis = np.array([-1, 1])
plt.figure(figsize=(10,10))
plt.tick_params(
    axis='x',
    which='both',
    bottom='on',
    top='off',
    labelbottom='on')
plt.ylabel('$E(\phi_0)$', fontsize=12)
plt.xlabel('$\phi_0$', fontsize=12)
plt.xlim(phi0[0], phi0[-1])
#plt.ylim(np.min(energy),np.max(energy))
plt.ylim(-5, 5)


for i in range(Nm*Ns):
  plt.plot(phi0, energy[i,:], '.', color = 'b', markersize = 1.5)
  #plt.plot(phi0, energy[i,:], color = 'b', markersize = .1)

#for i in range(Nm*Ns):
#  plt.plot(xaxis,[energy[i,:],energy[i,:]], 'g')

plt.tight_layout()

plt.show()
