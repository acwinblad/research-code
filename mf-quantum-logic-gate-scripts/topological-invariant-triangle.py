#!/usr/bin/python

import hollow_triangle_module as htm
import numpy as np
import matplotlib.pyplot as plt
from pfapack import pfaffian as pf


PBC = True
# Define parameters
t = 1
delta = t
nmu = 10
muvals = np.linspace(-6*t,6*t,nmu+1)
a = 1
nr = 10
# critical vector potential magnitude for linear vector potential
B0 = 8 * np.pi / (3 * np.sqrt(3) * a**2 * (2 * nr - 3) )
# use the hollow triangle module to create a filled triangle lattice quickly
width = 1
yp = np.sqrt(3)/2
tri, innertri = htm.build_hollow_triangle(a, nr, a*(nr-1), width, yp)
coords = htm.hollow_triangle_coords(a, yp, nr, tri)

# Loop through the varying values of B for the vector potential
nB = 20
dB = B0 / nB
Bmults = int(4)
Bmax = Bmults * B0
if(Bmults>1):
  bvals = np.linspace(-Bmax,Bmax,Bmults*nB+1)
else:
  bvals = np.linspace(0,B0,nB)

n = np.size(coords[:,0])
U = (1/np.sqrt(2)) * np.matrix([[1,1],[-1.0j,1.0j]])
U = np.kron(U,np.identity(n))

majNum = np.zeros((np.size(muvals),np.size(bvals)))

for l, muval in enumerate(muvals):
  for k, bval in enumerate(bvals):
    # Construct the BdG Hamiltonian
    bdg = np.zeros((2*n,2*n), dtype='complex')
    for i in range(n):
      for j in range(n-i):
        dx = coords[i+j,0] - coords[i,0]
        dy = coords[i+j,1] - coords[i,1]
        d = np.sqrt(dx**2 + dy**2)

        # Self site
        if(d<1e-5):
          bdg[i,i] = -muval/2
          bdg[i+n,i+n] = -bdg[i,i]

        elif(np.abs(d-a)<1e-5):
          phase = np.arctan(dy/dx)
          if dx<0:
            phase += np.pi
          phi = - (bval/2) * (dy/dx) * (coords[i+j,0]**2 - coords[i,0]**2)
          bdg[i+j,i] = -t * np.exp(1.0j*phi)
          bdg[i+j+n,i+n] = t * np.exp(-1.0j*phi)
          bdg[i+n,i+j] = delta * np.exp(-1.0j*phase)
          bdg[i+j+n,i] = -delta * np.exp(-1.0j*phase)

        # Else nothing
    if(PBC==True):
      bdg[n-1,n-nr] = -t
      bdg[2*n-1,2*n-nr] = t
      bdg[2*n-nr,n-1] = delta
      bdg[2*n-1,n-nr] = -delta

    bdg = bdg + np.conjugate(np.transpose(bdg))
    A = -1.0j * U * bdg * np.conjugate(np.transpose(U))
    np.savetxt('./data/top-inv-a.txt', np.real(A), fmt='%1.4f')
    eng, vec = np.linalg.eigh(bdg)
    #vec = np.real(np.multiply(vec, np.conj(vec)))
    majNum[l,k] = np.sign(np.real(pf.pfaffian(A)))
    #print(majNum, bval)

np.savetxt('./data/majnum.txt', majNum, fmt='%i')

#xaxis = np.array([-1,1])
plt.figure(figsize=(2,4))
#plt.xlim(-1.2,1.2)
#for i in range(2*n):
#  plt.plot(xaxis, [eng[i], eng[i]], 'b', linewidth=0.5)
plt.imshow(majNum)
plt.tight_layout()
plt.show()
plt.close()


