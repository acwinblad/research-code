#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from pfapack import pfaffian as pf


# Define parameters
PI = np.pi
PBC = True
PlotFlag = True
t = 1
delta = t
#mu = -0.8
nmu= 120
muvals = np.linspace(-1.8,0,nmu+1)
a = 1
## The size didn't seem to affect the PI/15 and mu=0 result
n = 16

x = np.array([ (i - (n-1) / 2) * a for i in range(n)])
XX = (a**2 + 2*a*x[:-1])/2

## Tweaking the magnitude (PI/15 seems to be a nice number for mu=0)
Bmax = PI
nB = 125
Bvals = np.linspace(0,Bmax,nB+1)
eob = np.zeros((2*n,nB+1))

U = np.sqrt(0.5) * np.matrix([[1,1],[-1.0j,1.0j]])
U = np.kron(U,np.identity(n))
majNum = np.zeros((nmu+1,nB+1))

for j, muvalues in enumerate(muvals):
  for i, bvalues in enumerate(Bvals):
    ## quick way to build the BdG Hamiltonian
    h1 = -muvalues*np.eye(n) - t*(np.diag(np.exp(1.0j*bvalues*XX),k=-1) + np.diag(np.exp(-1.0j*bvalues*XX),k=1))
    h2 = delta*(np.diag(np.ones(n-1),1)-np.diag(np.ones(n-1),-1))
    ht = np.hstack((h1,np.transpose(h2)))
    hb = np.hstack((h2,-np.transpose(h1)))
    h = np.vstack((ht,hb))
    ## Add in the PBC terms, can turn it off earlier in the code
    if(PBC==True):
      h[n-1,0] = -t
      h[0,n-1] = -t
      h[2*n-1,n] = t
      h[n,2*n-1] = t
      h[2*n-1,0] = delta
      h[0,2*n-1] = delta
      h[n,n-1] = -delta
      h[n-1,n] = -delta
    eng, vec = np.linalg.eigh(h)
    vec = np.real(np.multiply(vec, np.conj(vec)))
    #eob[:,i] = eng
    ## Calculate the Majorana number for each vector field strength
    A = -1.0j * U * h * np.conjugate(np.transpose(U))
    np.savetxt('./data/top-inv-a.txt', np.real(A), fmt='%1.1f')
    majNum[j,i] = np.sign(pf.pfaffian(A))

#print(majNum)
## May have to switch between +1 and -1 to find 'gap closings/openings', I can't find a range of B values that are of different sign.
#print(np.where(majNum==1))
#print(Bvals[np.where(majNum==1)[0]])

## Can turn off earlier in the script if needed
#if(PlotFlag==True):
#  plt.figure(figsize=(2,2))
#  for i in range(2*n):
#    plt.plot(Bvals,eob[i,:], ':b')
#  plt.plot(Bvals,majNum, '.r')
#  plt.tight_layout()
#  plt.show()
#  plt.close()
if(PlotFlag==True):
  plt.figure()
  plt.imshow(majNum, cmap='summer')
  plt.show()
  plt.close()


