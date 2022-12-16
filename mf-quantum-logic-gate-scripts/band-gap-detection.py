#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from pfapack import pfaffian as pf


# Define parameters
PI = np.pi
PBC = True
PlotFlag = False
t = 1
delta = t
#mu = -0.8
nmu= 15
muvals = np.linspace(0, 2.0,nmu+1)
a = 1
n = 85

#x = np.array([ (i - (n-1) / 2) * a for i in range(n)])
#XX = (a**2 + 2*x[:-1])/2
x = np.array([ a for i in range(n)])
XX = x[:-1]

# Create B values for the vector potential strength
Bmax = 1*PI/a
nB = 2*nmu
Bvals = np.linspace(0*PI,Bmax,nB+1)
eob = np.zeros((2*n,nB+1))

U = np.sqrt(0.5) * np.matrix([[1,1],[-1.0j,1.0j]]) / n
U = np.kron(U,np.identity(n))
majNum = np.zeros((nmu+1,nB+1))

bdg = np.zeros((2*n,2*n), dtype='complex')
delarr = delta*np.ones(n-1)
bdg[n:2*n, 0:n] = np.diag(delarr, k=1) - np.diag(delarr, k=-1)
bdg[0:n, n:2*n] = -bdg[n:2*n, 0:n]
## Add in the PBC terms, can turn it off earlier in the code
if(PBC==True):
  bdg[n-1,0] = -t
  bdg[0,n-1] = -t
  bdg[2*n-1,n] = t
  bdg[n,2*n-1] = t
  bdg[2*n-1,0] = delta
  bdg[0,2*n-1] = delta
  bdg[n,n-1] = -delta
  bdg[n-1,n] = -delta

for j, muvalues in enumerate(muvals):
  for i, bvalues in enumerate(Bvals):
    tarr = t*np.exp(1.0j*bvalues*XX)
    ## quick way to build the BdG Hamiltonian since it's a linear chain
    bdg[0:n, 0:n] = -muvalues*np.eye(n) - np.diag(tarr,k=-1) - np.diag(np.conjugate(tarr),k=1)
    bdg[n:2*n, n:2*n] = -np.conjugate(bdg[0:n, 0:n])

    # Solve the eigenvalue problem for energies only
    eng = np.linalg.eigvalsh(bdg)

    ## Calculate the Majorana number for each vector field strength
    A = -1.0j * U * bdg * np.conjugate(np.transpose(U))
    np.savetxt('./data/top-inv-a.txt', np.real(A), fmt='%1.1f')
    #majNum[j,i] = np.sign(pf.pfaffian(A))
    majNum[j,i] = np.sign(np.real(pf.pfaffian(A)))*np.abs(eng[n]-eng[n-1])

np.savetxt('./data/top-inv-majorana-number.txt', majNum)
print('Finished')
#print(majNum)
## May have to switch between +1 and -1 to find 'gap closings/openings', I can't find a range of B values that are of different sign.
#print(np.where(majNum==1))
#print(Bvals[np.where(majNum==1)[0]])

## Can turn off earlier in the script if needed
#if(PlotFlag==True):
#  plt.figure(figsize=(2,2))
#  for i in range(2*n):
#    plt.plot(Bvals,eob[i,:], ':b')
#  plt.plot(Bvals,majNum[nmu//2+1,:], '.r')
#  plt.tight_layout()
#  plt.show()
#  plt.close()
if(PlotFlag==True):
  plt.figure()
  plt.imshow(majNum, cmap='Blues_r')
  plt.show()
  plt.close()



