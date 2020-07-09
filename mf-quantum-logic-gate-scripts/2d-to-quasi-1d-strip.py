#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

plotFlag = False

Vz = 50.
delta = 25.
t = 10.
mu = Vz+2.0*t
alpha = .25*t
eps = 2*t-mu

Vz = 100.
delta = 25.0
t = 10.0
alpha = 0.25*t

npw = 300
nsw = 1*1.5*npw
n = int(nsw+npw) # create zero matrix

np.savetxt('./data/2d-1d-config.txt', [npw, n], fmt='%i')

bdg = np.zeros((4*n,4*n))
# since v will change as a function of x we will make it into a matrix
v = np.zeros(n)
v[n//2-npw//2:n//2+npw//2] = Vz
v = np.diag(v)
mu = v+.01*t
eps = 2*t-mu

# since it is hermitian we only need to fill the lower triangle
# it is built by going down the 0th block diagonal then the rext diagonal after, repeat
# since we know the bdg has 16 section but only 8 to fill that are static we write them explicitly
bdg[0:n, 0:n] = eps*np.eye(n)+v-t*(np.eye(n,k=-1) + np.eye(n,k=-n) )
bdg[n:2*n, n:2*n] = eps*np.eye(n)-v-t*(np.eye(n,k=-1) + np.eye(n,k=-n) )
bdg[2*n:3*n, 2*n:3*n] = -eps*np.eye(n)-v+t*(np.eye(n,k=-1) + np.eye(n,k=-n) )
bdg[3*n:4*n, 3*n:4*n] = -eps*np.eye(n)+v+t*(np.eye(n,k=-1) + np.eye(n,k=-n) )
bdg[2*n:3*n, 0:n] = alpha * ( np.eye(n,k=-1) - np.eye(n,k=1) - np.eye(n,k=-n) + np.eye(n,k=n) )
bdg[2*n:3*n, n:2*n] = -delta*np.eye(n)
bdg[3*n:4*n, 1*n:2*n] = -alpha * ( np.eye(n,k=-1) - np.eye(n,k=1) - np.eye(n,k=-n) + np.eye(n,k=n) )
bdg[3*n:4*n, 0:n] = delta*np.eye(n)
print(bdg)
#print()

eng, states = np.linalg.eigh(bdg)
#states = states[0:n, :]
#idx = eng.argsort()[::-1]
#eng = eng[idx]
#states = states[:, idx]
#print(eng[4*(nsw+npw//2)-1:4*(nsw+npw//2)+1])
#print(eng[4*(nsw+8):4*(nsw+npw-8)])
np.savetxt('./data/2d-1d-eigenvalues.txt', eng)
prob = np.multiply(states,np.conj(states))
np.savetxt('./data/2d-1d-states.txt', prob)



if plotFlag:
  p = 20
  x = np.arange(-n//2, n//2, 1)
  for i in range(p):
    j = 2*n-p//2+i
    plt.figure(1000, figsize=(6,6))
    plt.title('%1.4e' % eng[j])
    tmp = prob[0:n,j]
    plt.plot([-npw//2,-npw//2],[0,np.max(tmp)],'gray')
    plt.plot([ npw//2, npw//2],[0,np.max(tmp)],'gray')
    plt.plot(x,tmp,':')
    plt.savefig('./data/figures/eigenstate-%i.pdf' % i)
    #plt.show(1000)
    plt.close(1000)
