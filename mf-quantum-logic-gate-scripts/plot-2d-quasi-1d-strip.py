#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

config = np.loadtxt('./data/2d-1d-config.txt')
npw = int(config[0])
n = int(config[1])

zeeman = np.loadtxt('./data/2d-1d-zeeman-dist.txt')
energy = np.loadtxt('./data/2d-1d-eigenvalues.txt')
states = np.loadtxt('./data/2d-1d-states.txt')
idx = np.where(zeeman[0:n//2]==0)[0]

p = 10
x = np.arange(-n//2, n//2, 1)
for i in range(p):
  j = 2*n-p//2+i
  plt.figure(1000, figsize=(6,6))
  plt.title('$\mu=v$,  $E=$%1.2e' % energy[j])
  plt.ylabel('$|\Psi|^2_r$', fontsize=12)
  plt.xlabel('$x$', fontsize=12)
#  plt.xlim(-n/2+3*idx[-1]/2,n/2-3*idx[-1]/2)
  plt.ylim(-0.01,1.1)
#  plt.yscale('log')
  tmp = states[0:n,j] / np.max(states[0:n,j])
  plt.plot([-npw//2,-npw//2],[0,np.max(tmp)],'gray')
  plt.plot([ npw//2, npw//2],[0,np.max(tmp)],'gray')
  plt.plot(x,zeeman/np.max(zeeman))
  #plt.plot(x,tmp,':')
  plt.plot(x,tmp)
  plt.savefig('./data/figures/eigenstate-%i-v-mu.pdf' % i)
  #plt.show(1000)
  plt.close(1000)

