#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

config = np.loadtxt('./data/2d-1d-config.txt')
npw = int(config[0])
n = int(config[1])

energy = np.loadtxt('./data/2d-1d-eigenvalues.txt')
states = np.loadtxt('./data/2d-1d-states.txt')

p = 20
x = np.arange(-n//2, n//2, 1)
for i in range(p):
  j = 2*n-p//2+i
  plt.figure(1000, figsize=(6,6))
  plt.title('%1.4e' % energy[j])
  plt.yscale('log')
  tmp = states[0:n,j]
  #plt.plot([-npw//2,-npw//2],[0,np.max(tmp)],'gray')
  plt.plot([ npw//2, npw//2],[0,np.max(tmp)],'gray')
  #plt.plot(x,tmp,':')
  plt.plot(x,tmp)
  plt.savefig('./data/figures/eigenstate-%i.pdf' % i)
  #plt.show(1000)
  plt.close(1000)

