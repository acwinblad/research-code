#!/usr/bin/python
#
# Created by: Aidan Winblad
#

import numpy as np
from numpy import linalg as la
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

#load in eigen-energy matrix
energy = np.loadtxt('../data/bdg-spectral-flow-mu.txt')
nmu = np.size(energy[:,0])/2
nbins = 60
ggE = np.zeros((nbins,nmu))

EmaxR= np.where(energy[:,1:]==energy[:,1:].max())[0][0]
EmaxC= np.where(energy[:,1:]==energy[:,1:].max())[1][0]+1
EminR= np.where(energy[:,1:]==energy[:,1:].min())[0][0]
EminC= np.where(energy[:,1:]==energy[:,1:].min())[1][0]+1

for i in xrange(nmu):
  gE, E = np.histogram(energy[i+nmu/2,1:], bins=nbins)
  ggE[:,i] = gE

XB = E[1:]-np.abs(E[0]-E[1])/2.
#XB = E
YB = energy[nmu/2:3*nmu/2,0]
X, Y = np.meshgrid(XB,YB)
#plt.pcolor(X,Y,ggE.T)
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(X,Y,ggE.T, cmap=cm.viridis)
plt.show()

