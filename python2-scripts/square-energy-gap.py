#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

# property values 
delta = 0.1+0j
t = 10*np.abs(delta)
mu = 6*t
a = 1.

# dimensions for the graph
d = np.pi/a
nr = 50
ds = d/nr
x = np.array([-d+2*i*ds for i in xrange(nr)])
y = np.array([-d+2*i*ds for i in xrange(nr)])
x,y = np.meshgrid(x,y)


# calculate the Energy_+/- 
eps = -2*t*(np.cos(y)+np.cos(x)) -(mu-4*t)
delP =2*delta*1j*(np.sin(x)+1j*np.sin(y))
Ep = np.sqrt(eps**2+np.real(np.conj(delP)*delP))
Em = -np.sqrt(eps**2+np.real(np.conj(delP)*delP))

# Plot and save both energy spectrums to one figure
ax.view_init(elev=10., azim=60)
ax.plot_surface(x,y, Ep, cmap='viridis')
ax.plot_surface(x,y, Em, cmap='viridis')

plt.show()
