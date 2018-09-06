#!#/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import axes3d


a = 1
nr = 5
n = nr*(nr+1)/2

x = np.zeros(n)
y = np.zeros(n)
latticeCtr = 0
for i in xrange(nr):
  for j in xrange(i+1):
    x[latticeCtr] = a*(j-i/2.)
    y[latticeCtr] = i*a*np.sqrt(3)/2.
    latticeCtr += 1

triang = mtri.Triangulation(x,y)
z = x**2+y**2
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')

#plt.tricontourf(triang,z)
ax.plot_trisurf(triang,z)
#plt.triplot(triang, 'ko-')
plt.show()
