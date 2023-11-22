#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

vz = 1000.0
t = 10.0
tr = 0.5*t
ma = 0.5/t
ma = (ma*vz)/(vz+ma*tr**2)
mu = vz+6*t
dels = 25.0
kf = np.sqrt(2*ma*mu)
delp = (dels*tr*kf)/(2*vz*mu)
a = 2e1

nj = 751
nk = 501
j = np.arange(0,nj,1)
k = np.arange(0,nk,1)
J, K = np.meshgrid(j,k)
lsq = J**2/9 + K**2/3


lf = 2*np.pi/(a*kf)
En = np.matrix(np.real(mu*np.sqrt(1+lf**4*lsq**2 - lf**2*(2+(lf*delp/kf)**2)*lsq+0.j)))

plt.imshow(En, aspect='auto')
plt.colorbar()
plt.show()
plt.close()
