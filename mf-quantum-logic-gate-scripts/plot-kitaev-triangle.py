#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

energy = np.loadtxt('./data/kitaev-triangle-energy.txt')
states = np.loadtxt('./data/kitaev-triangle-states.txt')

x,y = np.loadtxt('./data/kitaev-triangle-coord.txt', unpack=True)
triang = tri.Triangulation(x,y)


plt.figure()
plt.tricontourf(triang, states[:,-1])
plt.show()
plt.close()
