#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

energy = np.loadtxt('./data/eng-matrix.txt')
rows, columns = np.shape(energy)

plt.figure()
plt.ylim(-1,1)
for i in range(rows):
  plt.plot(energy[i,:])

plt.show()
plt.close()
