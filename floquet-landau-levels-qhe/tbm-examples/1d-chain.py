#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt

h = 1
mu = 0
a = 1
n = 10
L = n*a

k = np.arange(1,n+1)*2*np.pi/(L)
print(k)

H = mu * np.diag(np.ones(n),k=0) - h * np.diag(np.ones(n-1),k=-1)

evals, evecs = np.linalg.eigh(H, UPLO = 'L')
print(evals)
print(evals[0],evals[-1])

plt.figure()
plt.plot(k,evals, 'ko')
plt.show()
plt.close()
