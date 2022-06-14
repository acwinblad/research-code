#!/usr/bin/python3

import numpy as np

t = 1
delta = t
mu = 0
n = 9

h1 = -mu*np.eye(n) - t*(np.diag(np.ones(n-1),k=-1)+np.diag(np.ones(n-1),k=1))
h2 = delta*(np.diag(np.ones(n-1),-1)-np.diag(np.ones(n-1),1))
ht = np.hstack((h1,h2))
hb = np.hstack((h2,h1))
h = np.vstack((ht,-hb))
print(h)

eng, vec = np.linalg.eigh(h)

print(eng[n-2:n+2])
print(vec[:,n].conj()*vec[:,n])
