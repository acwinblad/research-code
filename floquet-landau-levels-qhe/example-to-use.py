#model example to use for my code

import numpy 
from numpy import exp, zeros, pi, cos
import numpy as np


class PerturbedMatrix(object):
    def __init__(self, g, a, size):
        super(PerturbedMatrix, self).__init__()
        self.matrix = construct_matrix(g, a, size)
        self.alphas = [a for i in range(0, size)]

    @property
    def eigenvalues(self):
        return numpy.linalg.eigvals(self.matrix)


def construct_matrix(g, a, size):
    _matrix = zeros(shape=(size, size))
    matrix_norm = (2 * pi * a)
    minus_computed = exp(-g)
    plus_computed = exp(g)

    for i in range(0, size):
        _matrix[i, i] = 2*cos(matrix_norm*i)
        minus, plus = _construct_indeces(i, size)

        if minus is not None:
            _matrix[i, minus] = minus_computed

        if plus is not None:
            _matrix[i, plus] = plus_computed

    return _matrix

def _calculate_eigenvalues(matrix):
    return numpy.linalg.eigvals(matrix)

def matrix_eigenvalues(g, a, size):
    matrix = construct_matrix(g, a, size)

    return _calculate_eigenvalues(matrix)


def _construct_indeces(index, size):
    if index == 0:
        return size - 1, 1
    if index == size - 1:
        return index - 1, 0

    return index - 1, index + 1

import matplotlib.pyplot as plt

values = []

# Vertical indices
maximum = 100
# Horizontal indices 
sz = 100


def _make_alphas(a, size):
    return [a for i in range(0, size)]


for index in range(0, maximum + 1):
    alpha = float(index)/float(maximum)
    alpha_list = _make_alphas(alpha, sz)
    eigenvalues = matrix_eigenvalues(0, alpha, sz)

    values.append((eigenvalues, alpha_list))
print(np.shape(values))
print(values[0][0])
plt.figure()
for eigenvalues, alphas in values:
    plt.plot(eigenvalues, alphas, '.', color='r', markersize=1.5)

plt.axis([-4.5, 4.5, 0, 1])
#plt.show()
plt.close()


Emax = np.max(eigenvalues)
#Emin = np.min(eigenvalues)
Emin = -Emax
nE = 75
dE = (Emax-Emin)/(nE-1)
E = np.array([i*dE+Emin for i in range(nE)])
gE = np.zeros((maximum,nE))
print(np.shape(gE))
for i in range(maximum):
  for j in range(nE-1):
    idx = np.where(np.logical_and(values[i][0]>E[j],values[i][0]<E[j+1]))[0]
    gE[i,j+1] = np.size(idx)

print(gE)
plt.figure(figsize=(10,10))
#plt.colorbar()

# include cmap='some_color' into the imshow function
#'viridis', aspect=3/5, RdGy, alpha=0.5
plt.imshow(gE, origin='lower', interpolation="none", cmap= 'viridis', aspect=3/5)
#plt.savefig('../../data/fig-dos-spectral-flow.pdf')
#plt.axis(aspect='image')
plt.show()
plt.close()

