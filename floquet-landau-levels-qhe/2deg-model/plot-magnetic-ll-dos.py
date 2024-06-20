#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import glob
from scipy.stats import gaussian_kde

# load configuration values
mc = 0
rc = 5
filepath = './data/'
#config = np.loadtxt(filepath+'config.txt')
config = np.loadtxt('./data/magnetic-config.txt')
rc = int(config[0])
mc = int(config[1])
t = float(config[2])
Efmin = float(config[3])
Efmax = float(config[4])
nEf = int(config[5])
#Ef = np.array([(i/nEf)**(1/2) for i in range(nEf)])*phimax
strnEf = str(nEf)

nr = 2*rc+1
nm = 2*mc+1
m0 = (mc-0)*nr
mf = (mc+1)*nr

# load calculated values and states
energy = np.loadtxt(filepath+'magnetic-eng-matrix.txt')

# calculate a weighted/projected density of states as a function of E
Emax = +4.0*t
Emin = -4.0*t
nE = 400
dE = (Emax - Emin)/(nE-1)
#E = np.array([i*dE+Emin for i in range(nE)])
E = np.arange(0,nE)*dE+Emin

# place weighted eigenvalues in an energy box-bin (also a normal dos)
gauswE = np.zeros((nE,nEf))
sigma = dE
sig2 = sigma**2
norm = (sigma*np.sqrt(np.pi))**(-1)

for i in range(nEf):
  #gauswE[:, i] = norm*np.sum(weight[i,:]*np.exp( -(E - energy[:,i])**2 / sig2))
  for j in range(nE):
    gauswE[j, i] = norm*np.sum(np.exp( -(E[j] - energy[:,i])**2 / sig2))

# setup plot for weighted density of states
fig, ax = plt.subplots(1,1)
x = np.linspace(Efmin,Efmax,nEf//2)
Ef = np.linspace(Efmin,Efmax,nEf)
Eng = np.linspace(Emin,Emax,nE)
y = 0.9*x**1.5 +0.191*2
X, Y = np.meshgrid(Ef,Eng)

# set x-axis
xticks = np.linspace(Efmin,Efmax,5, endpoint=True)
xlabelarray = np.linspace( Efmin, Efmax, 5, endpoint=True)
ax.set_xticks(xticks)
ax.set_xticklabels(['%1.1e' % val for val in xlabelarray])
ax.set_xlabel('$E$')

# set y-axis
yticks = np.linspace(Emin,Emax,5, endpoint=True)
ylabelarray = np.linspace(Emin, Emax, 5, endpoint=True)
ax.set_yticks(yticks)
ax.set_yticklabels(['%1.2e' % val for val in ylabelarray])
ax.set_ylabel('$Energy\ (eV)$')

# plot and save figures
# bottom band projected gaussian dos
img = ax.imshow( gauswE**(1.0), origin='lower', interpolation='spline16', cmap='inferno_r', extent=[X.min(), X.max(), Y.min(), Y.max()], aspect='auto')
plt.savefig('./figures/dos-magnetic-projection-gaussian.pdf', bbox_inches='tight')

