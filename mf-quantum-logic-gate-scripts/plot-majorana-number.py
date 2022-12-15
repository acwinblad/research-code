#!/usr/bin/python3
#
#
# Plots a colormap of the Majorana number
# Created by: Aidan Winblad
# 09/27/2022
#

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

filein = './data/top-inv-majorana-number.txt'
filein = './data/majorana-number-hollow-triangle.txt'
majNum = np.loadtxt(filein)
majNum = np.vstack((np.flipud(majNum[1:,:]),majNum))
if(majNum[0,0] < 0):
  majNum *= -1

vmax = np.max(majNum)
vmin = np.min(majNum)
if(np.abs(vmax) > np.abs(vmin)):
    vmin = -vmax
else:
    vmax = -vmin

nx = np.size(majNum[0,:])
ny = np.size(majNum[:,0])
x = np.linspace(0,2.3*np.pi,nx)
y = np.linspace(-2,2,ny)
X, Y = np.meshgrid(x,y)
fig, ax = plt.subplots(1,1)
plt.xlabel('$B (\pi)$')
plt.ylabel('$\mu (t)$')
pcm = ax.pcolormesh(X, Y, majNum, norm=colors.SymLogNorm(linthresh=0.010, linscale=0.5,vmin=vmin,vmax=vmax),cmap='RdBu', linewidth=0, rasterized='True')
pcm.set_edgecolor('face')
fig.colorbar(pcm, ax=ax, extend='both')
ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
#ax.plot([1.5,1.5,1.5],[0.7,1.1,1.4], color='black', lw = 3, marker='s')
#ax.plot([1.6,1.6,1.6],[0.7,1.1,1.4], color='black', lw = 3, marker='s')
plt.tight_layout()
fig.savefig('./data/figures/triangular-chain-majorana-number.pdf', bbox_inches='tight')
#plt.show()
plt.close()
