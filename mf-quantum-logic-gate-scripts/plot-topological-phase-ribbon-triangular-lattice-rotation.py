#!/usr/bin/python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

a = 1
w = 3
mu = -2.50
absmu = abs(mu)
mustring = f"{absmu:1.4f}"
mustring = mustring.replace('.','_')
if(mu >= 0):
  mustring = 'p' + mustring
else:
  mustring = 'n' + mustring

A0 = 0*np.pi
Af = 2*np.pi / (np.sqrt(3) *a)
Af = 1.0*np.pi

tp = np.loadtxt('./data/majorana-number-w-%01i-mu-%s.txt' % (w, mustring))
nt = np.size(tp[0,:])

ntr = 1*nt//3
ntl = 2*nt//3

tpr = np.hstack([tp[:,ntr:], tp[:,0:ntr]])
tpl = np.hstack([tp[:,ntl:], tp[:,0:ntl]])
tpb = tp
tpr = (tpr-0*0.8)**3
tpl = (tpl+0*1.1)**3
tpov = 1*tpb + 1*tpr + 1*tpl

bpb = -(tpb-1)//2
bpr = -(tpr-1)//2
bpl = -(tpl-1)//2
bpov = 1*bpb + 1*bpr*2 + 1*bpl*4

map2to3 = np.where(bpov==2)
map3to2 = np.where(bpov==3)
map6to4 = np.where(bpov==6)
map4to5 = np.where(bpov==4)
map5to6 = np.where(bpov==5)
bpov[map2to3] = 3
bpov[map3to2] = 2
bpov[map6to4] = 4
bpov[map4to5] = 5
bpov[map5to6] = 6

plt.rcParams.update({'font.size':13})

xmin = 0
xmax = np.pi
ymin = A0
ymax = Af
vmin = np.min(tpov)*1.0
vmax = np.max(tpov)*1.0

plt.figure()
fig, ax = plt.subplots(1,1)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)

plt.ylabel("$A$", fontsize=16)
plt.yticks(np.linspace(ymin, np.round(2*ymax)/2, 4*np.int32(ymax-ymin)+1))
#plt.yticks(np.linspace(ymin, ymax, 11))
plt.xlabel(r"$\varphi $", fontsize=16)
nx = 6
ax.set_xticks(np.linspace(xmin,xmax,nx+1))
#plt.locator_params(axis='x', nbins=12)
plt.gca().yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:1.2f}"))
plt.gca().xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:1.2f}"))

#cax = ax.imshow(bpov, cmap='Accent', extent=[xmin,xmax,ymin,ymax], vmin=0, vmax=7)
cmap = plt.get_cmap('inferno',8)
ticks = np.arange(1,17,2)*7/16
cax = ax.imshow(bpov, cmap=cmap, extent=[xmin,xmax,ymin,ymax], vmin=0, vmax=7)
cbar = fig.colorbar(cax, ticks=ticks, pad=0.01)
cbar.ax.set_yticklabels(['[1,1,1]','[-1,1,1]','[-1,-1,1]','[1,-1,1]','[1,-1,-1]','[1,1,-1]','[-1,1,-1]','[-1,-1,-1]'])
cbar.set_label('$\mathcal{M}_{b,r,l}$', rotation=270, fontsize=18, labelpad=15)
ax.set_aspect('auto')
fig.tight_layout()
plt.savefig('./data/figures/topological-phase-diagram-w-%01i-mu-%s.pdf' % (w, mustring) )
plt.close()
plt.clf()
plt.cla()



