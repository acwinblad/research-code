#!/usr/bin/python

import numpy as np

nf = 100
vz = np.full(nf,1000.)
vy = np.linspace(0,vz[0]/100.,nf)
vx = np.full(nf,0.)
delta = np.full(nf,25.)
t = np.full(nf,10.)
mufctr = np.full(nf,3.)
alphafctr = np.full(nf,0.5)
runfile = np.linspace(1,nf,nf)
tagfile = np.char.mod('%d.py', runfile)
nrl = np.full(nf,40)
nrm = np.full(nf,500)

np.savetxt('./vz.rep',               vz, fmt='%.4f')
np.savetxt('./vy.rep',               vy, fmt='%.4f')
np.savetxt('./vx.rep',               vx, fmt='%.4f')
np.savetxt('./delta.rep',         delta, fmt='%.4f')
np.savetxt('./t.rep',                 t, fmt='%.4f')
np.savetxt('./mufctr.rep',       mufctr, fmt='%.4f')
np.savetxt('./alphafctr.rep', alphafctr, fmt='%.4f')
np.savetxt('./runfile.rep',     runfile, fmt='%04i')
np.savetxt('./nrl.rep',             nrl, fmt='%i')
np.savetxt('./nrm.rep',             nrm, fmt='%i')
np.savetxt('./tagfile.txt', tagfile, fmt='%s')

