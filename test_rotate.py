#!/usr/bin/env -S singularity exec /home/astrojhgu/singularity/radio.simg python3

import numpy as np
import healpy
from healpy.rotator import Rotator, angdist
from healpy import pix2ang
nside=256
npix=healpy.nside2npix(nside)
#data=np.sin(np.arange(npix)/1024*2*np.pi)
pt0=[np.radians(45), np.radians(0)]
angles=pix2ang(nside, range(npix))
t0=np.exp(-angdist(pt0, angles)**2)

pq0=[np.radians(90), np.radians(0)]
angles=pix2ang(nside, range(npix))
q0=np.exp(-angdist(pq0, angles)**2)

pu0=[np.radians(135), np.radians(0)]
angles=pix2ang(nside, range(npix))
u0=np.exp(-angdist(pu0, angles)**2)

healpy.write_map("T0.fits", t0, overwrite=True)
healpy.write_map("Q0.fits", q0, overwrite=True)
healpy.write_map("U0.fits", u0, overwrite=True)

rot=Rotator((45,45,10), deg=True, eulertype='X')

T1,Q1,U1=rot.rotate_map_pixel((t0,q0,u0))
healpy.write_map("T1.fits", T1, overwrite=True)
healpy.write_map("Q1.fits", Q1, overwrite=True)
healpy.write_map("U1.fits", U1, overwrite=True)
