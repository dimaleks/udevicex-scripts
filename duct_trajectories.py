#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:39:29 2018

@author: alexeedm
"""
import glob
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import proposal18_plot as myplt

def trajectories(case, ceny, cenz, size):
    files = sorted(glob.glob(case + "/pos/*.txt"))
    lines = list(itertools.chain.from_iterable([open(f).readlines() for f in files]))
        
    try:
        y = np.array([ float(x.split()[3]) for x in lines ])[0:]
        z = np.array([ float(x.split()[4]) for x in lines ])[0:]
    except:
        print("Error reading")
        return [], []
    
    y = (y - ceny) / size
    z = (z - cenz) / size
    
    return y[0:4000], z[0:4000]

def plottraj(traj, signs, alpha):
    
    for t in traj:
        y = signs[0]*t[0]
        z = signs[1]*t[1]

        if len(z) < 1:
            continue
    
        plt.plot(y, z, lw=0.05, zorder=1, color="C7", alpha=0.5*alpha)
        plt.plot(y[0], z[0],   "x", ms=1, color="C7", zorder=1, alpha=alpha, markeredgewidth=0.5)
        plt.plot(y[-1], z[-1], "o", ms=2.5, color="lime", markeredgewidth=0.4, markeredgecolor='black', zorder=10, alpha=alpha)

#%%
# Duct
folder = "/home/alexeedm/extern/daint/project/alexeedm/focusing_square_free/"
name = "case_8_0.04__80_40_1.5__"
reference = mpimg.imread("/home/alexeedm/Pictures/choi_fig2f.png")

variants = sorted(glob.glob(folder + name + "*/"))

traj = []
for case in variants:
    print(case)
    traj.append(trajectories(case, 52, 52, 50.0))


#%%

fig = plt.figure()
plt.imshow(reference, extent=(-1,1, -1,1), zorder=0)

plt.axes().set_xlim([-1.1, 1.1])
plt.axes().set_ylim([-1.1, 1.1])

# Mirror
#for t in traj:
#    t = (-t[0], t[1])

plottraj(traj, (-1, 1),  0.25)
plottraj(traj, (1, -1),  0.25)
plottraj(traj, (-1, -1), 0.25)
plottraj(traj, (1, 1), 1.0)

ax=plt.gca()

plt.xticks([])
plt.yticks([])

myplt.set_font_sizes(ax)
myplt.set_figure_size(fig, 3)
myplt.save_figure(fig, 2.5, 'square_duct_trajectories.pdf', dpi=400, pad_inches = 0)
plt.close()


