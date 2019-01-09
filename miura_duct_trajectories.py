#!/usr/bin/env python
import glob
import itertools
import numpy as np
import matplotlib as mpl
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
    
        plt.plot(y[::10], z[::10], lw=0.05, zorder=1, color="C7", alpha=0.5*alpha)
        plt.plot(y[0], z[0],   "x", ms=1, color="C7", markeredgewidth=0.5, zorder=1, alpha=alpha)
        plt.plot(y[-1], z[-1], "o", ms=2.5, color="limegreen", markeredgewidth=0.4, markeredgecolor='black', zorder=10, alpha=alpha)

#%%
# Duct
folder = "/home/alexeedm/extern/daint/project/alexeedm/focusing_square_free/"
#name = "case_8_0.00769__80_20_1.5__"
name = "scattered_5_0.0315__80_20_1.5__"
#name = "case_5_0.03149__80_20_1.5__"
reference = mpimg.imread("/home/alexeedm/Pictures/miura_fig5a.png")

variants = sorted(glob.glob(folder + name + "*/"))

traj = []
for case in variants[0:-1]:
    print(case)
    traj.append(trajectories(case, 48.295,48.295, 46.295))


#%%

myplt.set_qt_backend()
fig = plt.figure()
plt.imshow(reference, extent=(-1,1, -1,1), zorder=0)

plt.axes().set_xlim([-1.1, 1.1])
plt.axes().set_ylim([-1.1, 1.1])

plottraj(traj, (-1, 1),  0.25)
plottraj(traj, (1, -1),  0.25)
plottraj(traj, (-1, -1), 0.25)
plottraj(traj, (1, 1), 1.0)

ax=plt.gca()

plt.xticks([])
plt.yticks([])

myplt.set_font_sizes(ax)
myplt.set_figure_size(fig, 3)
myplt.save_figure(fig, 2.5, 'square_duct_trajectories_144.pdf', dpi=400, pad_inches = 0)
plt.close()
#plt.show()

