#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 19:45:00 2018

@author: alexeedm
"""

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
    
    return y, z

def plottraj(case, alpha=1.0):
    z, y = trajectories(case, 62, 62, 60.0 )
    if len(z) < 1:
        return
    
    plt.plot(y, z, lw=0.05, zorder=1, color="C7", alpha=0.5*alpha)
    plt.plot(y[0], z[0],   "x", ms=1, color="C7", zorder=1, alpha=alpha, markeredgewidth=0.5)
    plt.plot(y[-1], z[-1], "o", ms=2.5, color="lime", markeredgewidth=0.4, markeredgecolor='black', zorder=10, alpha=alpha)


fig = plt.figure()

# Pipe
folder = "/home/alexeedm/extern/daint/project/alexeedm/focusing_free/"
name = "case_5_0.05__80_40_1.5__"
#name = "case_5_0.02828__80_30_1.5__"


reference = mpimg.imread("/home/alexeedm/Pictures/choi_lee_fig4i.png")
plt.imshow(reference, extent=(-1.02,1.02, -1.02,1.02), zorder=0)

variants = sorted(glob.glob(folder + name + "*/"))

plt.axes().set_xlim([-1.1, 1.1])
plt.axes().set_ylim([-1.1, 1.1])



for case in variants:
    print(case)
    plottraj(case)
    
    
ax=plt.gca()

plt.xticks([])
plt.yticks([])

myplt.set_font_sizes(ax)
myplt.save_figure(fig, 2.5, 'circular_pipe_trajectories.pdf', dpi=400, pad_inches = 0)
plt.close()


