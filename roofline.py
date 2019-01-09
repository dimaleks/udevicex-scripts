#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:54:11 2018

@author: alexeedm
"""
import numpy as np
import matplotlib.pyplot as plt
import thesis_plot as myplt

def roofline(band, flops):
    
    lo, hi = 1e-2, 1e+3
    
    ridge = flops / band
    
    ois = np.array([lo, ridge, hi])
    
    return ois, np.minimum(ois * band, flops)


def prepare_plot(ax):
    
    myplt.make_grid(ax, True)
    myplt.set_font_sizes(ax)
    ax.set_xscale("log", nonposx='clip')
    ax.set_yscale("log", nonposy='clip')
    
    ax.set_xlabel("Operational intensity")
    ax.set_ylabel("GFLOPs / s")
    
    
def draw_one(x, y, style, color, width, label):
    plt.plot(x, y, linestyle=style, color=color, label=label, linewidth=width, zorder=2)
    

cpu = roofline(47.545, 540.990)
gpu = roofline(554.833, 6318.301)

cpuref = roofline(68, 998)
gpuref = roofline(732, 8524)

myplt.set_pgf_backend()
fig = plt.figure()

draw_one(gpu[0], gpu[1], '-', 'C2', 1.5, 'Nvidia Tesla P100 16GB')
draw_one(cpu[0], cpu[1], '-', 'royalblue', 1.5, 'Intel Xeon E5-2690 v3')

draw_one(gpuref[0], gpuref[1], ':', 'C2', 0.75, '')
draw_one(cpuref[0], cpuref[1], ':', 'royalblue', 0.75, '')

prepare_plot(plt.gca())

plt.legend()
myplt.save_figure(fig, 3, 'new_roofline.pdf')
plt.close(fig)




xeon = roofline(402.503e-3  * 72,  2467.047e-3  * 103)
amd  = roofline(306.105e-3  * 72,  1368.974e-3  * 103)
gpu  = roofline(1846.646e-3 * 72,  25507.547e-3 * 90)

xeonref = roofline(51.2, 332.8)
amdref  = roofline(51.2, 332.8) # intentionally same
gpuref  = roofline(250, 3935)

myplt.set_pgf_backend()
fig = plt.figure()

draw_one(gpu[0],  gpu[1],  '-', 'C2', 1.5, 'Nvidia Tesla K20X')
draw_one(xeon[0], xeon[1], '-', 'royalblue', 1.5, 'Intel Xeon E5-2670')
draw_one(amd[0],  amd[1],  '-', 'peru', 1.5, 'AMD Opteron 6274')

draw_one(gpuref[0],  gpuref[1],  ':', 'C2', 0.75, '')
draw_one(xeonref[0], xeonref[1], ':', 'royalblue', 0.75, '')
draw_one(amdref[0],  amdref[1],  ':', 'peru', 0.75, '')

prepare_plot(plt.gca())

plt.legend()
myplt.save_figure(fig, 3, 'old_roofline.pdf')
plt.close(fig)
