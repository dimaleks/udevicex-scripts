#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 01:04:00 2018

@author: alexeedm
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle

import thesis_plot as myplt

fint = np.array( pickle.load(open('data/focusing/energies_soft_square.pckl', 'rb')) )

myplt.set_pgf_backend()
fig = plt.figure()

for lbd, c in [(1.0, 'C0'), (5.0, 'C1'), (25.0, 'C2')]:
    for Ca, fmt in [(1.0, 'o'), (0.1, 's'), (0.01, 'D')]:
       
        idx = np.where( (fint[:,1] == lbd) & (fint[:,2] == Ca) )
        data = fint[idx[0], :]
        
        err = 0.5 * ( np.abs(data[:, 4] - data[:, 3]) + np.abs(data[:, 5] - data[:, 3]) )
        
        #plt.errorbar(data[:,0], data[:,3], yerr=err, fmt=fmt, color=c, ms=3, linewidth=1, label=r'$Ca = ' + str(Ca) + r', \lambda = ' + str(lbd) + r'$')
        plt.plot(data[:,0], -data[:,3], fmt, color=c, ms=4, linewidth=1, label=r'$ \begin{aligned}Ca &= ' + str(Ca) + r',\\ \lambda &= ' + str(lbd) + r'\end{aligned}')

#        print(data)
#        print("")
    
plt.legend(ncol=3, handletextpad=0.01)

ax = plt.gca()
ax.set_ylabel(r'$W = \int_{\bm S} C^{||}_l \, ds$')
ax.set_xlabel(r'$Re$')
myplt.set_font_sizes(ax)
myplt.set_figure_size(fig, 4.5)

y = plt.ylim()
plt.ylim(-0.018, y[1])

#plt.show()
myplt.save_figure(fig, 4.5, 'tube_soft_lift_work.pdf')