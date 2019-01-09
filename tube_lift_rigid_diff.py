#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:57:57 2018

@author: alexeedm
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import thesis_plot as myplt
import scipy.optimize as scopt

# (x, f, err, gp, Re, kappa, rot);  [x0, err, Re, kappa, rot]
processed, intersections = pickle.load(open('data/tube_rigid_lifts.pckl', 'rb'))

alldiff = {}
for Re in [50, 100, 200]:
    for kappa in [0.15, 0.22, 0.3]:
        free  = [ elem for elem in processed if elem[4] == Re and elem[5] == kappa and elem[6] == 0 ]
        norot = [ elem for elem in processed if elem[4] == Re and elem[5] == kappa and elem[6] == 1 ]
        
        diff = free[0][1] - norot[0][1]
        err  = np.abs(free[0][2]) + np.abs(norot[0][2])
        
        alldiff[(Re, kappa)] =  np.vstack((free[0][0], diff, err, free[0][7][:diff.size]))


myplt.set_pgf_backend()

def transform(x,y,Re,kappa, x0):
    a,b,c = x0
    return x**(-1) * y * Re**(1/3.0) * (1 + b/Re/((1-kappa)-x))

def transform0(x, y, Re, kappa):
    return transform(x,y, Re, kappa, 0.5, 10)

def func(x0):
        
    mean = 0
    count = 0
    for Re, kappa in alldiff.keys():
        data = alldiff[(Re, kappa)]
        idx = np.where((data[0,:] < 0.65) & (data[0,:] > 0.01))
        data = data[:, idx[0]]
    
        collapse = transform(data[0,:], data[1,:], Re, kappa, x0)
        mean += np.mean(collapse)
        count += 1
        
    mean /= count
    
    total = 0
    for Re, kappa in alldiff.keys():
        collapse = transform(data[0,:], data[1,:], Re, kappa, x0)
        total += np.sum( np.abs( (collapse - mean) / mean )**2 )
            
    return np.sqrt(total) 

res = scopt.minimize(func, [0.15, 1, -1], tol=1e-15, method='Powell', options={'maxiter' : 1})
print(res)

fig = plt.figure()

for Re, kappa in alldiff.keys():
    data = alldiff[(Re, kappa)]
    if Re == 50:
        c = 'C0'
    if Re == 100:
        c = 'C1'
    if Re == 200:
        c = 'C2'
        
    if kappa == 0.15:
        fmt = 'o'
    if kappa == 0.22:
        fmt = 's'
    if kappa == 0.3:
        fmt = 'D'
    #plt.plot(data[0,:], data[2,:], 'o', label=r'$Re = ' + str(Re) + r', \kappa = ' + str(kappa) + r'$')
    plt.errorbar(data[0,:], transform(data[0,:], data[1,:], Re, kappa, res.x), yerr=transform(data[0,:], data[2,:], Re, kappa, res.x),
                 fmt=fmt, color=c, ms=3, linewidth=1, label=r'$Re = ' + str(Re) + r', \kappa = ' + str(kappa) + r'$')
    
plt.legend()

ax = plt.gca()
ax.set_ylabel(r'$ \dfrac{C_{l}}{y} \times Re^{1/3} \left( 1 + \dfrac{7.894}{Re \left( 1 - \kappa - y \right)} \right)$')
ax.set_xlabel(r'$y$')
myplt.set_font_sizes(ax)
myplt.save_figure(fig, 4.1, 'tube_lift_collapse.pdf')













