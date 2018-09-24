#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:38:42 2018

@author: alexeedm
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import os
import pickle

rho = 8
r = 5
kappa = 0.22
H = 2*r / kappa
u = 3.5

def coefficient(frc):
    return frc / (rho * u**2 * (2*r)**4 / H**2)

def extract_vars(path):
    
    lastdir = os.path.basename(os.path.normpath(path))

    f, a, gamma, ry, rz, wy, wz = [ float( re.search('_' + nm + r'_([^_]+)', lastdir).group(1) )
                                for nm in ['f', 'a', 'gamma', 'ry', 'rz', 'wy', 'wz'] ]
            
    return (f, a, gamma, ry, rz, wy, wz)

def read_one(fnames):
    raw = np.vstack( tuple([np.loadtxt(f) for f in fnames]) )
    
    res = np.array([])
    
    time = raw[:,1]
    for i in range(2, raw.shape[1]):
        col = raw[:,i]
        idx = np.where(time > 1000.0)
        res = np.append( res, np.mean(col[idx]) )
        res = np.append( res, np.var(col[idx]) / idx[0].size )
    
    return res
    
def read_data(prefix, fname):
    cases = sorted(glob.glob(prefix))
    
    alldata = None
    for case in cases:        
        print(case)
        raw = read_one( sorted(glob.glob(case + '/' + fname + '/*.txt')) )
        raw[0:6] = coefficient(raw[0:6])
                
        f, a, gamma, ry, rz, wy, wz = extract_vars(case)
        raw = np.hstack( (np.array([ry, rz]), raw, np.array([wy, wz])) )
        
        if alldata is None:
            alldata = raw
        else:
            alldata = np.vstack( (alldata, raw) )
            
    return alldata
    
def read_cache(func, folder, fname, override_cache=False):
    try:
        res = pickle.load(open(folder + fname + '.pckl', 'rb'))
        success = True
    except:
        success = False
    
    if not success or override_cache:
        res = func()
        pickle.dump(res, open(folder + fname + '.pckl', 'wb'))
    else:
        print('Successfully got cached values')
        
    return res

scratch1600 = '/home/alexeedm/extern/daint/scratch1600/focusing_square_rigid_massive/newcode/'
Re = 100
Res = str(int(Re))
data = read_cache( lambda : read_data(scratch1600 + 'case___Re_' + Res + '*/', 'pinning_force'), 'data/focusing/', 'modrotation'+Res, True)

fig = plt.figure()
plt.quiver(data[:,0], data[:,1], data[:,4], data[:,6])
plt.axes().set_aspect('equal', 'datalim')
plt.show()

#
#coo = data[:,[0,1]]
#tau = coo[1:] - coo[:-1]
#
#f = data[:,[4,6]]
#
#mag = np.sum(np.multiply(f[:-1], tau), axis=1)
#length = np.sum(np.multiply(tau, tau), axis=1)
#t = np.add.accumulate(length)
#
#
#plt.plot(t, mag, 'o')
##plt.plot(data[:, 1], w, 'x')
#plt.show()







