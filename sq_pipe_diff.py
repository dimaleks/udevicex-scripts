#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 10:02:40 2018

@author: alexeedm
"""

import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import scipy.optimize as scopt
import scipy.integrate as scintegr
import scipy.interpolate as scinterp


kappa= 0.22
Re = 100

fname = 'Re_' + str(Re) + '_kappa_' + str(kappa) + '__rotation_'
pickle_folder = 'data/focusing/'
figure_folder = 'media/focusing/'


def coo_force_rot_slip__from_data():
    
    f_rotation = pickle.load(open(pickle_folder + 'forcedata_' + fname + '0.pckl','rb'))
    f_norot    = pickle.load(open(pickle_folder + 'forcedata_' + fname + '1.pckl','rb'))
    rotation   = pickle.load(open(pickle_folder + 'rotationdata_' + fname + '0.pckl','rb'))
    slip_vel   = pickle.load(open(pickle_folder + 'velocitydata_' + fname + '0.pckl','rb'))
    
    ids_rot   =  f_rotation[:,0]*10 + f_rotation[:,1]
    ids_norot =  f_norot[:,0]*10 + f_norot[:,1]
    
    tol = 1e-10
    norot_ind = (np.abs(ids_rot[:,None] - ids_norot) < tol).any(0)
    rot_ind = (np.abs(ids_norot[:,None] - ids_rot) < tol).any(0)
    
    y = f_rotation[rot_ind, 0]
    z = f_rotation[rot_ind, 1]
    
    fy = f_rotation[rot_ind, 2] - f_norot[norot_ind, 2]
    fz = f_rotation[rot_ind, 3] - f_norot[norot_ind, 3]
    
    wy = rotation[rot_ind, 2]
    wz = rotation[rot_ind, 3]
    
    slip = slip_vel[rot_ind, 2]
    
    return y, z, fy, fz, wy, wz, slip

def coo_force_rot_slip__from_gaussian():
    
    f_rotation = pickle.load(open(pickle_folder + 'force_' + fname + '0.pckl','rb'))
    f_norot    = pickle.load(open(pickle_folder + 'force_' + fname + '1.pckl','rb'))
    rotation   = pickle.load(open(pickle_folder + 'omega_' + fname + '0.pckl','rb'))
    slip_vel   = pickle.load(open(pickle_folder + 'slip_' + fname + '0.pckl','rb'))
    
    x = np.linspace(0, 0.7, 20)
    y = np.linspace(0, 0.7, 20)
    X,Y = np.meshgrid(x,y)
    grid = np.array([X.flatten(),Y.flatten()]).T
    
    fy = f_rotation[0].predict(grid) - f_norot[0].predict(grid)
    fz = f_rotation[1].predict(grid) - f_norot[1].predict(grid)
    
    wy = rotation[0].predict(grid)
    wz = rotation[1].predict(grid)
    
    slip = slip_vel.predict(grid)
    
    return grid[:,0], grid[:,1], fy, fz, wy, wz, slip


def force_vs_rot(y, z, fy, fz, wy, wz, save=False, fname=''):
    
    fig = plt.figure()
           
    plt.quiver(y, z, fy, fz)
    plt.quiver(y, z, wy, wz, color='C3')
    
    if save:
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def angle(y, z, fy, fz, wy, wz, save=False, fname=''):
    
    fig = plt.figure()

    ns = np.ceil(np.sqrt(y.size)).astype(int)
    
    Y = y.reshape(ns, ns)
    Z = z.reshape(ns, ns)
    ang = np.abs( (fy*wy + fz*wz) / (fy*fy + fz*fz)**0.5 / (wy*wy + wz*wz)**0.5)
    
    plt.pcolormesh(Y, Z, ang.reshape(Y.shape), shading='gouraud', cmap=plt.cm.plasma, vmax=0.4)
    plt.colorbar()

    if save:
        fig.savefig(fname, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def lift_coeff(y, z, fy, fz, wy, wz, slip, save=False, fname=''):
    
    magf = np.sqrt(fy**2 + fz**2)
    magw = np.sqrt(wy**2 + wz**2)
    
    colors = np.array([ [v, 1.0, 0.0] for v in y ]).flatten()
    
    plt.scatter(np.abs(slip), np.abs( (magf / slip**2) / (magw / slip) ), c=z)
    plt.show()


y, z, fy, fz, wy, wz, slip = coo_force_rot_slip__from_gaussian()
lift_coeff(y, z, fy, fz, wy, wz, slip)

