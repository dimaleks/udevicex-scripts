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

import thesis_plot as myplt


kappa = 0.22
Re = 100
torques = {50 : 4000, 100 : 3000, 200 : 2000}

figname = 'Re_' + str(Re) + '_kappa_' + str(kappa) + '_torque_' + str(torques[Re]) 

dir1 = 'data/focusing/'
fname1 = 'Re_' + str(Re) + '_kappa_' + str(kappa) + '__rotation_0.pckl'

dir2 = 'data/focusing/' 
fname2 = 'Re_' + str(Re) + '_kappa_' + str(kappa) + '__rotation_1.pckl'

#dir2 = 'data/focusing/torque/' 
#fname2 = 'Re_' + str(Re) + '_kappa_' + str(kappa) + '__torque_' + str(torques[Re]) + '.pckl'

figure_folder = 'media/focusing/'

def pois_square_velocity_gradient(y, z, H, dp, mu):
    
    def term(n):
        return [-1 / (2 * n + 1) ** 2 * np.pi / H * np.sinh((2 * n + 1) * np.pi * y / H) / np.cosh((2 * n + 1) * np.pi / 2) * np.sin((2 * n + 1) * np.pi * z / H),
                -1 / (2 * n + 1) ** 2 * np.pi / H * (-np.cosh((2 * n + 1) * np.pi / 2) + np.cosh((2 * n + 1) * np.pi * y / H)) * np.cos((2 * n + 1) * np.pi * z / H) / np.cosh((2 * n + 1) * np.pi / 2)]
    nterms = 20
    
    total = np.zeros((2, y.size))
    for n in range(nterms):
        total += term(n)

    return 4.0*H*H * dp / (np.pi**3 * mu) * total


def coo_force_rot_slip__from_data():
    
    f1 = pickle.load(open(dir1 + 'forcedata_' + fname1,'rb'))
    f2 = pickle.load(open(dir2 + 'forcedata_' + fname2,'rb'))
    
    w1 = pickle.load(open(dir1 + 'rotationdata_' + fname1,'rb'))
    try:
        w2 = pickle.load(open(dir2 + 'rotationdata_' + fname2,'rb'))
    except:
        w2 = f2*0
    
    slip_vel = pickle.load(open(dir1 + 'velocitydata_' + fname1,'rb'))
        
    ids1 = f1[:,0]*100 + f1[:,1]
    ids2 = f2[:,0]*100 + f2[:,1]
        
    tol = 1e-10
    common1 = (np.abs(ids2[:,None] - ids1) < tol).any(0)
    common2 = (np.abs(ids1[:,None] - ids2) < tol).any(0)
    
    y = f1[common1, 0]
    z = f1[common1, 1]
    
    fy = f1[common1, 2] - f2[common2, 2]
    fz = f1[common1, 3] - f2[common2, 3]
    
    wy = w1[common1, 2] - w2[common2, 2]
    wz = w1[common1, 3] - w2[common2, 3]
    
    slip = slip_vel[common1, 2]
    
    return y, z, fy, fz, wy, wz, slip

def coo_force_rot_slip__from_gaussian():
    
    f1 = pickle.load(open(dir1 + 'force_' + fname1,'rb'))
    f2 = pickle.load(open(dir2 + 'force_' + fname2,'rb'))
    
    w1 = pickle.load(open(dir1 + 'omega_' + fname1,'rb'))
    try:
        w2 = pickle.load(open(dir2 + 'omega_' + fname2,'rb'))
    except:
        w2 = None
        
    slip_vel   = pickle.load(open(dir1 + 'slip_' + fname1,'rb'))
    
    x = np.linspace(0, 0.75, 20)
    y = np.linspace(0, 0.75, 20)
    X,Y = np.meshgrid(x,y)
    grid = np.array([X.flatten(),Y.flatten()]).T
    
    fy = f1[0].predict(grid) - f2[0].predict(grid)
    fz = f1[1].predict(grid) - f2[1].predict(grid)
    
    if w2 is None:
        wy = w1[0].predict(grid)
        wz = w2[1].predict(grid)
    else:
        wy = w1[0].predict(grid) - w2[0].predict(grid)
        wz = w1[1].predict(grid) - w2[1].predict(grid)
    
    
    slip = slip_vel.predict(grid)
    
    return grid[:,0], grid[:,1], fy, fz, wy, wz, slip


def force_vs_rot(y, z, fy, fz, wy, wz, save=False, fname='', title=''):
    
    fig = plt.figure()
    
    plt.title(title)
    plt.quiver(y, z, fy, fz)
    plt.quiver(y, z, wy, wz, color='C3')
    
    if save:
        fig.savefig(fname, bbox_inches='tight')
        plt.show()
        #plt.close(fig)
    else:
        plt.show()

def angle(y, z, fy, fz, wy, wz, save=False, fname='', title=''):
    
    fig = plt.figure()
    
    N = 20
    yi = np.linspace(0, 0.8, N)
    zi = np.linspace(0, 0.8, N)
    Y,Z = np.meshgrid(yi,zi)

    
    ang = np.abs( (fy*wy + fz*wz) / (fy*fy + fz*fz)**0.5 / (wy*wy + wz*wz)**0.5 )
    
    # Z is a matrix of x-y values
    angi = scinterp.griddata((y, z), ang, (yi[None,:], zi[:,None]), method='cubic')
    
    #plt.title(title)
    plt.pcolormesh(Y, Z, angi.reshape(Y.shape), shading='gouraud', cmap=plt.cm.plasma, vmax=0.12, vmin=0.0)
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 20
    cbar.ax.set_ylabel(r'$\cos(\bm{F}^{\text{diff}}_l, \bm{\omega})$', size=9, rotation=270)
    cbar.ax.tick_params(labelsize=8) 

    
    plt.xlim(0, 0.65)
    plt.ylim(0, 0.65)
    
    plt.xlabel('$y / (H/2)$')
    plt.ylabel('$z / (H/2)$')
    
    myplt.set_font_sizes(plt.gca())

    if save:
        myplt.save_figure(fig, 3, fname, dpi=600)
        #plt.close(fig)
    else:
        plt.show()


def lift_coeff(y, z, fy, fz, wy, wz, slip, save=False, fname='', title=''):
    
    fig = plt.figure()

    magf = np.sqrt(fy**2 + fz**2)
    magw = np.sqrt(wy**2 + wz**2)
    
    colors = np.array([ [v, 1.0, 0.0] for v in y ]).flatten()
    
    xdata = np.abs(y)[ y+z > 0.001 ]
    ydata = np.abs( (magf / slip**2) / (magw / slip) )[ y+z > 0.001 ]
    
    p = plt.scatter(xdata, ydata, c=z[ y+z > 0.001 ])
    cbar = plt.colorbar(p)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('z', rotation=270, fontsize=16)
    
    plt.xlabel(r'y', fontsize=16)
    plt.ylabel(r'$\frac{C_{l}}{|\mathbf{\omega}|}$', fontsize=18)
    plt.title(title)
    plt.tight_layout()
    
    if save:
        fig.savefig(fname, bbox_inches='tight')
        plt.show()
        #plt.close(fig)
    else:
        plt.show()


y, z, fy, fz, wy, wz, slip = coo_force_rot_slip__from_gaussian()

H = 45.455
vel_grad = pois_square_velocity_gradient(y*H/2, (1+z)*H/2, H, 1.0, 1.0)

myplt.set_qt_backend()

#lift_coeff(y, z, fy, fz, wy, wz, slip, save=True, fname='force_vs_wall'+str(Re)+'.pdf', title=r'Re='+str(Re))

angle(y, z, wy, wz, fy, fz, save=True, fname='square/force_vs_rot'+str(Re)+'.pdf')

#angle(y, z, wy, wz, fy, fz, save=True, fname='force_vs_0_rot'+str(Re)+'.png',
#      title=r'$\cos( \phi(\mathbf{\omega}, \mathbf{F}_{diff}) )$, Re='+str(Re))

#force_vs_rot(y, z, wy, wz, fy, fz)








