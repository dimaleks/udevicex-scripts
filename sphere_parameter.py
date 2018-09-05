import pickle
from scipy import interpolate
from scipy.optimize import fsolve
import numpy as np

def params(Re, r, u, s):
	
	rho = 8.0
	
	mu = 2*r*u*rho / Re
	#print mu
	
	gamma = fsolve(lambda g : s(g)-mu, mu)[0]	
	
	return gamma

#s = pickle.load(open('../data/visc_80.0_0.5_backup.pckl', 'rb'))
s = pickle.load(open('../data/visc_160.0_1.0_backup.pckl', 'rb'))
r = 5.0

for Re, r, u, kx, kyz in [(0.5,   5,  0.4, 24, 8),
						  (2.0,   5,  0.4, 24, 8),
						  (10.0, 10,  0.8, 16, 6),
						  (25.0, 10,  0.8, 16, 6),
						  (50.0, 12,  1.0, 16, 6)]:
	gamma = params(Re, r, u, s)
	lx = kx * 2*r
	lyz = kyz * 2*r
	print('./run.sh  160 %8.4f 3.0 1.0 8  dt  %3.0f %3.0f  %3.2f   %.0f %.0f %.0f   '
       % (gamma, r, 2.0*r, u, lx, lyz, lyz))
