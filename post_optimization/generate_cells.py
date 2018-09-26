#!/usr/bin/env python

import argparse
import numpy as np
import sys

def gen_ic(domain, cell_volume, hematocrit, extent=(7,3,7)):
    assert(0.0 < hematocrit and hematocrit < 0.7)
    
    norm_extent = np.array(extent) / ((extent[0]*extent[1]*extent[2])**(1/3.0))
    
    domain_vol = domain[0]*domain[1]*domain[2]
    ncells = domain_vol*hematocrit / cell_volume
    
    gap = domain_vol**(1/3.0) / (ncells**(1/3.0) + 1)
    
    nx, ny, nz = [ int(domain[i] / (gap*norm_extent[i])) for i in range(3) ]
    real_ht = nx*ny*nz * cell_volume / domain_vol
    h = [ domain[0]/nx, domain[1]/ny, domain[2]/nz ]
    
    com_q = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                com_q.append( [i*h[0], j*h[1], k*h[2],  1, 1, 0, 0] )
                    
    return real_ht, (nx, ny, nz), com_q
    

def get_rbc_params(udx, lscale = 1.5):
    prms = udx.Interactions.MembraneParameters()
    
    p              = 0.000906667 * lscale
    prms.x0        = 0.457    
    prms.ka        = 4900.0
    prms.kb        = 44.4444 * lscale**2
    prms.kd        = 5000
    prms.kv        = 7500.0
    prms.gammaC    = 52.0 * lscale
    prms.gammaT    = 0.0
    prms.kbT       = 0.0444 * lscale**2
    prms.mpow      = 2.0
    prms.theta     = 6.97
    prms.totArea   = 62.2242 * lscale**2
    prms.totVolume = 26.6649 * lscale**3
    prms.ks        = prms.kbT / p
    prms.rnd       = False

    return prms

#====================================================================================
#====================================================================================
    
parser = argparse.ArgumentParser(description='Run the simulation')

parser.add_argument('--debug-lvl', help='Debug level', type=int, default=3)
parser.add_argument('--resource-folder', help='Path to all the required files', type=str, default='./')

parser.add_argument('--a', help='a', type=float, default=80.0)
parser.add_argument('--gamma', help='gamma', type=float, default=20.0)
parser.add_argument('--kbt', help='kbt', type=float, default=1.5)
parser.add_argument('--dt', help='Time step', type=float, default=0.001)
parser.add_argument('--power', help='Kernel exponent', type=float, default=0.5)

parser.add_argument('--final-time', help='Final time', type=float, default=20.0)

parser.add_argument('ht',  help='Hematocrit level', type=float)
parser.add_argument('vol', help='Volume of a single cell', type=float)

parser.add_argument('--dry-run', help="Don't run the simulation, just report the parameters", action='store_true')

args, unknown = parser.parse_known_args()

#====================================================================================
#====================================================================================

rho = 8.0
ranks  = (1, 1, 1)

sdf_file = 'sdf.dat'

with open(sdf_file, 'r', encoding='latin-1') as sdf:
    header = sdf.readline()
    domain = tuple( [float(v) for v in header.split(' ')] )
    
real_ht, ncells, rbcs_ic = gen_ic(domain, args.vol, args.ht)

niters = int(args.final_time / args.dt)

#====================================================================================
#====================================================================================

def report():
    print('Started with the following parameters: ' + str(args))
    if unknown is not None and len(unknown) > 0:
        print('Some arguments are not recognized and will be ignored: ' + str(unknown))
    print('Domain size is: ' + str(domain))
    print('Generated %d cells %s, real hematocrit is %f' % (len(rbcs_ic), str(ncells), real_ht))
    print('')
    sys.stdout.flush()


if args.dry_run:
    report()
    quit()

import udevicex as udx
u = udx.udevicex(ranks, domain, debug_level=args.debug_lvl, log_filename='generate')

if u.isMasterTask():
    report()

#====================================================================================
#====================================================================================

# Interactions:
#   DPD
dpd = udx.Interactions.DPD('dpd', rc=1.0, a=args.a, gamma=args.gamma, kbt=args.kbt, dt=args.dt, power=args.power)
u.registerInteraction(dpd)
#   Contact (LJ)
contact = udx.Interactions.LJ('contact', rc=1.0, epsilon=10, sigma=1.0, object_aware=True, max_force=2000)
u.registerInteraction(contact)
#   Membrane
params = get_rbc_params(udx)
params.gammaT = 1.0
membrane_int = udx.Interactions.MembraneForces('int_rbc', params, stressFree=False, grow_until=args.final_time*0.5)
u.registerInteraction(membrane_int)

# Integrator
vv = udx.Integrators.VelocityVerlet('vv', args.dt)
u.registerIntegrator(vv)

# Wall
post = udx.Walls.SDF('post', sdf_file)
u.registerWall(post)

# RBCs
mesh_rbc = udx.ParticleVectors.MembraneMesh(args.resource_folder + 'rbc_mesh.off')
rbcs = udx.ParticleVectors.MembraneVector('rbc', mass=1.0, mesh=mesh_rbc)
u.registerParticleVector(pv=rbcs, ic=udx.InitialConditions.Membrane(rbcs_ic, global_scale=0.5), checkpoint_every = niters-5)


# Stitching things with each other
#   contact
u.setInteraction(contact, rbcs, rbcs)
#   membrane
u.setInteraction(membrane_int, rbcs, rbcs)


# Integration
u.setIntegrator(vv, rbcs)


# Wall bounce and repulsion
u.registerPlugins( udx.Plugins.createWallRepulsion('repulsion', rbcs, post, C=args.a*4, h=0.4, max_force=args.a*5) )


#====================================================================================
#====================================================================================

u.registerPlugins(udx.Plugins.createStats('stats', every=5000))


u.run(niters)
