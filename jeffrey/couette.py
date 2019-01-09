#!/usr/bin/env python

import argparse
import udevicex as udx
import sys
import trimesh
import numpy as np

#====================================================================================
#====================================================================================
    
parser = argparse.ArgumentParser(description='Run the simulation')

parser.add_argument('--debug-lvl', help='Debug level', type=int, default=3)
parser.add_argument('--niters', help='Number of steps to run', type=int, default=100000000)

parser.add_argument('--shear', help='Shear rate x', required=True, type=float)
parser.add_argument('--axes',  help='Ellipsoid semi-axes', required=True, type=float, nargs=3)
parser.add_argument('--domain', help='Domain size', type=float, nargs=3, default=[64, 64, 64])

parser.add_argument('--a', help='a', required=True, type=float, default=50.0)
parser.add_argument('--gamma', help='gamma', required=True, type=float, default=50.0)
parser.add_argument('--kbt', help='kbt', required=True, type=float, default=1.0)
parser.add_argument('--dt', help='Time step', default=0.001, type=float)
parser.add_argument('--power', help='Kernel exponent', default=0.5, type=float)

parser.add_argument('--dry-run', help="Don't run the simulation, just report the parameters", action='store_true')

args, unknown = parser.parse_known_args()

#====================================================================================
#====================================================================================

rho = 8.0
ranks  = (1, 1, 1)
rc = 1
vx = args.shear * (args.domain[2] - rc) / 2.0


def report():
    print('Started with the following parameters: ' + str(args))
    if unknown is not None and len(unknown) > 0:
        print('Some arguments are not recognized and will be ignored: ' + str(unknown))
        
    print('Wall velocity: ' + str(vx))
    print('')
    sys.stdout.flush()


if args.dry_run:
    report()
    quit()
    


u = udx.udevicex(ranks, args.domain, debug_level=args.debug_lvl, log_filename='log')

if u.isMasterTask():
    report()

#====================================================================================
#====================================================================================
    

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=rho)
u.registerParticleVector(pv=pv, ic=ic)

# Object
coords = np.loadtxt('ellipsoid.txt').tolist()
com_q = [[0.5 * args.domain[0], 0.5 * args.domain[1], 0.5 * args.domain[2],   1., 0, 0, 0]]

sph_mesh = trimesh.creation.icosphere(subdivisions=3, radius = 1.0)
for i in range(3):
    sph_mesh.vertices[:,i] *= args.axes[i]
mesh = udx.ParticleVectors.Mesh(sph_mesh.vertices.tolist(), sph_mesh.faces.tolist())
pvEllipsoid = udx.ParticleVectors.RigidEllipsoidVector('ellipsoid', mass=1, object_size=len(coords), semi_axes=args.axes, mesh=mesh)

vvEllipsoid = udx.Integrators.RigidVelocityVerlet("ellvv", args.dt)
u.registerParticleVector(pv=pvEllipsoid, ic=udx.InitialConditions.Rigid(com_q=com_q, coords=coords))
u.registerIntegrator(vvEllipsoid)
u.setIntegrator(vvEllipsoid, pvEllipsoid)

# DPD
dpd = udx.Interactions.DPD('dpd', rc=rc, a=args.a, gamma=args.gamma, kbt=args.kbt, dt=args.dt, power=args.power)
u.registerInteraction(dpd)

# Walls
plate_lo = udx.Walls.MovingPlane('plate_lo', normal=(0, 0, -1), pointThrough=(0, 0,                   rc), velocity=( vx, 0, 0))
plate_hi = udx.Walls.MovingPlane('plate_hi', normal=(0, 0,  1), pointThrough=(0, 0,  args.domain[2] - rc), velocity=(-vx, 0, 0))

u.registerWall(plate_lo, 10000)
u.registerWall(plate_hi, 10000)

vv = udx.Integrators.VelocityVerlet("vv", args.dt)
frozen_lo = u.makeFrozenWallParticles(pvName="plate_lo", walls=[plate_lo], interaction=dpd, integrator=vv, density=rho, nsteps=100)
frozen_hi = u.makeFrozenWallParticles(pvName="plate_hi", walls=[plate_hi], interaction=dpd, integrator=vv, density=rho, nsteps=100)

u.setWall(plate_lo, pv)
u.setWall(plate_hi, pv)

for p in [pv, frozen_lo, frozen_hi, pvEllipsoid]:
    u.setInteraction(dpd, p, pv)

u.registerIntegrator(vv)
u.setIntegrator(vv, pv)

move_plus  = udx.Integrators.Translate('move_p', dt=args.dt, velocity=( vx, 0, 0))
move_minus = udx.Integrators.Translate('move_m', dt=args.dt, velocity=(-vx, 0, 0))
u.registerIntegrator(move_plus)
u.registerIntegrator(move_minus)
u.setIntegrator(move_plus,  frozen_lo)
u.setIntegrator(move_minus, frozen_hi)

#sampleEvery = 2
#dumpEvery   = 1000
#binSize     = (8., 8., 1.0)
#field = udx.Plugins.createDumpAverage('field', [pv], sampleEvery, dumpEvery, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
#u.registerPlugins(field)

velocity = (0,0,0)
omega = (udx.Plugins.PinObject.Unrestricted,)*3
u.registerPlugins( udx.Plugins.createPinObject('pin', ov=pvEllipsoid, dump_every=5000, path='force/', velocity=velocity, angular_velocity=omega) )
u.registerPlugins( udx.Plugins.createStats('stats', 'stats.txt', 1000) )
u.registerPlugins( udx.Plugins.createDumpObjectStats('obj_stats', ov=pvEllipsoid, dump_every=100, path='pos') )

u.run(args.niters)

