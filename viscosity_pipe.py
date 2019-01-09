#!/usr/bin/env python

import ymero as ymr
import argparse

#====================================================================================
#====================================================================================
    
parser = argparse.ArgumentParser(description='Run the simulation')

parser.add_argument('-d', '--debug-lvl', help='Debug level', type=int, default=3)
parser.add_argument('-n', '--niters', help='Number of steps to run', type=int, default=600000)

parser.add_argument('-f', '--force',  help='Pushing force', required=True, type=float)
parser.add_argument('-r', '--radius', help='Pipe radius',   default=20, type=float)

parser.add_argument('--rho', help='Number density', required=True, type=float)
parser.add_argument('--a', help='a', required=True, type=float)
parser.add_argument('--gamma', help='gamma', required=True, type=float)
parser.add_argument('--kbt', help='kbt', required=True, type=float)

parser.add_argument('--dt', help='Time step', default=0.001, type=float)
parser.add_argument('--power', help='Kernel exponent', default=0.5, type=float)
parser.add_argument('--rc', help='Cut-off radius', default=1.0, type=float)


parser.add_argument('--dry-run', help="Don't run the simulation, just report the parameters", action='store_true')
#parser.add_argument('--in-place', help="Compute viscosity on-the-fly instead of saving files", action='store_true')

args = parser.parse_args()

#====================================================================================
#====================================================================================

ranks  = (1, 1, 1)

w = (args.radius + args.rc) * 2 
domain = (args.radius*4, w, w)

u = ymr.ymero(ranks, domain, debug_level=3, log_filename='stdout')

pv = ymr.ParticleVectors.ParticleVector('pv', mass = 1)
ic = ymr.InitialConditions.Uniform(density=args.rho)
u.registerParticleVector(pv=pv, ic=ic)
    
dpd = ymr.Interactions.DPD('dpd', rc=args.rc, a=args.a, gamma=args.gamma, kbt=args.kbt, dt=args.dt, power=args.power)
u.registerInteraction(dpd)

center=(domain[1]*0.5, domain[2]*0.5)
wall = ymr.Walls.Cylinder('cylinder', center=center, radius=0.5*domain[1]-args.rc, axis="x", inside=True)
u.registerWall(wall, 10000)

vv = ymr.Integrators.VelocityVerlet('vv', args.dt)
frozen_wall = u.makeFrozenWallParticles(pvName='wall', walls=[wall], interaction=dpd, integrator=vv, density=args.rho)

u.setWall(wall, pv)

for p in (pv, frozen_wall):
    u.setInteraction(dpd, p, pv)

vvf = ymr.Integrators.VelocityVerlet_withConstForce('vvf', args.dt, force=(args.force, 0, 0))
u.registerIntegrator(vvf)
u.setIntegrator(vvf, pv)

#====================================================================================
#====================================================================================

sampleEvery = 5
dumpEvery   = args.niters // 20
binSize     = (domain[0] / 8, 0.25, 0.25)

field = ymr.Plugins.createDumpAverage('field', [pv], sampleEvery, args.niters*2, binSize, [("velocity", "vector_from_float8")], 'h5/solvent-')
u.registerPlugins(field)
u.run(args.niters + 5)
    
