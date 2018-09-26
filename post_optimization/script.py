#!/usr/bin/env python

import argparse
import pickle
import math
from scipy.optimize import fsolve
import sys


class Viscosity_getter:
    def __init__(self, folder, a, power):
        self.s = pickle.load(open(folder + 'visc_' + str(a) + '_' + str(power) + '_backup.pckl', 'rb'))
        
    def predict(self, gamma):
        return self.s(gamma)

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


def get_fsi_gamma(eta, power, rc=1.0):
    return 0.15e2 / 0.83e2 * eta * (8 * power ** 5 + 60 * power ** 4 + 170 * power ** 3 + 225 * power ** 2 + 137 * power + 30) / math.pi / rc ** 5 / rho

#====================================================================================
#====================================================================================
    
parser = argparse.ArgumentParser(description='Run the simulation')

parser.add_argument('--debug-lvl', help='Debug level', type=int, default=3)
parser.add_argument('--resource-folder', help='Path to all the required files', type=str, default='./')
parser.add_argument('--niters', help='Number of steps to run', type=int, default=100000000)

parser.add_argument('--vx', help='Target vx', required=True, type=float)
parser.add_argument('--vy', help='Target vy', required=True, type=float)

parser.add_argument('--a', help='a', required=True, type=float)
parser.add_argument('--gamma', help='gamma', required=True, type=float)
parser.add_argument('--kbt', help='kbt', required=True, type=float)

parser.add_argument('--dt', help='Time step', default=0.001, type=float)
parser.add_argument('--power', help='Kernel exponent', default=0.5, type=float)

parser.add_argument('--lbd', help='RBC to plasma viscosity ratio', default=5.0, type=float)
parser.add_argument('--fsi-ratio', help='Ratio of viscosity used in liquid and in FSI interaction', default=3.0/4.0, type=float)

parser.add_argument('--dry-run', help="Don't run the simulation, just report the parameters", action='store_true')

parser.add_argument('--verbose', help="Output flow and cells", action='store_true')

args, unknown = parser.parse_known_args()

#====================================================================================
#====================================================================================

rho = 8.0
ranks  = (1, 1, 1)

sdf_file = 'sdf.dat'

with open(sdf_file, 'r', encoding='latin-1') as sdf:
    header = sdf.readline()
    domain = tuple( [float(v) for v in header.split(' ')] )
    
visc_getter = Viscosity_getter(args.resource_folder, args.a, args.power)

mu_outer = visc_getter.predict(args.gamma)
mu_inner = mu_outer * args.lbd

args.outer_gamma = args.gamma
args.inner_gamma = fsolve(lambda g : visc_getter.predict(g) - mu_inner, mu_inner)[0]
args.outer_fsi_gamma = get_fsi_gamma(mu_outer, args.power)
args.inner_fsi_gamma = get_fsi_gamma(mu_inner, args.power)

# just in case
args.gamma = None

#====================================================================================
#====================================================================================

def report():
    print('Started with the following parameters: ' + str(args))
    if unknown is not None and len(unknown) > 0:
        print('Some arguments are not recognized and will be ignored: ' + str(unknown))
    print('Domain size is: ' + str(domain))
    print('')
    sys.stdout.flush()


if args.dry_run:
    report()
    quit()

import udevicex as udx
u = udx.udevicex(ranks, domain, debug_level=args.debug_lvl, log_filename='log')

if u.isMasterTask():
    report()

#====================================================================================
#====================================================================================
    
# outer PV
outer = udx.ParticleVectors.ParticleVector('outer', mass = 1.0)
ic = udx.InitialConditions.Uniform(density=rho)
u.registerParticleVector(pv=outer, ic=ic)

# Interactions:
#   DPD
dpd = udx.Interactions.DPD('dpd', rc=1.0, a=args.a, gamma=args.outer_gamma, kbt=args.kbt, dt=args.dt, power=args.power)
u.registerInteraction(dpd)
#   Contact (LJ)
contact = udx.Interactions.LJ('contact', rc=1.0, epsilon=1.0, sigma=0.9, object_aware=True, max_force=750)
u.registerInteraction(contact)
#   Membrane
membrane_int = udx.Interactions.MembraneForces('int_rbc', get_rbc_params(udx), stressFree=True)
u.registerInteraction(membrane_int)

# Integrator
vv = udx.Integrators.VelocityVerlet('vv', args.dt)
u.registerIntegrator(vv)

subvv = udx.Integrators.SubStepMembrane('subvv', args.dt, 5, membrane_int)
u.registerIntegrator(subvv)

# Wall
post = udx.Walls.SDF('post', sdf_file)
u.registerWall(post, 1000)
frozen = u.makeFrozenWallParticles('frozen', walls=[post], interaction=dpd, integrator=vv, density=rho,  nsteps = 10)

# RBCs
mesh_rbc = udx.ParticleVectors.MembraneMesh(args.resource_folder + 'rbc_mesh.off')
rbcs = udx.ParticleVectors.MembraneVector('rbc', mass=1.0, mesh=mesh_rbc)
u.registerParticleVector(pv=rbcs, ic=udx.InitialConditions.Restart('generated/'))

checker = udx.BelongingCheckers.Mesh('checker')
u.registerObjectBelongingChecker(checker, rbcs)
inner = u.applyObjectBelongingChecker(checker, outer, inside='inner', correct_every=5000)

# Bouncer
bouncer = udx.Bouncers.Mesh('bouncer')
u.registerBouncer(bouncer)



# Stitching things with each other
#   dpd
if u.isComputeTask():
    dpd.setSpecificPair(rbcs,  outer,  a=0, gamma=args.outer_fsi_gamma)
    dpd.setSpecificPair(rbcs,  inner,  a=0, gamma=args.inner_fsi_gamma)
    dpd.setSpecificPair(inner, outer,  gamma=0, kbt=0)
    dpd.setSpecificPair(inner, inner,  gamma=args.inner_gamma)

u.setInteraction(dpd, outer,  outer)
u.setInteraction(dpd, frozen, outer)
u.setInteraction(dpd, outer, inner)
u.setInteraction(dpd, outer, rbcs)
    
u.setInteraction(dpd, inner, inner)
u.setInteraction(dpd, rbcs, inner)
u.setInteraction(dpd, frozen, inner)

u.setInteraction(dpd, frozen, rbcs)

#   contact
u.setInteraction(contact, rbcs, rbcs)

#   membrane
# don't set it since we're using substep
# u.setInteraction(membrane_int, rbcs, rbcs)


# Integration
for pv in [inner, outer]:
    u.setIntegrator(vv, pv)
u.setIntegrator(subvv, rbcs)


# Wall bounce and repulsion
u.setWall(post, outer)
u.registerPlugins( udx.Plugins.createWallRepulsion('repulsion', rbcs, post, C=args.a*2, h=0.25, max_force=args.a*3) )

# Membrane bounce
u.setBouncer(bouncer, rbcs, inner)
u.setBouncer(bouncer, rbcs, outer)

#====================================================================================
#====================================================================================

factor = 0.08
Kp = 2.0 * factor
Ki = 1.0 * factor
Kd = 8.0 * factor

u.registerPlugins(udx.Plugins.createStats('stats', every=500, filename='stats.txt'))
u.registerPlugins(udx.Plugins.createDumpObjectStats('obj', ov=rbcs, dump_every=200, path='pos/'))

u.registerPlugins(udx.Plugins.createVelocityControl('vc', filename='vcont.txt',
                                                    pvs = [outer], low = (0, 0, 0), high = domain,
                                                    sampleEvery = 5, dumpEvery = 500, targetVel = (args.vx, args.vy, 0),
                                                    Kp=Kp, Ki=Ki, Kd=Kd))

if args.verbose:
    u.dumpWalls2XDMF([post], (0.5, 0.5, 0.5))
    u.registerPlugins(udx.Plugins.createDumpAverage(
            'field', [inner, outer],
            sample_every=5, dump_every=1000, bin_size=(0.5, 0.5, 0.5),
            channels=[("velocity", "vector_from_float8")], path='xdmf/flow-'))
    
    u.registerPlugins(udx.Plugins.createDumpMesh('rbcs', rbcs, dump_every=500, path = 'ply/'))

u.run(args.niters)
