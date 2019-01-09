import udevicex as udx

ranks  = (1, 1, 1)
domain = (32, 16, 4)
u = udx.udevicex(ranks, domain, debug_level=3, log_filename='log')

pv = udx.ParticleVectors.ParticleVector('pv', mass = 1)
ic = udx.InitialConditions.Uniform(density=-10)
u.registerParticleVector(pv=pv, ic=ic)

u.run(1)