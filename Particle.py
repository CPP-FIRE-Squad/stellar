import numpy as np

class Particle:

    def __init__(self, pos = [0,0,0], mass = 0, vel = [0,0,0], id = 0, idGen = 0, idChild = 0):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]
        self.mass = mass
        self.vx = vel[0]
        self.vy = vel[1]
        self.vz = vel[2]
        self.id = id
        self.idGen = idGen
        self.idChild = idChild
        self.velocity3d = np.sqrt(np.square(vel[0]) + np.square(vel[1]) + np.square(vel[2]))
        self.velocity2d = np.sqrt(np.square(vel[0]) + np.square(vel[2]))
        self.r3d = np.sqrt(np.square(pos[0]) + np.square(pos[1]) + np.square(pos[2]))
        self.r2d = np.sqrt(np.square(pos[0]) + np.square(pos[2]))

class Star(Particle):
    
    def __init__(self, pos = [0,0,0], mass = 0, vel = [0,0,0], id = 0, idGen = 0, idChild = 0, age = 0, massfraction = 0):
        Particle.__init__(self, pos, mass, vel, id, idGen, idChild)
        self.age = age
        self.massfraction = massfraction

class Gas(Particle):
    
    def __init__(self, pos = [0,0,0], mass = 0, vel = [0,0,0], 
                 id = 0, idGen = 0, idChild = 0, density = 0, massfraction = 0, 
                 temp = 0, hydrogenfraction = 0, electronfraction = 0, size = 0, sfr = 0):
        
        Particle.__init__(self, pos, mass, vel, id, idGen, idChild)
        self.density = density
        self.massfraction = massfraction
        self.temp = temp
        self.electronfraction = electronfraction
        self.hydrogenfraction = hydrogenfraction
        self.size = size
        self.sfr = sfr

class Dark(Particle):
        
    def __init__(self, pos = [0,0,0], mass = 0, vel = [0,0,0], id = 0, idGen = 0, idChild = 0):
        Particle.__init__(self, pos, mass, vel, id, idGen, idChild)


# Helper function to get all particle data for a given halo and particle type
def get_particles(sim, particle, halo_index, halo_center, halo_velocity ):
    # Each particle has a position, velocity, mass, id, idGen, idChild
    positions = sim.particles[particle]['position'] - halo_center
    velocities = sim.particles[particle]['velocity'] - halo_velocity
    masses = sim.particles[particle]['mass']
    ids = sim.particles[particle]['id']
    idGens = sim.particles[particle]['id.generation']
    idChilds = sim.particles[particle]['id.child']
    # Get the distance of each particle from the center of the indicated dark matter halo
    distances = np.sqrt(np.square(positions[:,0]) + np.square(positions[:,1]) + np.square(positions[:,2]))
    # Get the radius of the halo that can actually hold particles. Rhalo, Mhalo, Vhalo <-> Rvir, Mvir, Vvir 
    rgal = sim.get_field('12')[halo_index]
    # Filter out all particles that are too far away
    positions = positions[distances < rgal]
    velocities = velocities[distances < rgal]
    masses = masses[distances < rgal]
    ids = ids[distances < rgal]
    idGens = idGens[distances < rgal]
    idChilds = idChilds[distances < rgal]

    # Now that the general data is filtered, we can filter out the specific data for each particle type

    if particle == 'star':
        # Stars have an additional age and massfraction
        ages = sim.particles[particle]['age']
        massfractions = sim.particles[particle]['massfraction']
        # Filter these out as well
        ages = ages[distances < rgal]
        massfractions = massfractions[distances < rgal]

        stars = []
        for i in range(len(positions)):
            star = Star(positions[i], masses[i], velocities[i], ids[i], idGens[i], idChilds[i], 
                        ages[i], massfractions[i])
            stars.append(star)
        return stars
    elif particle == 'gas':
        # Gas has an additional density, massfraction, temp, hydrogenfraction, electronfraction, size, sfr
        densities = sim.particles[particle]['density']
        massfractions = sim.particles[particle]['massfraction']
        temps = sim.particles[particle]['temperature']
        hydrogenfractions = sim.particles[particle]['hydrogen.neutral.fraction']
        electronfractions = sim.particles[particle]['electron.fraction']
        sizes = sim.particles[particle]['size']
        sfrs = sim.particles[particle]['sfr']
        # Filter these out as well
        densities = densities[distances < rgal]
        massfractions = massfractions[distances < rgal]
        temps = temps[distances < rgal]
        hydrogenfractions = hydrogenfractions[distances < rgal]
        electronfractions = electronfractions[distances < rgal]
        sizes = sizes[distances < rgal]
        sfrs = sfrs[distances < rgal]

        gasses = []
        for i in range(len(positions)):
            g = Gas(positions[i], masses[i], velocities[i], ids[i], idGens[i], idChilds[i], 
                    densities[i], massfractions[i], temps[i], hydrogenfractions[i], electronfractions[i], sizes[i], sfrs[i])
            gasses.append(g)
        return gasses
    elif particle == 'dark':
        darks = []
        for i in range(len(positions)):
            d = Dark(positions[i], masses[i], velocities[i], ids[i], idGens[i], idChilds[i])
            darks.append(d)
        return darks
    