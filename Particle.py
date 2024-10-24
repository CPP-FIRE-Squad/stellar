import abc

class Particle(abc.ABC):
    # NOTE: This class is only referencing the values stored in the parent ParticleGroup. We could also just store the values in this object.
    #   Pros of storing values here: 
    #     Faster to access, since every access of an attribute of a particle requires a __getattribute__ call and array access
    #   Cons of storing values here: 
    #     Actions like centering the halo will require an entire new Particles list to be calculated
    #     We would need to store (and thus calculate) *every* attribute, no matter if its used or not. 
    #     Otherwise, if we use the load-on-access functionality like above, we would need to do that for every single star which, when you're only accessing everything once or twice, is very redundant and slow.

    POS_ATTR = None
    VEL_ATTR = None
    MASS_ATTR = None
    ID_ATTR = None
    ID_CHILD_ATTR = None
    ID_GENERATION_ATTR = None
    DISTANCE_ATTR = None
    R2D_ATTR = None
    SPEED_ATTR = None

    def __init__(self, parent_halo, index_in_halo):
        self.parent_halo = parent_halo
        self.index_in_halo = index_in_halo

    @property
    def pos(self):
        return self.parent_halo.__getattribute__(self.POS_ATTR)[self.index_in_halo]

    @property
    def x(self):
        return self.pos[0]

    @property
    def y(self):
        return self.pos[1]

    @property
    def z(self):
        return self.pos[2]

    @property
    def vel(self):
        return self.parent_halo.__getattribute__(self.VEL_ATTR)[self.index_in_halo]

    @property
    def distance(self):
        return self.parent_halo.__getattribute__(self.DISTANCE_ATTR)[self.index_in_halo]

    @property
    def r2d(self):
        return self.parent_halo.__getattribute__(self.R2D_ATTR)[self.index_in_halo]

    @property
    def speed(self):
        return self.parent_halo.__getattribute__(self.SPEED_ATTR)[self.index_in_halo]

    @property
    def vx(self):
        return self.vel[0]

    @property
    def vy(self):
        return self.vel[1]

    @property
    def vz(self):
        return self.vel[2]

    @property
    def mass(self):
        return self.parent_halo.__getattribute__(self.MASS_ATTR)[self.index_in_halo]

    @property
    def id(self):
        return self.parent_halo.__getattribute__(self.ID_ATTR)[self.index_in_halo]
    
    @property
    def id_child(self):
        return self.parent_halo.__getattribute__(self.ID_CHILD_ATTR)[self.index_in_halo]
    
    @property
    def id_generation(self):
        return self.parent_halo.__getattribute__(self.ID_GENERATION_ATTR)[self.index_in_halo]
    
class Star(Particle):
    POS_ATTR = "star_pos"
    VEL_ATTR = "star_vel"
    MASS_ATTR = "star_mass"
    ID_ATTR = "star_id"
    ID_CHILD_ATTR = "star_id_child"
    ID_GENERATION_ATTR = "star_id_generation"
    DISTANCE_ATTR = "star_distance"
    R2D_ATTR = "star_r2d"
    SPEED_ATTR = "star_speed"

    SCALE_FACTOR_ATTR = "star_scale_factor"
    MASS_FRACTION_ATTR = "star_mass_fraction"
    
    @property
    def scale_factor(self):
        return self.parent_halo.__getattribute__(self.SCALE_FACTOR_ATTR)[self.index_in_halo]  # Don't necessarily need to use __getattribute__, since this isn't gonna have a child class. But I don't care because consistency.

    @property
    def mass_fraction(self):
        return self.parent_halo.__getattribute__(self.MASS_FRACTION)[self.index_in_halo]
    
class Gas(Particle):
    POS_ATTR = "gas_pos"
    VEL_ATTR = "gas_vel"
    MASS_ATTR = "gas_mass"
    ID_ATTR = "gas_id"
    ID_CHILD_ATTR = "gas_id_child"
    ID_GENERATION_ATTR = "gas_id_generation"
    DISTANCE_ATTR = "gas_distance"
    R2D_ATTR = "gas_r2d"
    SPEED_ATTR = "gas_speed"

    MASS_FRACTION_ATTR = "gas_mass_fraction"
    DENSITY_ATTR = "gas_density"
    ELECTRON_FRACTION_ATTR = "gas_electron_fraction"
    TEMPERATURE_ATTR = "gas_temperature"
    HYDROGEN_NEUTRAL_FRACTION_ATTR = "gas_hydrogen_neutral_fraction"
    SIZE_ATTR = "gas_size"
    SFR_ATTR = "gas_sfr"

    @property
    def mass_fraction(self):
        return self.parent_halo.__getattribute__(self.MASS_FRACTION_ATTR)[self.index_in_halo]

    @property
    def density(self):
        return self.parent_halo.__getattribute__(self.DENSITY_ATTR)[self.index_in_halo]

    @property
    def electron_fraction(self):
        return self.parent_halo.__getattribute__(self.ELECTRON_FRACTION_ATTR)[self.index_in_halo]

    @property
    def temperature(self):
        return self.parent_halo.__getattribute__(self.TEMPERATURE_ATTR)[self.index_in_halo]

    @property
    def hydrogen_neutral_fraction(self):
        return self.parent_halo.__getattribute__(self.HYDROGEN_NEUTRAL_FRACTION_ATTR)[self.index_in_halo]

    @property
    def size(self):
        return self.parent_halo.__getattribute__(self.SIZE_ATTR)[self.index_in_halo]
    
    @property
    def sfr(self):
        return self.parent_halo.__getattribute__(self.SFR_ATTR)[self.index_in_halo]
  
class Dark(Particle):
    POS_ATTR = "dark_pos"
    VEL_ATTR = "dark_vel"
    MASS_ATTR = "dark_mass"
    ID_ATTR = "dark_id"
    ID_CHILD_ATTR = "dark_id_child"
    ID_GENERATION_ATTR = "dark_id_generation"
    DISTANCE_ATTR = "dark_distance"
    R2D_ATTR = "dark_r2d"
    SPEED_ATTR = "dark_speed"
    
class Dark2(Particle):
    POS_ATTR = "dark2_pos"
    VEL_ATTR = "dark2_vel"
    MASS_ATTR = "dark2_mass"
    ID_ATTR = "dark2_id"
    ID_CHILD_ATTR = "dark2_id_child"
    ID_GENERATION_ATTR = "dark2_id_generation"
    DISTANCE_ATTR = "dark2_distance"
    R2D_ATTR = "dark2_r2d"
    SPEED_ATTR = "dark2_speed"
