import stellarutil
import numpy as np
from typing import Sequence
import pickle

class ParticleAttrs:
    POSITION = "position"
    VELOCITY = "velocity"
    MASS = "mass"
    DISTANCE = "distance"
    SPEED = "speed"
    ID = "id"
    
    STAR_OR_GAS_MASS_FRACTION = "massfraction"
    STAR_SCALE_FACTOR = "form.scalefactor"
    GAS_DENSITY = "density"
    GAS_ELECTRON_FRACTION = "electron.fraction"
    GAS_TEMPERATURE = "temperature"
    GAS_HYDROGEN_NEUTRAL_FRACTION = "hydrogen.neutral.fraction"
    GAS_SIZE = "size"

class Faces:
    xy = "xy"
    xz = "xz"
    yz = "yz"

class Halo:
    #TODO: Make parent ParticleGroup class, make halo class a child w/ extra halo-specific functionality
    #TODO: Make optional immediately_load parameter
    #TODO: Add get_host() and get_children(). I'm assuming it should just return Halo & List[Halo], with same include settings (maybe have immediately_load parameter an option)
    #        or just use *args and **kwargs
    #TODO: Make custom functions for masking restrictions (i.e. |, ^, etc.). If & is chosen, existing boolean optimizations can be used. 
    #      Otherwise, just set mask to old_mask <operation> new_mask and call set_particles with boolean=False
    #TODO: Comment code and add docstrings

    #TODO: Make every call that generates particles a getter. Structure, given particle (i.e. star) and attribute (i.e. pos)
    # My original thought included making self.particle_mask a getter, but I don't think this would be of any use
    """
    @property
    def particle_attribute(self):
        if self._attribute is None:
            self._attribute = self.sim.particles["particle"][self.particle_mask]
        return self._attribute
        
    def restrict(self):
        ...  # Do restrictions, which change self.particle_mask
        self._attribute = None  # For every attribute

    def apply_AND_boolean_mask():  # If I want to do this (I might just scrap the efficient AND boolean masking anyways)
        ...  # Find boolean mask
        self._attribute = self._attribute[mask]  # For every attribute
    """
    # Trying to think of ways to not have to just make 26 different getters all doing basically the same thing (except for maybe 3DR and speed, but even those are repeated) (+6 for dark2)
    # We could do set_attr but I really don't want to do that because it just gets so messy and I don't think that would work well with documentation and IDEs
    """Example of set_attr to define functions. test_function can then be dyamically generated to quickly create all getters

    class TestClass:
        def __init__(self, test_variable):
            self.test_variable = test_variable

    def test_function(self, foo):
        print(self.test_variable, foo)

    setattr(TestClass, "test_method", test_function)

    a = TestClass(123)
    a.test_method("bar")
    """
    # But, then again, hear me out:
    """
    class Halo:
        star_pos: list[list[float]]  # This line is not required, and simply makes it more intuitive to use this with IDEs
        '''Array of positions of all the stars in this halo'''  # This appears as a description in VSCode (and likely other IDEs)

        def __init__(self):
            ...

    attributes = [
        {"method_name": "star_pos", "variable_name": "_star_pos", "particle_type": "star", "attribute_name": "position", "mask_name": "star_mask"},
        {"method_name": "gas_vel", "variable_name": "_gas_vel", "particle_type": "gas", "attribute_name": "velocity", "mask_name": "gas_mask"}
    ]  # I'm just using a dictionary here cause it's easy and nicer to read. This could be a list of lists or objects as well

    for attribute in attributes:
        method_name, variable_name, particle_type, attribute_name, mask_name = attribute.values()
        setattr(Halo, variable_name, None)

        # Could also do:
        # variable_name = f"_{method_name}"

        @property
        def getter(self):
            if self.__getattribute__(variable_name) is None:
                # new_value = self.sim.particles[particle_type][attribute_name][mask_name]
                new_value = f"{particle_type} {attribute_name}"
                self.__setattr__(variable_name, new_value)
                return new_value
            else:
                return self.__getattribute__(variable_name)
            
        setattr(Halo, method_name, getter)

    a = Halo()
    print(a.star_pos)
    print(a.star_pos)

    """



    # New features:
    # Gas, dark, and dark2 are stored
    # *Every* parameter of each particle is stored
    # Values are stored in numpy arrays now (Stars list still does exist though)
    # You have a choice over what values/parameters/particles are loaded, and even if the particles are loaded before the first restriction
    # Restrict ids
    # Restrict can now be boolean masked (i.e. you can use restrict_ids with a boolean mask after restrict_percentage to restrict both)
    # Can center on specific coordinates
    # Can save to/load from file
    # Halo initialization is in halo class
    # General optimizations (with how values are calculated, etc.)
    # Extract_x, y, z for getting values from array of vectors

    def __init__(self, 
                 sim, 
                 halo_id: int, 
                 incl_stars: bool | Sequence[str] = True, 
                 incl_gas: bool | Sequence[str] = False, 
                 incl_dark: bool | Sequence[str] = False, 
                 incl_dark2: bool | Sequence[str] = False,
                 incl_halo_info: bool = True,
                 immediately_load_particles = True,
                 restrict_percentage: float | int | None = 100):
        """Get a new halo object including the selected particles.

        :param sim: The Sim object to extract halo from.
        :param halo_id: The ID of the halo to extract.
        :param incl_stars: Whether to include this halo's stars in this object. True/False to include all/no attributes, or iterable of strings/ParticleAttrs values to choose specific attributes.
        :param incl_gas: See incl_stars, but for gas particles.
        :param incl_dark: See incl_stars, but for all dark matter particles.
        :param incl_dark2: See incl_stars, but for a reduced quality of dark matter particles.
        :param incl_halo_info: Whether to load the halo's info beyond its center pos+vel and radius.
        :param immediately_load_particles: Whether to load the included particles upon init, using restrict_percentage with the provided value.
        :param restrict_percentage: If particles are immediately loaded, what restrict_percentage should be used. Set this to false if you are immediately filtering a different way.
        """
        self.id = halo_id

        # The position that all particle positions are relative to. The absolute coordinates of the Zero in the active reference frame. Not necessarily the center of this halo. This can be changed.
        self.center_pos = np.array([sim.ahf_data.field('Xc(6)')[halo_id], sim.ahf_data.field('Yc(7)')[halo_id], sim.ahf_data.field('Zc(8)')[halo_id]]) / sim.h
        # The absolute coordinates of the center of this halo. This does not change, and any particle restrictions are based on this center.
        self.this_halo_center_pos = np.array([self.center_pos[0], self.center_pos[1], self.center_pos[2]])
        # See center_pos, but for velocity
        self.center_vel = np.array([sim.ahf_data.field('VXc(9)')[halo_id], sim.ahf_data.field('VYc(10)')[halo_id], sim.ahf_data.field('VZc(11)')[halo_id]]) / sim.h
        # See this_halo_center_pos, but for velocity.
        self.this_halo_center_vel = np.array([self.center_vel[0], self.center_vel[1], self.center_vel[2]])
        self.halo_radius = sim.get_field('12')[halo_id]

        self.sim = sim

        if incl_halo_info:
            self.hostID = sim.ahf_data.field('hostHalo(2)')[halo_id]
            self.mass = sim.get_field('4')[halo_id]
            self.r_max = sim.ahf_data.field('Rmax(13)')[halo_id] / sim.h
            self.v_max = sim.ahf_data.field('Vmax(17)')[halo_id]
            self.v_esc = sim.ahf_data.field('v_esc(18)')[halo_id]
            self.num_gas = sim.ahf_data.field('n_gas(44)')[halo_id]
            self.gas_mass = sim.ahf_data.field('M_gas(45)')[halo_id]
            self.num_stars = sim.ahf_data.field('n_star(64)')[halo_id]
            self.star_mass = sim.ahf_data.field('M_star(65)')[halo_id]
            self.num_particles = sim.ahf_data.field('npart(5)')[halo_id]
        else:
            self.hostID = self.mass = self.r_max = self.v_max = self.v_esc = self.num_gas = self.gas_mass = self.num_stars = self.star_mass = self.num_particles = None

        self.incl_stars = incl_stars
        self.incl_gas = incl_gas
        self.incl_dark = incl_dark
        self.incl_dark2 = incl_dark2
        self.incl_halo_info = incl_halo_info

        self._stars = None
        self._particles_have_been_loaded = False

        self.star_pos = self.star_vel = self.star_id = self.star_mass = self.star_scale_factor = self.star_mass_fraction = self.star_distance = self.star_speed = None
        self.gas_pos = self.gas_vel = self.gas_id = self.gas_mass = self.gas_mass_fraction = self.gas_density = self.gas_electron_fraction = self.gas_temperature = self.gas_hydrogen_neutral_fraction = self.gas_size = self.gas_distance = self.gas_speed = None
        self.dark_pos = self.dark_vel = self.dark_id = self.dark_mass = self.dark_distance = self.dark_speed = None
        self.dark2_pos = self.dark2_vel = self.dark2_id = self.dark2_mass = self.dark2_distance = self.dark2_speed = None

        if immediately_load_particles:
            if restrict_percentage is not None:
                self.restrict_percentage(restrict_percentage)
            else:
                self.reset_restriction()

    def restrict_percentage(self, percentage: float | int | None, boolean=False):
        if percentage is None:
            return self.reset_restriction()  #TODO: If reset_restriction adds a boolean mask option, this should pass in boolean

        if boolean and not self._particles_have_been_loaded:
            boolean = False

        self.restricted_halo_radius = self.halo_radius * (percentage / 100)

        def get_particle_in_halo_filter(to_include, particles):
            if to_include:
                all_rel_particle_pos = (self.sim.particles[particles]['position'] - self.this_halo_center_pos) \
                    if isinstance(particles, str) else particles
                    
                particle_pos_sum_of_squares = np.sum(np.square(all_rel_particle_pos), 1)
                return particle_pos_sum_of_squares < self.restricted_halo_radius**2
            else:
                return None

        self.stars_in_halo_filter = get_particle_in_halo_filter(self.incl_stars, self.star_pos if boolean else 'star')
        self.gas_in_halo_filter = get_particle_in_halo_filter(self.incl_gas, self.gas_pos if boolean else 'gas')
        self.dark_in_halo_filter = get_particle_in_halo_filter(self.incl_dark, self.dark_pos if boolean else 'dark')
        self.dark2_in_halo_filter = get_particle_in_halo_filter(self.incl_dark2, self.dark2_pos if boolean else 'dark2')

        return self.set_particles(boolean)

    def restrict_slice(self, face = 'xy', proj_distance = 1, thickness = 1, boolean=False):
        face = face.lower()
        if boolean and not self._particles_have_been_loaded:
            boolean = False

        def get_particle_in_slice_filter(to_include, particles):
            if to_include:
                all_rel_particle_pos = (self.sim.particles[particles]['position'] - self.this_halo_center_pos) \
                    if isinstance(particles, str) else particles

                if face == 'xy' or face == 'yx': 
                    square_distance_to_axis = np.sum(np.square(all_rel_particle_pos[:, [0, 1]]), 1)
                    distance_to_plane = np.abs(all_rel_particle_pos[:, 2])
                elif face == 'xz' or face == 'zx': 
                    square_distance_to_axis = np.sum(np.square(all_rel_particle_pos[:, [0, 2]]), 1)
                    distance_to_plane = np.abs(all_rel_particle_pos[:, 1])
                elif face == 'yz' or face == 'zy': 
                    square_distance_to_axis = np.sum(np.square(all_rel_particle_pos[:, [1, 2]]), 1)
                    distance_to_plane = np.abs(all_rel_particle_pos[:, 0])

                return (square_distance_to_axis < proj_distance**2) & (distance_to_plane < thickness)
            else:
                return None
            
        self.stars_in_halo_filter = get_particle_in_slice_filter(self.incl_stars, self.star_pos if boolean else 'star')
        self.gas_in_halo_filter = get_particle_in_slice_filter(self.incl_gas, self.gas_pos if boolean else 'gas')
        self.dark_in_halo_filter = get_particle_in_slice_filter(self.incl_dark, self.dark_pos if boolean else 'dark')
        self.dark2_in_halo_filter = get_particle_in_slice_filter(self.incl_dark2, self.dark2_pos if boolean else 'dark2')

        return self.set_particles(boolean)

    def reset_restriction(self):
        self.stars_in_halo_filter = np.full(len(self.sim.particles['star']['position']), True) if self.incl_stars else None
        self.gas_in_halo_filter = np.full(len(self.sim.particles['gas']['position']), True) if self.incl_gas else None
        self.dark_in_halo_filter = np.full(len(self.sim.particles['dark']['position']), True) if self.incl_dark else None
        self.dark2_in_halo_filter = np.full(len(self.sim.particles['dark2']['position']), True) if self.incl_dark2 else None

        return self.set_particles(False)

    def restrict_ids(self, star_ids=None, gas_ids=None, dark_ids=None, dark2_ids=None, boolean=False):
        if boolean and not self._particles_have_been_loaded:
            boolean = False

        def get_particles_with_id_filter(to_include, particles, ids):
            if to_include and ids is not None:
                all_particle_ids = self.sim.particles[particles]['id'] if isinstance(particles, str) else particles
                return np.isin(all_particle_ids, ids)
            else:
                return None
        
        self.stars_in_halo_filter = get_particles_with_id_filter(self.incl_stars, self.star_id if boolean else "star", star_ids)
        self.gas_in_halo_filter = get_particles_with_id_filter(self.incl_gas, self.gas_id if boolean else "gas", gas_ids)
        self.dark_in_halo_filter = get_particles_with_id_filter(self.incl_dark, self.dark_id if boolean else "dark", dark_ids)
        self.dark2_in_halo_filter = get_particles_with_id_filter(self.incl_dark2, self.dark2_id if boolean else "dark2", dark2_ids)
            # Could sort stars by ID, then index it by star_ids (probably not, since IDs have gaps)
            # NEVERMIND np.isin IS AMAZING

        return self.set_particles(boolean)

    def set_particles(self, boolean=False):
        def conditional_attribute(to_include, attribute, enum_value, index_filter=None):
            if to_include is True or (to_include is not False and enum_value in to_include):
                return attribute[index_filter] if index_filter is not None else attribute
            return None
        
        if self.incl_stars:
            self.star_pos = conditional_attribute(self.incl_stars, self.star_pos if boolean else self.sim.particles['star']['position'], ParticleAttrs.POSITION, self.stars_in_halo_filter)
            if self.star_pos is not None and not boolean: self.star_pos -= self.center_pos
            self.star_vel = conditional_attribute(self.incl_stars, self.star_vel if boolean else self.sim.particles['star']['velocity'], ParticleAttrs.VELOCITY, self.stars_in_halo_filter)
            if self.star_vel is not None and not boolean: self.star_vel -= self.center_vel
            self.star_id = conditional_attribute(self.incl_stars, self.star_id if boolean else self.sim.particles['star']['id'], ParticleAttrs.ID, self.stars_in_halo_filter)
            self.star_mass = conditional_attribute(self.incl_stars, self.star_mass if boolean else self.sim.particles['star']['mass'], ParticleAttrs.MASS, self.stars_in_halo_filter)

            self.star_scale_factor = conditional_attribute(self.incl_stars, self.star_scale_factor if boolean else self.sim.particles['star']['form.scalefactor'], ParticleAttrs.STAR_SCALE_FACTOR, self.stars_in_halo_filter)
            self.star_mass_fraction = conditional_attribute(self.incl_stars, self.star_mass_fraction if boolean else self.sim.particles['star']['massfraction'], ParticleAttrs.STAR_OR_GAS_MASS_FRACTION, self.stars_in_halo_filter)

            self.star_distance = conditional_attribute(self.incl_stars, np.sqrt(np.sum(np.square(self.star_pos), 1)), ParticleAttrs.DISTANCE) if self.star_pos is not None and not boolean else self.star_distance[self.stars_in_halo_filter] if self.star_distance is not None else None
            self.star_speed = conditional_attribute(self.incl_stars, np.sqrt(np.sum(np.square(self.star_vel), 1)), ParticleAttrs.VELOCITY) if self.star_vel is not None and not boolean else self.star_speed[self.stars_in_halo_filter] if self.star_speed is not None else None

        if self.incl_gas:
            self.gas_pos = conditional_attribute(self.incl_gas, self.gas_pos if boolean else self.sim.particles['gas']['position'], ParticleAttrs.POSITION, self.gas_in_halo_filter)
            if self.gas_pos is not None and not boolean: self.gas_pos -= self.center_pos
            self.gas_vel = conditional_attribute(self.incl_gas, self.gas_vel if boolean else self.sim.particles['gas']['velocity'], ParticleAttrs.VELOCITY, self.gas_in_halo_filter)
            if self.gas_vel is not None and not boolean: self.gas_vel -= self.center_vel
            self.gas_id = conditional_attribute(self.incl_gas, self.gas_id if boolean else self.sim.particles['gas']['id'], ParticleAttrs.ID, self.gas_in_halo_filter)
            self.gas_mass = conditional_attribute(self.incl_gas, self.gas_mass if boolean else self.sim.particles['gas']['mass'], ParticleAttrs.MASS, self.gas_in_halo_filter)

            self.gas_mass_fraction = conditional_attribute(self.incl_gas, self.gas_mass_fraction if boolean else self.sim.particles['gas']['massfraction'], ParticleAttrs.STAR_OR_GAS_MASS_FRACTION, self.gas_in_halo_filter)
            self.gas_density = conditional_attribute(self.incl_gas, self.gas_density if boolean else self.sim.particles['gas']['density'], ParticleAttrs.GAS_DENSITY, self.gas_in_halo_filter)
            self.gas_electron_fraction = conditional_attribute(self.incl_gas, self.gas_electron_fraction if boolean else self.sim.particles['gas']['electron.fraction'], ParticleAttrs.GAS_ELECTRON_FRACTION, self.gas_in_halo_filter)
            self.gas_temperature = conditional_attribute(self.incl_gas, self.gas_temperature if boolean else self.sim.particles['gas']['temperature'], ParticleAttrs.GAS_TEMPERATURE, self.gas_in_halo_filter)
            self.gas_hydrogen_neutral_fraction = conditional_attribute(self.incl_gas, self.gas_hydrogen_neutral_fraction if boolean else self.sim.particles['gas']['hydrogen.neutral.fraction'], ParticleAttrs.GAS_HYDROGEN_NEUTRAL_FRACTION, self.gas_in_halo_filter)
            self.gas_size = conditional_attribute(self.incl_gas, self.gas_size if boolean else self.sim.particles['gas']['size'], ParticleAttrs.GAS_SIZE, self.gas_in_halo_filter)

            self.gas_distance = conditional_attribute(self.incl_gas, np.sqrt(np.sum(np.square(self.gas_pos), 1)), ParticleAttrs.DISTANCE) if self.gas_pos is not None and not boolean else self.gas_distance[self.gas_in_halo_filter] if self.gas_distance is not None else None
            self.gas_speed = conditional_attribute(self.incl_gas, np.sqrt(np.sum(np.square(self.gas_vel), 1)), ParticleAttrs.VELOCITY) if self.gas_vel is not None and not boolean else self.gas_speed[self.gas_in_halo_filter] if self.gas_speed is not None else None

        if self.incl_dark:
            self.dark_pos = conditional_attribute(self.incl_dark, self.dark_pos if boolean else self.sim.particles['dark']['position'], ParticleAttrs.POSITION, self.dark_in_halo_filter)
            if self.dark_pos is not None and not boolean: self.dark_pos -= self.center_pos
            self.dark_vel = conditional_attribute(self.incl_dark, self.dark_vel if boolean else self.sim.particles['dark']['velocity'], ParticleAttrs.VELOCITY, self.dark_in_halo_filter)
            if self.dark_vel is not None and not boolean: self.dark_vel -= self.center_vel
            self.dark_id = conditional_attribute(self.incl_dark, self.dark_id if boolean else self.sim.particles['dark']['id'], ParticleAttrs.ID, self.dark_in_halo_filter)
            self.dark_mass = conditional_attribute(self.incl_dark, self.dark_mass if boolean else self.sim.particles['dark']['mass'], ParticleAttrs.MASS, self.dark_in_halo_filter)

            self.dark_distance = conditional_attribute(self.incl_dark, np.sqrt(np.sum(np.square(self.dark_pos), 1)), ParticleAttrs.DISTANCE) if self.dark_pos is not None and not boolean else self.dark_distance[self.dark_in_halo_filter] if self.dark_distance is not None else None
            self.dark_speed = conditional_attribute(self.incl_dark, np.sqrt(np.sum(np.square(self.dark_vel), 1)), ParticleAttrs.VELOCITY) if self.dark_vel is not None and not boolean else self.dark_speed[self.dark_in_halo_filter] if self.dark_speed is not None else None

        if self.incl_dark2:
            self.dark2_pos = conditional_attribute(self.incl_dark2, self.dark2_pos if boolean else self.sim.particles['dark2']['position'], ParticleAttrs.POSITION, self.dark2_in_halo_filter)
            if self.dark2_pos is not None and not boolean: self.dark2_pos -= self.center_pos
            self.dark2_vel = conditional_attribute(self.incl_dark2, self.dark2_vel if boolean else self.sim.particles['dark2']['velocity'], ParticleAttrs.VELOCITY, self.dark2_in_halo_filter)
            if self.dark2_vel is not None and not boolean: self.dark2_vel -= self.center_vel
            self.dark2_id = conditional_attribute(self.incl_dark2, self.dark2_id if boolean else self.sim.particles['dark2']['id'], ParticleAttrs.ID, self.dark2_in_halo_filter)
            self.dark2_mass = conditional_attribute(self.incl_dark2, self.dark2_mass if boolean else self.sim.particles['dark2']['mass'], ParticleAttrs.MASS, self.dark2_in_halo_filter)

            self.dark2_distance = conditional_attribute(self.incl_dark2, np.sqrt(np.sum(np.square(self.dark2_pos), 1)), ParticleAttrs.DISTANCE) if self.dark2_pos is not None and not boolean else self.dark2_distance[self.dark2_in_halo_filter] if self.dark2_distance is not None else None
            self.dark2_speed = conditional_attribute(self.incl_dark2, np.sqrt(np.sum(np.square(self.dark2_vel), 1)), ParticleAttrs.VELOCITY) if self.dark2_vel is not None and not boolean else self.dark2_speed[self.dark2_in_halo_filter] if self.dark2_speed is not None else None

        self._stars = None
        self._particles_have_been_loaded = True

        return self

    def center_on_value(self, new_pos=None, new_vel=None):
        # Get the offset between this current center and new position, and subtract offset from each particle to center on the new position
        if new_pos is not None:
            pos_offset = new_pos - self.center_pos
            if self.star_pos is not None: self.star_pos -= pos_offset
            if self.gas_pos is not None: self.gas_pos -= pos_offset
            if self.dark_pos is not None: self.dark_pos -= pos_offset
            if self.dark2_pos is not None: self.dark2_pos -= pos_offset

            self.center_pos = new_pos
            self._stars = None

        if new_vel is not None:
            vel_offset = new_vel - self.center_vel
            if self.star_vel is not None: self.star_vel -= vel_offset
            if self.gas_vel is not None: self.gas_vel -= vel_offset
            if self.dark_vel is not None: self.dark_vel -= vel_offset
            if self.dark2_vel is not None: self.dark2_vel -= vel_offset

            self.center_vel = new_vel
            self._stars = None

        return self

    def center_on_halo(self, other_id, change_velocity=True, change_position=True):
        """
        Changes the center position (and/or velocity, if specified). 
        Recalculating particles (i.e. using calling_percentage) will still use the old center. This simply changes the reference frame of the positions and/or velocities.
        """
        # TODO: Actually I probably want to recalculate particles with new center. Cause they can just center_on *after* restricting, to use the old center for restricting

        new_pos = new_vel = None
        if change_position:
            new_pos = np.array([self.sim.ahf_data.field('Xc(6)')[other_id], 
                                self.sim.ahf_data.field('Yc(7)')[other_id], 
                                self.sim.ahf_data.field('Zc(8)')[other_id]]) / self.sim.h
            
        if change_velocity:
            new_vel = np.array([self.sim.ahf_data.field('Xc(6)')[other_id], 
                                self.sim.ahf_data.field('Yc(7)')[other_id], 
                                self.sim.ahf_data.field('Zc(8)')[other_id]]) / self.sim.h
            
        return self.center_on_value(new_pos, new_vel)


    @staticmethod
    def extract_x(array_of_vectors):
        return array_of_vectors[:, 0]
    
    @staticmethod
    def extract_y(array_of_vectors):
        return array_of_vectors[:, 1]
    
    @staticmethod
    def extract_z(array_of_vectors):
        return array_of_vectors[:, 2]
    
    @property
    def stars(self):
        if self._stars is None:
            self._stars = []
            for i in range(len(self.star_pos)):
                self._stars.append(stellarutil.Star(*self.star_pos[i], self.star_mass[i], self.star_scale_factor[i], *self.star_vel[i]))
        else:
            return self._stars

    def save_to_file(self, file_name):
        temp_sim = self.sim
        self.sim = None
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
        self.sim = temp_sim

        return self

    @staticmethod
    def load_from_file(file_name, sim=None):
        with open(file_name, 'rb') as file:
            loaded_object = pickle.load(file)
        if sim:
            loaded_object.sim = sim
        return loaded_object

    # Proposal for intuitive (optional) usage of particles without storing unecessary data:
    @property
    def stars(self):
        if self._stars is None:
            self._stars = [Star(self, i) for i in range(len(self.star_id))]  # This is dependent on the mask applied to Halo, but that's fine because self._stars is reset every time the mask is changed
                # COULD also make self._stars not a list and only generate the star objects when the individual one is accessed (through iterating or accessing at index)
                # Almost like a generator, but where you can access it by index.
                # On second thought, ignore the last two lines. They would only be more efficient if each star stayed in memory after being generated, by at that point it's too much processing power to make it worth it.

        return self._stars

    # Proposal for intuitive (optional) usage of particles without storing unecessary data:
    @property
    def gas_pos(self):
        if self._gas_pos is None:
            self._gas_pos = ...
        return self._gas_pos

    # Draft for (hopefully) intuitive bulk usage of particles (not necessary, I just thought it might be nice). On second thought, though, this might not really be useful/helpful.
    def get_gas_values_from_function(self, function):
        output_array = np.empty(len(self.gas))
        for i, gas_particle in enumerate(self.gas):
            output_array[i] = function(gas_particle)

# Usage:
def get_2dr(particle):
    return (particle.x**2 + particle.y**2)**0.5

gas_2dr = halo.get_gas_values_from_function(get_2dr)
# Or:
gas_2dr = halo.get_gas_values_from_function(lambda particle: (particle.x**2 + particle.y**2)**0.5)

import abc


star_list = halo.stars
for star in star_list:
    star.pos

class Particle(abc.ABC):
    # POS_ATTR = None

    def __init__(self):
        self.parent_halo = ...
        self.index_in_halo = ...

    """
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
    """
    
class Gas(Particle):
    POS_ATTR = "gas_pos"

    DENSITY_ATTR = "gas_density"

    @property
    def pos(self):
        return self.parent_halo.gas_pos[self.index_in_halo]

    # @property
    # def density(self):
    #     return self.parent_halo.__getattribute__(self.DENSITY_ATTR)[self.index_in_halo]  # Don't necessarily need to use __getattribute__, since this isn't gonna have a child class


# Example of need for ParticleGroup:

sim1 = stellarutil.Simulation( ... )
sim2 = stellarutil.Simulation( ... )

star_ids = Halo(sim1, 1, incl_stars=(ParticleAttrs.ID,), restrict_percentage=100).star_id
old_stars = ParticleGroup(sim2).restrict_ids(star_ids=star_ids)  # Using a halo class here would be redunant. What halo would we put in?


# halo = Halo(sim, 1, incl_stars=False, incl_dark=(ParticleAttrs.POSITION, ParticleAttrs.VELOCITY))

"""
# def get_value_list()

# a = Halo(sim, 0, (ParticleAttrs.POSITION, ParticleAttrs.VELOCITY))

# a.restrict_percentage(100)
# a.restrict_ids([0, 1, 2, 5], boolean=True)

# def get_col_from_particle #like can let you color it red if ID is in array
"""