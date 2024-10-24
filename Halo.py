import numpy as np, pickle, abc, Particle

# TODO: Make attributes return None if boolean mask is being applied

def _get_getter(var_name, generator, set_name_to_add_var_name=None):
    """Returns a getter that returns the attribute with the name given by var_name. 
    Before the attribute is accessed, it is not initialized and is None/inaccessible. When it is accessed by the getter for the first time, its initial value is created 
    by the generator function. This allows for values to not be initialized until they are accessed for the first time, reducing processing power by not initializing
    unused attributes.

    Example: 
    ```
    class Foo:
        bar_initial_value = 1

        bar = _get_getter("_bar", lambda self: self.bar_initial_value)

    foo = Foo()
    # foo.bar has not been initialized yet
    print(foo.bar)  # Returns 1
    # foo.bar has now been initialized and any future calls will just return the previously set foo.bar
    ```

    :param var_name: The name of the variable to store the attribute in
    :param generator: A function that returns the default/initial value for the attribute. self (the object the method is called on) is passed in as the only parameter
    :param set_name_to_add_var_name: Optional. The variable name of a set in the affiliated object that var_name will be automatically added to
    :return: A getter that, when called returns the attribute. This can be set to any class variable to affiliate it with a class
    """
    @property
    def getter(self):
        if getattr(self, var_name, None) is None:
            if set_name_to_add_var_name is not None:
                set_to_add_var_name = getattr(self, set_name_to_add_var_name, None)
                if set_to_add_var_name is None:
                    setattr(self, set_name_to_add_var_name, {var_name})
                else:
                    set_to_add_var_name.add(var_name)

            new_val = generator(self)
            setattr(self, var_name, new_val)
            return new_val
        else:
            return self.__getattribute__(var_name)
        
    return getter

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

class Species:
    all = "all"
    star = "star"
    gas = "gas"
    dark = "dark"
    dark2 = "dark2"
    blackhole = "blackhole"

class Boolean:
    AND = "and"
    NAND = "nand"
    OR = "or"
    NOR = "nor"
    XOR = "xor"
    XNOR = "xnor"

    FUNCTION_MAP = {
        "and": lambda a, b: a & b,
        "&": lambda a, b: a & b,
        "nand": lambda a, b: ~(a & b),
        "or": lambda a, b: a | b,
        "|": lambda a, b: a | b,
        "nor": lambda a, b: ~(a | b),
        "xor": lambda a, b: (a & ~b) | (~a & b),
        "xnor": lambda a, b: (a | ~b) & (~a | b)
    }

    @classmethod
    def get_boolean_function(cls, boolean):
        return cls.FUNCTION_MAP[boolean.lower()] if isinstance(boolean, str) and boolean in cls.FUNCTION_MAP else boolean


class ParticleGroup:
    #TODO: Comment code and add docstrings

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

    def __init__(self, sim, species: list[str] | tuple[str] = ("all",)):
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

        # The position that all particle positions are relative to. The absolute coordinates of the Zero in the active reference frame. This can be changed.
        self.center_pos = np.array([0., 0., 0.])
        # See center_pos, but for velocity
        self.center_vel = np.array([0., 0., 0.])
        
        self.sim = sim

        self.incl_stars = Species.star in sim.particles and (Species.all in species or Species.star in species)
        self.incl_gas = Species.gas in sim.particles and (Species.all in species or Species.gas in species)
        self.incl_dark = Species.dark in sim.particles and (Species.all in species or Species.dark in species)
        self.incl_dark2 = Species.dark2 in sim.particles and (Species.all in species or Species.dark2 in species)

        self._star_vars = set()
        self._gas_vars = set()
        self._dark_vars = set()
        self._dark2_vars = set()

    def reset_all_particle_attributes(self):
        for var_name in self._star_vars | self._gas_vars | self._dark_vars | self._dark2_vars:
            self.__setattr__(var_name, None)
        
        return self

    # Filters (arrays of booleans that filter which particles are being stored)
    stars_in_halo_filter = _get_getter("_stars_in_halo_filter", lambda self: self.generate_constant_filter_getter(True)(self, Species.star, None))
    gas_in_halo_filter = _get_getter("_gas_in_halo_filter", lambda self: self.generate_constant_filter_getter(True)(self, Species.gas, None))
    dark_in_halo_filter = _get_getter("_dark_in_halo_filter", lambda self: self.generate_constant_filter_getter(True)(self, Species.dark, None))
    dark2_in_halo_filter = _get_getter("_dark2_in_halo_filter", lambda self: self.generate_constant_filter_getter(True)(self, Species.dark2, None))

    # Function to apply a filter getter to all included particles
    def apply_filter(self, filter_getter, boolean=None):
        if isinstance(boolean, str) and boolean.lower() in ["and", "&", Boolean.AND]:
            # Apply the relative filter to all loaded attributes
            particles_to_filter = []
            if self.incl_stars: particles_to_filter.append((self._star_vars, filter_getter(self, Species.star, self.star_pos)))
            if self.incl_gas: particles_to_filter.append((self._gas_vars, filter_getter(self, Species.gas, self.gas_pos)))
            if self.incl_dark: particles_to_filter.append((self._dark_vars, filter_getter(self, Species.dark, self.dark_pos)))
            if self.incl_dark2: particles_to_filter.append((self._dark2_vars, filter_getter(self, Species.dark2, self.dark2_pos)))

            for var_set, boolean_filter in particles_to_filter:
                for var_name in var_set:
                    attribute = self.__getattribute__(var_name)
                    if attribute is not None:
                        self.__setattr__(var_name, attribute[boolean_filter])
        else:
            if getattr(self, "sim", None) is None:
                raise AttributeError("Must use boolean='and' or '&' or Boolean.AND if sim is undefined, since filtered-out particles are no longer accessible")
        
            if boolean is None:
                if self.incl_stars: self._stars_in_halo_filter = filter_getter(self, Species.star, None)
                if self.incl_gas: self._gas_in_halo_filter = filter_getter(self, Species.gas, None)
                if self.incl_dark: self._dark_in_halo_filter = filter_getter(self, Species.dark, None)
                if self.incl_dark2: self._dark2_in_halo_filter = filter_getter(self, Species.dark2, None)
            else:
                boolean = Boolean.get_boolean_function(boolean)
                if self.incl_stars: self._stars_in_halo_filter = boolean(self._stars_in_halo_filter, filter_getter(self, Species.star, None))
                if self.incl_gas: self._gas_in_halo_filter = boolean(self._gas_in_halo_filter, filter_getter(self, Species.gas, None))
                if self.incl_dark: self._dark_in_halo_filter = boolean(self._dark_in_halo_filter, filter_getter(self, Species.dark, None))
                if self.incl_dark2: self._dark2_in_halo_filter = boolean(self._dark2_in_halo_filter, filter_getter(self, Species.dark2, None))

            self.reset_all_particle_attributes()
        
        return self
    
    # Functions that generate return filter getters (filter getters are to be passed into apply_filter)
    def generate_constant_filter_getter(self, all_true_or_false):
        return lambda self, species, particle_positions=None: np.full(
            len(self.sim.particles[species]['position'] if particle_positions is None else particle_positions), 
            all_true_or_false
        )

    def generate_radius_filter_getter(self, radius):
        def filter_getter(self, species, particle_positions=None):
            all_rel_particle_pos = (self.sim.particles[species]['position'] - self.center_pos) \
                if particle_positions is None else particle_positions
                
            particle_pos_sum_of_squares = np.sum(np.square(all_rel_particle_pos), 1)
            return particle_pos_sum_of_squares < radius**2
        
        if radius is None:
            return self.generate_constant_filter_getter(True)
        else:
            return filter_getter

    def generate_slice_filter_getter(self, face = 'xy', proj_distance = 1, thickness = 1):
        face = face.lower()

        def filter_getter(self, species, particle_positions=None):
            all_rel_particle_pos = (self.sim.particles[species]['position'] - self.center_pos) \
                if particle_positions is None else particle_positions

            if face == 'xy' or face == 'yx': 
                square_distance_to_axis = np.sum(np.square(all_rel_particle_pos[:, [0, 1]]), 1)
                distance_to_plane = np.abs(all_rel_particle_pos[:, 2])
            elif face == 'xz' or face == 'zx': 
                square_distance_to_axis = np.sum(np.square(all_rel_particle_pos[:, [0, 2]]), 1)
                distance_to_plane = np.abs(all_rel_particle_pos[:, 1])
            elif face == 'yz' or face == 'zy': 
                square_distance_to_axis = np.sum(np.square(all_rel_particle_pos[:, [1, 2]]), 1)
                distance_to_plane = np.abs(all_rel_particle_pos[:, 0])

            if proj_distance is not None and thickness is not None:
                return (square_distance_to_axis < proj_distance**2) & (distance_to_plane < thickness)
            elif proj_distance is None:
                return (distance_to_plane < thickness)
            elif thickness is None:
                return (square_distance_to_axis < proj_distance**2)
            else:
                return self.generate_constant_filter_getter(True)
            
        
        return filter_getter
    
    def generate_ids_filter_getter(self, star_ids=[], gas_ids=[], dark_ids=[], dark2_ids=[]):
        id_map = {Species.star: star_ids, Species.gas: gas_ids, Species.dark: dark_ids, Species.dark2: dark2_ids}
        def filter_getter(species, particle_positions=None):
            ids = id_map[species]
            if ids == []:
                return self.generate_constant_filter_getter(False)
            elif ids is None:
                return self.generate_constant_filter_getter(True)
            
            all_particle_ids = self.sim.particles[species]['id'] if particle_positions is None else particle_positions
            return np.isin(all_particle_ids, ids)
        
            # Could sort stars by ID, then index it by star_ids (probably not, since IDs have gaps)
            # NEVERMIND np.isin IS AMAZING

        return filter_getter

    # Functions that quickly generate a filter getter and pass it into apply filter
    def reset_restriction(self, boolean=None):
        return self.apply_filter(self.generate_constant_filter_getter(True), boolean)

    def restrict_radius(self, radius: float | int | None, boolean=None):
        return self.apply_filter(self.generate_radius_filter_getter(radius), boolean)

    def restrict_slice(self, face = 'xy', proj_distance=1, thickness=1, boolean=None):
        return self.apply_filter(self.generate_slice_filter_getter(face, proj_distance, thickness), boolean)

    def restrict_ids(self, star_ids=[], gas_ids=[], dark_ids=[], dark2_ids=[], boolean=None):
        return self.apply_filter(self.generate_ids_filter_getter(star_ids, gas_ids, dark_ids, dark2_ids), boolean)
   
    # Star attributes
    star_pos = _get_getter("_star_pos", lambda self: self.sim.particles[Species.star]['position'][self.stars_in_halo_filter] - self.center_pos, "_star_vars")
    star_x, star_y, star_z = [property(lambda self: self.star_pos[:, i]) for i in range(3)]
    star_vel = _get_getter("_star_vel", lambda self: self.sim.particles[Species.star]['velocity'][self.stars_in_halo_filter] - self.center_vel, "_star_vars")
    star_vx, star_vy, star_vz = [property(lambda self: self.star_vel[:, i]) for i in range(3)]
    star_distance = _get_getter("_star_distance", lambda self: np.sqrt(np.sum(np.square(self.star_pos), 1)), "_star_vars")
    star_r2d = _get_getter("_star_r2d", lambda self: np.sqrt(np.sum(np.square(self.star_pos[:, [0, 1]]), 1)), "_star_vars")
    star_speed = _get_getter("_star_speed", lambda self: np.sqrt(np.sum(np.square(self.star_vel), 1)), "_star_vars")
    star_id = _get_getter("_star_id", lambda self: self.sim.particles[Species.star]['id'][self.stars_in_halo_filter], "_star_vars")
    star_id_child = _get_getter("_star_id_child", lambda self: self.sim.particles[Species.star]['id.child'][self.stars_in_halo_filter], "_star_vars")
    star_id_generation = _get_getter("_star_id_generation", lambda self: self.sim.particles[Species.star]['id.generation'][self.stars_in_halo_filter], "_star_vars")
    star_mass = _get_getter("_star_mass", lambda self: self.sim.particles[Species.star]['mass'][self.stars_in_halo_filter], "_star_vars")
    # TODO: Add id.child and id.generation to all particle attributes
    
    star_scale_factor = _get_getter("_star_scale_factor", lambda self: self.sim.particles[Species.star]['form.scalefactor'][self.stars_in_halo_filter], "_star_vars")
    star_mass_fraction = _get_getter("_star_mass_fraction", lambda self: self.sim.particles[Species.star]['massfraction'][self.stars_in_halo_filter], "_star_vars")
    # TODO: What is star scale_factor? Why isn't 'age' in particles['star']? Any correlation, since scale_factor isn't in other implementation but age is.

    # Gas attributes
    gas_pos = _get_getter("_gas_pos", lambda self: self.sim.particles[Species.gas]['position'][self.gas_in_halo_filter] - self.center_pos, "_gas_vars")
    gas_x, gas_y, gas_z = [property(lambda self: self.gas_pos[:, i]) for i in range(3)]
    gas_vel = _get_getter("_gas_vel", lambda self: self.sim.particles[Species.gas]['velocity'][self.gas_in_halo_filter] - self.center_vel, "_gas_vars")
    gas_vx, gas_vy, gas_vz = [property(lambda self: self.gas_vel[:, i]) for i in range(3)]
    gas_distance = _get_getter("_gas_distance", lambda self: np.sqrt(np.sum(np.square(self.gas_pos), 1)), "_gas_vars")
    gas_r2d = _get_getter("_gas_r2d", lambda self: np.sqrt(np.sum(np.square(self.gas_pos[:, [0, 1]]), 1)), "_gas_vars")
    gas_speed = _get_getter("_gas_speed", lambda self: np.sqrt(np.sum(np.square(self.gas_vel), 1)), "_gas_vars")
    gas_id = _get_getter("_gas_id", lambda self: self.sim.particles[Species.gas]['id'][self.gas_in_halo_filter], "_gas_vars")
    gas_id_child = _get_getter("_gas_id_child", lambda self: self.sim.particles[Species.gas]['id.child'][self.gas_in_halo_filter], "_gas_vars")
    gas_id_generation = _get_getter("_gas_id_generation", lambda self: self.sim.particles[Species.gas]['id.generation'][self.gas_in_halo_filter], "_gas_vars")
    gas_mass = _get_getter("_gas_mass", lambda self: self.sim.particles[Species.gas]['mass'][self.gas_in_halo_filter], "_gas_vars")
    
    gas_mass_fraction = _get_getter("_gas_mass_fraction", lambda self: self.sim.particles[Species.gas]['massfraction'][self.gas_in_halo_filter], "_gas_vars")
    gas_density = _get_getter("_gas_density", lambda self: self.sim.particles[Species.gas]['density'][self.gas_in_halo_filter], "_gas_vars")
    gas_electron_fraction = _get_getter("_gas_electron_fraction", lambda self: self.sim.particles[Species.gas]['electron.fraction'][self.gas_in_halo_filter], "_gas_vars")
    gas_temperature = _get_getter("gas_temperature", lambda self: self.sim.particles[Species.gas]['temperature'][self.gas_in_halo_filter], "_gas_vars")
    gas_hydrogen_neutral_fraction = _get_getter("_gas_hydrogen_neutral_fraction", lambda self: self.sim.particles[Species.gas]['hydrogen.neutral.fraction'][self.gas_in_halo_filter], "_gas_vars")
    gas_size = _get_getter("_gas_size", lambda self: self.sim.particles[Species.gas]['size'][self.gas_in_halo_filter], "_gas_vars")
    gas_sfr = _get_getter("_gas_sfr", lambda self: self.sim.particles[Species.gas]['sfr'][self.gas_in_halo_filter], "_gas_vars")

    # Dark attributes
    dark_pos = _get_getter("_dark_pos", lambda self: self.sim.particles[Species.dark]['position'][self.dark_in_halo_filter] - self.center_pos, "_dark_vars")
    dark_x, dark_y, dark_z = [property(lambda self: self.dark_pos[:, i]) for i in range(3)]
    dark_vel = _get_getter("_dark_vel", lambda self: self.sim.particles[Species.dark]['velocity'][self.dark_in_halo_filter] - self.center_vel, "_dark_vars")
    dark_vx, dark_vy, dark_vz = [property(lambda self: self.dark_vel[:, i]) for i in range(3)]
    dark_distance = _get_getter("_dark_distance", lambda self: np.sqrt(np.sum(np.square(self.dark_pos), 1)), "_dark_vars")
    dark_r2d = _get_getter("_dark_r2d", lambda self: np.sqrt(np.sum(np.square(self.dark_pos[:, [0, 1]]), 1)), "_dark_vars")
    dark_speed = _get_getter("_dark_speed", lambda self: np.sqrt(np.sum(np.square(self.dark_vel), 1)), "_dark_vars")
    dark_id = _get_getter("_dark_id", lambda self: self.sim.particles[Species.dark]['id'][self.dark_in_halo_filter], "_dark_vars")
    dark_id_child = _get_getter("_dark_id_child", lambda self: self.sim.particles[Species.dark]['id.child'][self.dark_in_halo_filter], "_dark_vars")
    dark_id_generation = _get_getter("_dark_id_generation", lambda self: self.sim.particles[Species.dark]['id.generation'][self.dark_in_halo_filter], "_dark_vars")
    dark_mass = _get_getter("_dark_mass", lambda self: self.sim.particles[Species.dark]['mass'][self.dark_in_halo_filter], "_dark_vars")

    # Dark2 attributes
    dark2_pos = _get_getter("_dark2_pos", lambda self: self.sim.particles[Species.dark2]['position'][self.dark2_in_halo_filter] - self.center_pos, "_dark2_vars")
    dark2_x, dark2_y, dark2_z = [property(lambda self: self.dark2_pos[:, i]) for i in range(3)]
    dark2_vel = _get_getter("_dark2_vel", lambda self: self.sim.particles[Species.dark2]['velocity'][self.dark2_in_halo_filter] - self.center_vel, "_dark2_vars")
    dark2_vx, dark2_vy, dark2_vz = [property(lambda self: self.dark2_vel[:, i]) for i in range(3)]
    dark2_distance = _get_getter("_dark2_distance", lambda self: np.sqrt(np.sum(np.square(self.dark2_pos), 1)), "_dark2_vars")
    dark2_r2d = _get_getter("_dark2_r2d", lambda self: np.sqrt(np.sum(np.square(self.dark2_pos[:, [0, 1]]), 1)), "_dark2_vars")
    dark2_speed = _get_getter("_dark2_speed", lambda self: np.sqrt(np.sum(np.square(self.dark2_vel), 1)), "_dark2_vars")
    dark2_id = _get_getter("_dark2_id", lambda self: self.sim.particles[Species.dark2]['id'][self.dark2_in_halo_filter], "_dark2_vars")
    dark2_id_child = _get_getter("_dark2_id_child", lambda self: self.sim.particles[Species.dark2]['id.child'][self.dark2_in_halo_filter], "_dark2_vars")
    dark2_id_generation = _get_getter("_dark2_id_generation", lambda self: self.sim.particles[Species.dark2]['id.generation'][self.dark2_in_halo_filter], "_dark2_vars")
    dark2_mass = _get_getter("_dark2_mass", lambda self: self.sim.particles[Species.dark2]['mass'][self.dark2_in_halo_filter], "_dark2_vars")

    # Lists of individual particles
    stars = _get_getter("_stars", lambda self: [Particle.Star(self, i) for i in range(len(self.star_id))])  # This is dependent on the mask applied to Halo, but that's fine because self._stars is reset every time the mask is changed
    gas = _get_getter("_gas", lambda self: [Particle.Gas(self, i) for i in range(len(self.gas_id))])
    dark = _get_getter("_dark", lambda self: [Particle.ark(self, i) for i in range(len(self.dark_id))])
    dark2 = _get_getter("_dark2", lambda self: [Particle.Dark2(self, i) for i in range(len(self.dark2_id))])

    def center_on_value(self, new_pos=None, new_vel=None):
        # Get the offset between this current center and new position, and subtract offset from each particle to center on the new position
        if new_pos is not None:
            pos_offset = new_pos - self.center_pos
            if getattr(self, "_star_pos", None) is not None: self._star_pos -= pos_offset
            if getattr(self, "_gas_pos", None) is not None: self._gas_pos -= pos_offset
            if getattr(self, "_dark_pos", None) is not None: self._dark_pos -= pos_offset
            if getattr(self, "_dark2_pos", None) is not None: self._dark2_pos -= pos_offset

            self.center_pos = new_pos
            self._stars = None

        if new_vel is not None:
            vel_offset = new_vel - self.center_vel
            if getattr(self, "_star_vel", None) is not None: self._star_vel -= vel_offset
            if getattr(self, "_gas_vel", None) is not None: self._gas_vel -= vel_offset
            if getattr(self, "_dark_vel", None) is not None: self._dark_vel -= vel_offset
            if getattr(self, "_dark2_vel", None) is not None: self._dark2_vel -= vel_offset

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

class Halo(ParticleGroup):

    def __init__(self, 
                 sim, 
                 halo_id: int, 
                 restrict_percentage: float | int | None = 100,
                 species: list[str] | tuple[str] = ("all",)):

        self.halo_id = halo_id
        self.halo_radius = sim.get_field('12')[halo_id]

        self.restricted_percentage = restrict_percentage

        super().__init__(sim, species)

        self.center_on_halo(halo_id)

        # The absolute coordinates of the center of this halo. This does not change.
        self.this_halo_center_pos = np.array([self.center_pos[0], self.center_pos[1], self.center_pos[2]])
        # See this_halo_center_pos, but for velocity.
        self.this_halo_center_vel = np.array([self.center_vel[0], self.center_vel[1], self.center_vel[2]])

    host_halo_id = _get_getter("_host_halo_id", lambda self: self.sim.ahf_data.field('hostHalo(2)')[self.halo_id])
    child_halo_ids = _get_getter("_child_halo_ids", lambda self: np.where(self.sim.ahf_data.field('hostHalo(2)') == self.halo_id))
    mass = _get_getter("_mass", lambda self: self.sim.get_field('4')[self.halo_id])
    r_max = _get_getter("_r_max", lambda self: self.sim.ahf_data.field('Rmax(13)')[self.halo_id] / self.sim.h)
    v_max = _get_getter("_v_max", lambda self: self.sim.ahf_data.field('Vmax(17)')[self.halo_id])
    v_esc = _get_getter("_v_esc", lambda self: self.sim.ahf_data.field('v_esc(18)')[self.halo_id])
    num_gas = _get_getter("_num_gas", lambda self: self.sim.ahf_data.field('n_gas(44)')[self.halo_id])
    gas_mass = _get_getter("_gas_mass", lambda self: self.sim.ahf_data.field('M_gas(45)')[self.halo_id])
    num_stars = _get_getter("_num_stars", lambda self: self.sim.ahf_data.field('n_star(64)')[self.halo_id])
    star_mass = _get_getter("_star_mass", lambda self: self.sim.ahf_data.field('M_star(65)')[self.halo_id])
    num_particles = _get_getter("_num_particles", lambda self: self.sim.ahf_data.field('npart(5)')[self.halo_id])

    stars_in_halo_filter = _get_getter("_stars_in_halo_filter", lambda self: self.generate_percentage_filter_getter(self.restricted_percentage)(self, Species.star, None))
    gas_in_halo_filter = _get_getter("_gas_in_halo_filter", lambda self: self.generate_percentage_filter_getter(self.restricted_percentage)(self, Species.gas, None))
    dark_in_halo_filter = _get_getter("_dark_in_halo_filter", lambda self: self.generate_percentage_filter_getter(self.restricted_percentage)(self, Species.dark, None))
    dark2_in_halo_filter = _get_getter("_dark2_in_halo_filter", lambda self: self.generate_percentage_filter_getter(self.restricted_percentage)(self, Species.dark2, None))

    def generate_percentage_filter_getter(self, percentage):
        if percentage is None:
            return self.generate_constant_filter_getter(True)
        else:
            return self.generate_radius_filter_getter(self.halo_radius * (percentage / 100))
        
    def restrict_percentage(self, percentage: float | int | None, boolean=None):
        return self.apply_filter(self.generate_percentage_filter_getter(percentage), boolean)
    
    def recenter_on_this_halo(self):
        self.center_on_halo(self.halo_id)

    def get_host_halo(self, **halo_kwargs):
        if self.host_halo_id == -1:
            return None
        else:
            return Halo(self.sim, self.host_halo_id, **halo_kwargs)
        
    def get_child_halos(self, **halo_kwargs):
        return [Halo(self.sim, child_id, **halo_kwargs) for child_id in self.child_halo_ids]
