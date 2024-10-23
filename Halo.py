import numpy as np, pickle, abc

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

class Halo:
    #TODO: Make parent ParticleGroup class, make halo class a child w/ extra halo-specific functionality
    #      Make restrict_radius that does the same as restrict_percentage, just with an absolute radius. Then, restrict_percentage calls it
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
                 species: list[str] | tuple[str] = ("all",),
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

        self.incl_stars = "star" in sim.particles and ("all" in species or "star" in species)
        self.incl_gas = "gas" in sim.particles and ("all" in species or "gas" in species)
        self.incl_dark = "dark" in sim.particles and ("all" in species or "dark" in species)
        self.incl_dark2 = "dark2" in sim.particles and ("all" in species or "dark2" in species)

        self._star_vars = set()
        self._gas_vars = set()
        self._dark_vars = set()
        self._dark2_vars = set()

        # self._touched = False

    # def touch(self):
    #     self._touched = True

    #     self._hostID = self.sim.ahf_data.field('hostHalo(2)')[self.halo_id]
    #     self._mass = self.sim.get_field('4')[self.halo_id]
    #     self._r_max = self.sim.ahf_data.field('Rmax(13)')[self.halo_id] / self.sim.h
    #     self._v_max = self.sim.ahf_data.field('Vmax(17)')[self.halo_id]
    #     self._v_esc = self.sim.ahf_data.field('v_esc(18)')[self.halo_id]
    #     self._num_gas = self.sim.ahf_data.field('n_gas(44)')[self.halo_id]
    #     self._gas_mass = self.sim.ahf_data.field('M_gas(45)')[self.halo_id]
    #     self._num_stars = self.sim.ahf_data.field('n_star(64)')[self.halo_id]
    #     self._star_mass = self.sim.ahf_data.field('M_star(65)')[self.halo_id]
    #     self._num_particles = self.sim.ahf_data.field('npart(5)')[self.halo_id]
    
    # def __getattribute__(self, name):
    #     if not self._touched:
    #         self.touch()
    #     return super().__getattribute__(name)

    # Do we want to only generate all this upon first attribute get? (in a "touch" method, per se, where the object is just a shell until it's touched)
    hostID = _get_getter("_hostID", lambda self: self.sim.ahf_data.field('hostHalo(2)')[self.halo_id])
    mass = _get_getter("_mass", lambda self: self.sim.get_field('4')[self.halo_id])
    r_max = _get_getter("_r_max", lambda self: self.sim.ahf_data.field('Rmax(13)')[self.halo_id] / self.sim.h)
    v_max = _get_getter("_v_max", lambda self: self.sim.ahf_data.field('Vmax(17)')[self.halo_id])
    v_esc = _get_getter("_v_esc", lambda self: self.sim.ahf_data.field('v_esc(18)')[self.halo_id])
    num_gas = _get_getter("_num_gas", lambda self: self.sim.ahf_data.field('n_gas(44)')[self.halo_id])
    gas_mass = _get_getter("_gas_mass", lambda self: self.sim.ahf_data.field('M_gas(45)')[self.halo_id])
    num_stars = _get_getter("_num_stars", lambda self: self.sim.ahf_data.field('n_star(64)')[self.halo_id])
    star_mass = _get_getter("_star_mass", lambda self: self.sim.ahf_data.field('M_star(65)')[self.halo_id])
    num_particles = _get_getter("_num_particles", lambda self: self.sim.ahf_data.field('npart(5)')[self.halo_id])

    stars_in_halo_filter = _get_getter("_stars_in_halo_filter", lambda self: np.full(len(self.sim.particles['star']['position']), True))
    gas_in_halo_filter = _get_getter("_gas_in_halo_filter", lambda self: np.full(len(self.sim.particles['gas']['position']), True))
    dark_in_halo_filter = _get_getter("_dark_in_halo_filter", lambda self: np.full(len(self.sim.particles['dark']['position']), True))
    dark2_in_halo_filter = _get_getter("_dark2_in_halo_filter", lambda self: np.full(len(self.sim.particles['dark2']['position']), True))

    def reset_all_particle_attributes(self):
        for var_name in self._star_vars + self._gas_vars + self._dark_vars + self._dark2_vars:
            self.__setattr__(var_name, None)
        
        return self

    def restrict_percentage(self, percentage: float | int | None, boolean=False):
        if percentage is None:
            if boolean:
                return self
            else:
                return self.reset_restriction()

        # if boolean and not self._particles_have_been_loaded:
        #     boolean = False

        self.restricted_halo_radius = self.halo_radius * (percentage / 100)

        def get_particle_in_halo_filter(particles):
            all_rel_particle_pos = (self.sim.particles[particles]['position'] - self.this_halo_center_pos) \
                if isinstance(particles, str) else particles
                
            particle_pos_sum_of_squares = np.sum(np.square(all_rel_particle_pos), 1)
            return particle_pos_sum_of_squares < self.restricted_halo_radius**2

        if self.incl_stars: self._stars_in_halo_filter = get_particle_in_halo_filter(self.star_pos if boolean else 'star')
        if self.incl_gas: self._gas_in_halo_filter = get_particle_in_halo_filter(self.gas_pos if boolean else 'gas')
        if self.incl_dark: self._dark_in_halo_filter = get_particle_in_halo_filter(self.dark_pos if boolean else 'dark')
        if self.incl_dark2: self._dark2_in_halo_filter = get_particle_in_halo_filter(self.dark2_pos if boolean else 'dark2')

        self.reset_all_particle_attributes()
        return self

    def restrict_slice(self, face = 'xy', proj_distance = 1, thickness = 1, boolean=False):
        face = face.lower()
        if boolean and not self._particles_have_been_loaded:
            boolean = False

        def get_particle_in_slice_filter(particles):
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
            
        if self.incl_stars: self._stars_in_halo_filter = get_particle_in_slice_filter(self.star_pos if boolean else 'star')
        if self.incl_gas: self._gas_in_halo_filter = get_particle_in_slice_filter(self.gas_pos if boolean else 'gas')
        if self.incl_dark: self._dark_in_halo_filter = get_particle_in_slice_filter(self.dark_pos if boolean else 'dark')
        if self.incl_dark2: self._dark2_in_halo_filter = get_particle_in_slice_filter(self.dark2_pos if boolean else 'dark2')

        self.reset_all_particle_attributes()
        return self

    def reset_restriction(self):
        if self.incl_stars: self._stars_in_halo_filter = np.full(len(self.sim.particles['star']['position']), True)
        if self.incl_gas: self._gas_in_halo_filter = np.full(len(self.sim.particles['gas']['position']), True)
        if self.incl_dark: self._dark_in_halo_filter = np.full(len(self.sim.particles['dark']['position']), True)
        if self.incl_dark2: self._dark2_in_halo_filter = np.full(len(self.sim.particles['dark2']['position']), True)

        self.reset_all_particle_attributes()
        return self

    def restrict_ids(self, star_ids=None, gas_ids=None, dark_ids=None, dark2_ids=None, boolean=False):
        if boolean and not self._particles_have_been_loaded:
            boolean = False

        def get_particles_with_id_filter(particles, ids):
            if ids is None:
                return None
            
            all_particle_ids = self.sim.particles[particles]['id'] if isinstance(particles, str) else particles
            return np.isin(all_particle_ids, ids)
        
        if self.incl_stars: self.stars_in_halo_filter = get_particles_with_id_filter(self.star_id if boolean else "star", star_ids)
        if self.incl_gas: self.gas_in_halo_filter = get_particles_with_id_filter(self.gas_id if boolean else "gas", gas_ids)
        if self.incl_dark: self.dark_in_halo_filter = get_particles_with_id_filter(self.dark_id if boolean else "dark", dark_ids)
        if self.incl_dark2: self.dark2_in_halo_filter = get_particles_with_id_filter(self.dark2_id if boolean else "dark2", dark2_ids)
            # Could sort stars by ID, then index it by star_ids (probably not, since IDs have gaps)
            # NEVERMIND np.isin IS AMAZING

        self.reset_all_particle_attributes()
        return self
        
    # Star attributes
    star_pos = _get_getter("_star_pos", lambda self: self.sim.particles['star']['position'][self.stars_in_halo_filter] - self.center_pos, "_star_vars")
    star_x, star_y, star_z = [property(lambda self: self.star_pos[:, i] for i in range(3))]
    star_vel = _get_getter("_star_vel", lambda self: self.sim.particles['star']['velocity'][self.stars_in_halo_filter] - self.center_vel, "_star_vars")
    star_vx, star_vy, star_vz = [property(lambda self: self.star_vel[:, i] for i in range(3))]
    star_distance = _get_getter("_star_distance", lambda self: np.sqrt(np.sum(np.square(self.star_pos), 1)), "_star_vars")
    star_r2d = _get_getter("_star_r2d", lambda self: np.sqrt(np.sum(np.square(self.star_pos[:, [0, 1]]), 1)), "_star_vars")
    star_speed = _get_getter("_star_speed", lambda self: np.sqrt(np.sum(np.square(self.star_vel), 1)), "_star_vars")
    star_id = _get_getter("_star_id", lambda self: self.sim.particles['star']['id'][self.stars_in_halo_filter], "_star_vars")
    star_mass = _get_getter("_star_mass", lambda self: self.sim.particles['star']['mass'][self.stars_in_halo_filter], "_star_vars")
    star_scale_factor = _get_getter("_star_scale_factor", lambda self: self.sim.particles['star']['form.scalefactor'][self.stars_in_halo_filter], "_star_vars")
    star_mass_fraction = _get_getter("_star_mass_fraction", lambda self: self.sim.particles['star']['massfraction'][self.stars_in_halo_filter], "_star_vars")

    # Gas attributes
    gas_pos = _get_getter("_gas_pos", lambda self: self.sim.particles['gas']['position'][self.gas_in_halo_filter] - self.center_pos, "_gas_vars")
    gas_x, gas_y, gas_z = [property(lambda self: self.gas_pos[:, i] for i in range(3))]
    gas_vel = _get_getter("_gas_vel", lambda self: self.sim.particles['gas']['velocity'][self.gas_in_halo_filter] - self.center_vel, "_gas_vars")
    gas_vx, gas_vy, gas_vz = [property(lambda self: self.gas_vel[:, i] for i in range(3))]
    gas_distance = _get_getter("_gas_distance", lambda self: np.sqrt(np.sum(np.square(self.gas_pos), 1)), "_gas_vars")
    gas_r2d = _get_getter("_gas_r2d", lambda self: np.sqrt(np.sum(np.square(self.gas_pos[:, [0, 1]]), 1)), "_gas_vars")
    gas_speed = _get_getter("_gas_speed", lambda self: np.sqrt(np.sum(np.square(self.gas_vel), 1)), "_gas_vars")
    gas_id = _get_getter("_gas_id", lambda self: self.sim.particles['gas']['id'][self.gas_in_halo_filter], "_gas_vars")

    gas_mass = _get_getter("_gas_mass", lambda self: self.sim.particles['gas']['mass'][self.gas_in_halo_filter], "_gas_vars")
    gas_mass_fraction = _get_getter("_gas_mass_fraction", lambda self: self.sim.particles['gas']['massfraction'][self.gas_in_halo_filter], "_gas_vars")
    gas_density = _get_getter("_gas_density", lambda self: self.sim.particles['gas']['density'][self.gas_in_halo_filter], "_gas_vars")
    gas_electron_fraction = _get_getter("_gas_electron_fraction", lambda self: self.sim.particles['gas']['electron.fraction'][self.gas_in_halo_filter], "_gas_vars")
    gas_temperature = _get_getter("gas_temperature", lambda self: self.sim.particles['gas']['temperature'][self.gas_in_halo_filter], "_gas_vars")
    gas_hydrogen_neutral_fraction = _get_getter("_gas_hydrogen_neutral_fraction", lambda self: self.sim.particles['gas']['hydrogen.neutral.fraction'][self.gas_in_halo_filter], "_gas_vars")
    gas_size = _get_getter("_gas_size", lambda self: self.sim.particles['gas']['size'][self.gas_in_halo_filter], "_gas_vars")

    # Dark attributes
    dark_pos = _get_getter("_dark_pos", lambda self: self.sim.particles['dark']['position'][self.dark_in_halo_filter] - self.center_pos, "_dark_vars")
    dark_x, dark_y, dark_z = [property(lambda self: self.dark_pos[:, i] for i in range(3))]
    dark_vel = _get_getter("_dark_vel", lambda self: self.sim.particles['dark']['velocity'][self.dark_in_halo_filter] - self.center_vel, "_dark_vars")
    dark_vx, dark_vy, dark_vz = [property(lambda self: self.dark_vel[:, i] for i in range(3))]
    dark_distance = _get_getter("_dark_distance", lambda self: np.sqrt(np.sum(np.square(self.dark_pos), 1)), "_dark_vars")
    dark_r2d = _get_getter("_dark_r2d", lambda self: np.sqrt(np.sum(np.square(self.dark_pos[:, [0, 1]]), 1)), "_dark_vars")
    dark_speed = _get_getter("_dark_speed", lambda self: np.sqrt(np.sum(np.square(self.dark_vel), 1)), "_dark_vars")
    dark_id = _get_getter("_dark_id", lambda self: self.sim.particles['dark']['id'][self.dark_in_halo_filter], "_dark_vars")
    dark_mass = _get_getter("_dark_mass", lambda self: self.sim.particles['dark']['mass'][self.dark_in_halo_filter], "_dark_vars")

    # Dark2 attributes
    dark2_pos = _get_getter("_dark2_pos", lambda self: self.sim.particles['dark2']['position'][self.dark2_in_halo_filter] - self.center_pos, "_dark2_vars")
    dark2_x, dark2_y, dark2_z = [property(lambda self: self.dark2_pos[:, i] for i in range(3))]
    dark2_vel = _get_getter("_dark2_vel", lambda self: self.sim.particles['dark2']['velocity'][self.dark2_in_halo_filter] - self.center_vel, "_dark2_vars")
    dark2_vx, dark2_vy, dark2_vz = [property(lambda self: self.dark2_vel[:, i] for i in range(3))]
    dark2_distance = _get_getter("_dark2_distance", lambda self: np.sqrt(np.sum(np.square(self.dark2_pos), 1)), "_dark2_vars")
    dark2_r2d = _get_getter("_dark2_r2d", lambda self: np.sqrt(np.sum(np.square(self.dark2_pos[:, [0, 1]]), 1)), "_dark2_vars")
    dark2_speed = _get_getter("_dark2_speed", lambda self: np.sqrt(np.sum(np.square(self.dark2_vel), 1)), "_dark2_vars")
    dark2_id = _get_getter("_dark2_id", lambda self: self.sim.particles['dark2']['id'][self.dark2_in_halo_filter], "_dark2_vars")
    dark2_mass = _get_getter("_dark2_mass", lambda self: self.sim.particles['dark2']['mass'][self.dark2_in_halo_filter], "_dark2_vars")

    # Lists of individual particles
    stars = _get_getter("_stars", lambda self: [Star(self, i) for i in range(len(self.star_id))])  # This is dependent on the mask applied to Halo, but that's fine because self._stars is reset every time the mask is changed
    gas = _get_getter("_gas", lambda self: [Gas(self, i) for i in range(len(self.gas_id))])
    dark = _get_getter("dark", lambda self: [Dark(self, i) for i in range(len(self.dark_id))])
    dark2 = _get_getter("dark2", lambda self: [Dark2(self, i) for i in range(len(self.dark2_id))])

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
    
class Star(Particle):
    POS_ATTR = "star_pos"
    VEL_ATTR = "star_vel"
    MASS_ATTR = "star_mass"
    ID_ATTR = "star_id"
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
    DISTANCE_ATTR = "gas_distance"
    R2D_ATTR = "gas_r2d"
    SPEED_ATTR = "gas_speed"

    MASS_FRACTION_ATTR = "gas_mass_fraction"
    DENSITY_ATTR = "gas_density"
    ELECTRON_FRACTION_ATTR = "gas_electron_fraction"
    TEMPERATURE_ATTR = "gas_temperature"
    HYDROGEN_NEUTRAL_FRACTION_ATTR = "gas_hydrogen_neutral_fraction"
    SIZE_ATTR = "gas_size"

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
  
class Dark(Particle):
    POS_ATTR = "dark_pos"
    VEL_ATTR = "dark_vel"
    MASS_ATTR = "dark_mass"
    ID_ATTR = "dark_id"
    DISTANCE_ATTR = "dark_distance"
    R2D_ATTR = "dark_r2d"
    SPEED_ATTR = "dark_speed"
    
class Dark2(Particle):
    POS_ATTR = "dark2_pos"
    VEL_ATTR = "dark2_vel"
    MASS_ATTR = "dark2_mass"
    ID_ATTR = "dark2_id"
    DISTANCE_ATTR = "dark2_distance"
    R2D_ATTR = "dark2_r2d"
    SPEED_ATTR = "dark2_speed"


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