from __future__ import annotations
import numpy as np, pickle, Particle
from numpy.typing import NDArray
from typing import Sequence, Union

# TODO: This vvv
import sys
sys.path.insert(0, "/home/olive/Documents/Research Python/playground/src")
import galaxy_3d_renderer

# TODO: I'm counting particles by len(self.star_id) or len(self.star_pos). Make a variable that stores this instead, without having to access attributes
# TODO: Make a variable/getter/method to check if a given particle is included. Currently i'm doing if self.snapshot and Species.particle in self.snapshot.particles
#   Some cases could also just use n_particles == 0

# TODO: Just make it more clear what variables represent the existence/definedness/whatever of attributes/particles (and how/when to check each different one)

# TODO: Make attributes return None if boolean mask is being applied

# TODO: Deal with self.num_stars_in_filter being undefined until a filter is applied

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

class ClassType:
    SIMULATION = "simulation"
    SNAPSHOT = "snapshot"

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
    # All methods return the class, for method chaining (i.e. halo = Halo(...).restrict_slice(...).restrict_ids(...))

    def __init__(self, snapshot, species: Sequence[str] = ("all",)):
        """Get a new particle group object representing a user-defined set of particles in the provided simulation. 
        Initially contains all particles, but restrictions can be called on this object to filter specific particles.

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
        
        # *Technically* we can just set self.snapshot to a Simulation if one is passed in (since they both implement the same relevant methods), but this just makes it simpler for error raising and intuition
        self.snapshot = snapshot.snapshot if snapshot.CLASS_TYPE == ClassType.SIMULATION else snapshot

        self.incl_stars = Species.star in snapshot.particles and (Species.all in species or Species.star in species)
        self.incl_gas = Species.gas in snapshot.particles and (Species.all in species or Species.gas in species)
        self.incl_dark = Species.dark in snapshot.particles and (Species.all in species or Species.dark in species)
        self.incl_dark2 = Species.dark2 in snapshot.particles and (Species.all in species or Species.dark2 in species)

        self._star_vars = set()
        self._gas_vars = set()
        self._dark_vars = set()
        self._dark2_vars = set()

    def reset_all_particle_attributes(self):
        for var_name in self._star_vars | self._gas_vars | self._dark_vars | self._dark2_vars:
            self.__setattr__(var_name, None)
        
        return self

    # Filters (arrays of booleans that filter which particles are being stored)
    stars_in_halo_filter: NDArray[bool] = _get_getter("_stars_in_halo_filter", lambda self: self.generate_constant_filter_getter(True)(self, Species.star, None))
    gas_in_halo_filter: NDArray[bool] = _get_getter("_gas_in_halo_filter", lambda self: self.generate_constant_filter_getter(True)(self, Species.gas, None))
    dark_in_halo_filter: NDArray[bool] = _get_getter("_dark_in_halo_filter", lambda self: self.generate_constant_filter_getter(True)(self, Species.dark, None))
    dark2_in_halo_filter: NDArray[bool] = _get_getter("_dark2_in_halo_filter", lambda self: self.generate_constant_filter_getter(True)(self, Species.dark2, None))

    # Function to apply a filter getter to all included particles
    def apply_filter(self, filter_getter, boolean=None) -> ParticleGroup:

        if boolean is True or isinstance(boolean, str) and boolean.lower() in ["and", "&", Boolean.AND]:
            # Apply the relative filter to all loaded attributes
            particles_to_filter = []
            if self.incl_stars: particles_to_filter.append((Species.star, self._star_vars, filter_getter(self, Species.star, self.star_pos)))
            if self.incl_gas: particles_to_filter.append((Species.gas, self._gas_vars, filter_getter(self, Species.gas, self.gas_pos)))
            if self.incl_dark: particles_to_filter.append((Species.dark, self._dark_vars, filter_getter(self, Species.dark, self.dark_pos)))
            if self.incl_dark2: particles_to_filter.append((Species.dark2, self._dark2_vars, filter_getter(self, Species.dark2, self.dark2_pos)))

            for species, var_set, boolean_filter in particles_to_filter:
                num_of_particle = None
                for var_name in var_set:
                    attribute = self.__getattribute__(var_name)
                    if attribute is not None:
                        new_attr = attribute[boolean_filter]
                        self.__setattr__(var_name, new_attr)
                    
                        if num_of_particle is None: num_of_particle = len(new_attr)

                if species == Species.star and num_of_particle is not None: self.num_stars_in_filter = num_of_particle
                if species == Species.gas and num_of_particle is not None: self.num_gas_in_filter = num_of_particle
                if species == Species.dark and num_of_particle is not None: self.num_dark_in_filter = num_of_particle
                if species == Species.dark2 and num_of_particle is not None: self.num_dark2_in_filter = num_of_particle

        else:
            if getattr(self, "snapshot", None) is None:
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

            if self.incl_stars: self.num_stars_in_filter = np.count_nonzero(self._stars_in_halo_filter)
            if self.incl_gas: self.num_gas_in_filter = np.count_nonzero(self._gas_in_halo_filter)
            if self.incl_dark: self.num_dark_in_filter = np.count_nonzero(self._dark_in_halo_filter)
            if self.incl_dark2: self.num_dark2_in_filter = np.count_nonzero(self._dark2_in_halo_filter)

            self.reset_all_particle_attributes()
        

        return self
    
    # Functions that generate return filter getters (filter getters are to be passed into apply_filter)
    def generate_constant_filter_getter(self, all_true_or_false):
        return lambda self, species, particle_positions=None: np.full(
            len(self.snapshot.particles[species]['position'] if particle_positions is None else particle_positions), 
            all_true_or_false
        )

    def generate_radius_filter_getter(self, radius):
        def filter_getter(self, species, particle_positions=None):
            all_rel_particle_pos = (self.snapshot.particles[species]['position'] - self.center_pos) \
                if particle_positions is None else particle_positions
            # print(species, all_rel_particle_pos)
                
            particle_pos_sum_of_squares = np.sum(np.square(all_rel_particle_pos), 1)
            return particle_pos_sum_of_squares < radius**2
        
        if radius is None:
            return self.generate_constant_filter_getter(True)
        else:
            return filter_getter

    def generate_slice_filter_getter(self, face = 'xy', proj_distance = 1, thickness = 1):
        face = face.lower()

        def filter_getter(self, species, particle_positions=None):
            all_rel_particle_pos = (self.snapshot.particles[species]['position'] - self.center_pos) \
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
        def filter_getter(self, species, particle_positions=None):
            ids = id_map[species]
            if ids is None:
                return self.generate_constant_filter_getter(True)
            elif len(ids) == 0:
                return self.generate_constant_filter_getter(False)
            
            all_particle_ids = self.snapshot.particles[species]['id'] if particle_positions is None else particle_positions
            return np.isin(all_particle_ids, ids)
        
            # Could sort stars by ID, then index it by star_ids (probably not, since IDs have gaps)
            # NEVERMIND np.isin IS AMAZING

        return filter_getter

    def generate_quantity_filter_getter(self, star_num=0, gas_num=0, dark_num=0, dark2_num=0):
        num_map = {Species.star: star_num, Species.gas: gas_num, Species.dark: dark_num, Species.dark2: dark2_num}

        def filter_getter(self, species, particle_positions=None):
            all_rel_particle_pos = (self.snapshot.particles[species]['position'] - self.center_pos) \
                if particle_positions is None else particle_positions
            
            quantity = num_map[species]

            if quantity == 0:
                return self.generate_constant_filter_getter(False)(self, species, particle_positions)
            elif quantity >= len(all_rel_particle_pos):
                return self.generate_constant_filter_getter(True)(self, species, particle_positions)
            
            true_values = np.full(quantity, True)
            false_values = np.full(len(all_rel_particle_pos) - quantity, False)
            all_values = np.concatenate([true_values, false_values])
            np.random.shuffle(all_values)
            return all_values

        return filter_getter
    
    def generate_quantity_by_percentage_filter_getter(self, star_percent=0, gas_percent=0, dark_percent=0, dark2_percent=0):
        return self.generate_quantity_filter_getter(
            int(self.num_stars_in_filter * star_percent / 100) if self.star_id is not None else 0,
            int(self.num_gas_in_filter * gas_percent / 100) if self.gas_id is not None else 0,
            int(self.num_dark_in_filter * dark_percent / 100) if self.dark_id is not None else 0,
            int(self.num_dark2_in_filter * dark2_percent / 100) if self.dark2_id is not None else 0,
        )

    def generate_proportional_quantity_filter(self, total_quantity, star_weight=1, gas_weight=1, dark_weight=1, dark2_weight=1):
        num_stars = self.num_stars_in_filter * star_weight if self.num_stars_in_filter is not None else 0
        num_gas = self.num_gas_in_filter * gas_weight if self.num_gas_in_filter is not None else 0
        num_dark = self.num_dark_in_filter * dark_weight if self.num_dark_in_filter is not None else 0
        num_dark2 = self.num_dark2_in_filter * dark2_weight if self.num_dark2_in_filter is not None else 0

        # print(num_dark2)

        total_num = num_stars + num_gas + num_dark + num_dark2

        if total_num == 0:
            return self.generate_constant_filter_getter(False)
        
        return self.generate_quantity_filter_getter(
            int(num_stars / total_num * total_quantity),
            int(num_gas / total_num * total_quantity),
            int(num_dark / total_num * total_quantity),
            int(num_dark2 / total_num * total_quantity)
        )
        

    # Functions that quickly generate a filter getter and pass it into apply filter
    def reset_restriction(self, boolean=None) -> ParticleGroup:
        return self.apply_filter(self.generate_constant_filter_getter(True), boolean)

    def restrict_radius(self, radius: Union[float, int, None], boolean=None) -> ParticleGroup:
        return self.apply_filter(self.generate_radius_filter_getter(radius), boolean)

    def restrict_slice(self, face = 'xy', proj_distance=1, thickness=1, boolean=None) -> ParticleGroup:
        return self.apply_filter(self.generate_slice_filter_getter(face, proj_distance, thickness), boolean)

    def restrict_ids(self, star_ids=[], gas_ids=[], dark_ids=[], dark2_ids=[], boolean=None) -> ParticleGroup:
        return self.apply_filter(self.generate_ids_filter_getter(star_ids, gas_ids, dark_ids, dark2_ids), boolean)

    def restrict_quantity(self, star_num=0, gas_num=0, dark_num=0, dark2_num=0, boolean=None) -> ParticleGroup:
        return self.apply_filter(self.generate_quantity_filter_getter(star_num, gas_num, dark_num, dark2_num), boolean)

    def restrict_quantity_by_percentage(self, star_percent=0, gas_percent=0, dark_percent=0, dark2_percent=0, boolean=None) -> ParticleGroup:
        return self.apply_filter(self.generate_quantity_by_percentage_filter_getter(star_percent, gas_percent, dark_percent, dark2_percent), boolean)

    def restrict_quantity_proportionally(self, total_quantity, star_weight=1, gas_weight=1, dark_weight=1, dark2_weight=1, boolean=None) -> ParticleGroup:
        return self.apply_filter(self.generate_proportional_quantity_filter(total_quantity, star_weight, gas_weight, dark_weight, dark2_weight), boolean)
   
    # Star attributes
    star_pos: NDArray[np.float64] = _get_getter("_star_pos", lambda self: self.snapshot.particles[Species.star]['position'][self.stars_in_halo_filter] - self.center_pos if self.snapshot and Species.star in self.snapshot.particles else None, "_star_vars")
    star_x, star_y, star_z = [property(lambda self: self.star_pos[:, i] if self.snapshot and Species.star in self.snapshot.particles else None) for i in range(3)]
    star_vel: NDArray[np.float32] = _get_getter("_star_vel", lambda self: self.snapshot.particles[Species.star]['velocity'][self.stars_in_halo_filter] - self.center_vel if self.snapshot and Species.star in self.snapshot.particles else None, "_star_vars")
    star_vx, star_vy, star_vz = [property(lambda self: self.star_vel[:, i] if self.snapshot and Species.star in self.snapshot.particles else None) for i in range(3)]
    star_distance: NDArray[np.float64] = _get_getter("_star_distance", lambda self: np.sqrt(np.sum(np.square(self.star_pos), 1)) if self.snapshot and Species.star in self.snapshot.particles else None, "_star_vars")
    star_r2d: NDArray[np.float64] = _get_getter("_star_r2d", lambda self: np.sqrt(np.sum(np.square(self.star_pos[:, [0, 1]]), 1)) if self.snapshot and Species.star in self.snapshot.particles else None, "_star_vars")
    star_speed: NDArray[np.float32] = _get_getter("_star_speed", lambda self: np.sqrt(np.sum(np.square(self.star_vel), 1)) if self.snapshot and Species.star in self.snapshot.particles else None, "_star_vars")
    star_id: NDArray[np.uint32] = _get_getter("_star_id", lambda self: self.snapshot.particles[Species.star]['id'][self.stars_in_halo_filter] if self.snapshot and Species.star in self.snapshot.particles else None, "_star_vars")
    star_id_child: NDArray[np.uint8] = _get_getter("_star_id_child", lambda self: self.snapshot.particles[Species.star]['id.child'][self.stars_in_halo_filter] if self.snapshot and Species.star in self.snapshot.particles else None, "_star_vars")
    star_id_generation: NDArray[np.uint8] = _get_getter("_star_id_generation", lambda self: self.snapshot.particles[Species.star]['id.generation'][self.stars_in_halo_filter] if self.snapshot and Species.star in self.snapshot.particles else None, "_star_vars")
    star_mass: NDArray[np.float32] = _get_getter("_star_mass", lambda self: self.snapshot.particles[Species.star]['mass'][self.stars_in_halo_filter] if self.snapshot and Species.star in self.snapshot.particles else None, "_star_vars")
    
    star_scale_factor: NDArray[np.float32] = _get_getter("_star_scale_factor", lambda self: self.snapshot.particles[Species.star]['form.scalefactor'][self.stars_in_halo_filter] if self.snapshot and Species.star in self.snapshot.particles else None, "_star_vars")
    star_mass_fraction: NDArray[np.float32] = _get_getter("_star_mass_fraction", lambda self: self.snapshot.particles[Species.star]['massfraction'][self.stars_in_halo_filter] if self.snapshot and Species.star in self.snapshot.particles else None, "_star_vars")
    # TODO: What is star scale_factor? Why isn't 'age' in particles['star']? Any correlation, since scale_factor isn't in other implementation but age is.

    # Gas attributes
    gas_pos: NDArray[np.float64] = _get_getter("_gas_pos", lambda self: self.snapshot.particles[Species.gas]['position'][self.gas_in_halo_filter] - self.center_pos if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_x, gas_y, gas_z = [property(lambda self: self.gas_pos[:, i] if self.snapshot and Species.gas in self.snapshot.particles else None) for i in range(3)]
    gas_vel: NDArray[np.float32] = _get_getter("_gas_vel", lambda self: self.snapshot.particles[Species.gas]['velocity'][self.gas_in_halo_filter] - self.center_vel if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_vx, gas_vy, gas_vz = [property(lambda self: self.gas_vel[:, i] if self.snapshot and Species.gas in self.snapshot.particles else None) for i in range(3)]
    gas_distance: NDArray[np.float64] = _get_getter("_gas_distance", lambda self: np.sqrt(np.sum(np.square(self.gas_pos), 1)) if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_r2d: NDArray[np.float64] = _get_getter("_gas_r2d", lambda self: np.sqrt(np.sum(np.square(self.gas_pos[:, [0, 1]]), 1)) if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_speed: NDArray[np.float32] = _get_getter("_gas_speed", lambda self: np.sqrt(np.sum(np.square(self.gas_vel), 1)) if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_id: NDArray[np.uint32] = _get_getter("_gas_id", lambda self: self.snapshot.particles[Species.gas]['id'][self.gas_in_halo_filter] if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_id_child: NDArray[np.uint8] = _get_getter("_gas_id_child", lambda self: self.snapshot.particles[Species.gas]['id.child'][self.gas_in_halo_filter] if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_id_generation: NDArray[np.uint8] = _get_getter("_gas_id_generation", lambda self: self.snapshot.particles[Species.gas]['id.generation'][self.gas_in_halo_filter] if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_mass: NDArray[np.float32] = _get_getter("_gas_mass", lambda self: self.snapshot.particles[Species.gas]['mass'][self.gas_in_halo_filter] if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    
    gas_mass_fraction: NDArray[np.float32] = _get_getter("_gas_mass_fraction", lambda self: self.snapshot.particles[Species.gas]['massfraction'][self.gas_in_halo_filter] if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_density: NDArray[np.float32] = _get_getter("_gas_density", lambda self: self.snapshot.particles[Species.gas]['density'][self.gas_in_halo_filter] if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_electron_fraction: NDArray[np.float32] = _get_getter("_gas_electron_fraction", lambda self: self.snapshot.particles[Species.gas]['electron.fraction'][self.gas_in_halo_filter] if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_temperature: NDArray[np.float32] = _get_getter("gas_temperature", lambda self: self.snapshot.particles[Species.gas]['temperature'][self.gas_in_halo_filter] if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_hydrogen_neutral_fraction: NDArray[np.float32] = _get_getter("_gas_hydrogen_neutral_fraction", lambda self: self.snapshot.particles[Species.gas]['hydrogen.neutral.fraction'][self.gas_in_halo_filter] if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_size: NDArray[np.float32] = _get_getter("_gas_size", lambda self: self.snapshot.particles[Species.gas]['size'][self.gas_in_halo_filter] if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")
    gas_sfr: NDArray[np.float32] = _get_getter("_gas_sfr", lambda self: self.snapshot.particles[Species.gas]['sfr'][self.gas_in_halo_filter] if self.snapshot and Species.gas in self.snapshot.particles else None, "_gas_vars")

    # Dark attributes
    dark_pos: NDArray[np.float64] = _get_getter("_dark_pos", lambda self: self.snapshot.particles[Species.dark]['position'][self.dark_in_halo_filter] - self.center_pos if self.snapshot and Species.dark in self.snapshot.particles else None, "_dark_vars")
    dark_x, dark_y, dark_z = [property(lambda self: self.dark_pos[:, i] if self.snapshot and Species.dark in self.snapshot.particles else None) for i in range(3)]
    dark_vel: NDArray[np.float32] = _get_getter("_dark_vel", lambda self: self.snapshot.particles[Species.dark]['velocity'][self.dark_in_halo_filter] - self.center_vel if self.snapshot and Species.dark in self.snapshot.particles else None, "_dark_vars")
    dark_vx, dark_vy, dark_vz = [property(lambda self: self.dark_vel[:, i] if self.snapshot and Species.dark in self.snapshot.particles else None) for i in range(3)]
    dark_distance: NDArray[np.float64] = _get_getter("_dark_distance", lambda self: np.sqrt(np.sum(np.square(self.dark_pos), 1)) if self.snapshot and Species.dark in self.snapshot.particles else None, "_dark_vars")
    dark_r2d: NDArray[np.float64] = _get_getter("_dark_r2d", lambda self: np.sqrt(np.sum(np.square(self.dark_pos[:, [0, 1]]), 1)) if self.snapshot and Species.dark in self.snapshot.particles else None, "_dark_vars")
    dark_speed: NDArray[np.float32] = _get_getter("_dark_speed", lambda self: np.sqrt(np.sum(np.square(self.dark_vel), 1)) if self.snapshot and Species.dark in self.snapshot.particles else None, "_dark_vars")
    dark_id: NDArray[np.uint32] = _get_getter("_dark_id", lambda self: self.snapshot.particles[Species.dark]['id'][self.dark_in_halo_filter] if self.snapshot and Species.dark in self.snapshot.particles else None, "_dark_vars")
    dark_id_child: NDArray[np.uint8] = _get_getter("_dark_id_child", lambda self: self.snapshot.particles[Species.dark]['id.child'][self.dark_in_halo_filter] if self.snapshot and Species.dark in self.snapshot.particles else None, "_dark_vars")
    dark_id_generation: NDArray[np.uint8] = _get_getter("_dark_id_generation", lambda self: self.snapshot.particles[Species.dark]['id.generation'][self.dark_in_halo_filter] if self.snapshot and Species.dark in self.snapshot.particles else None, "_dark_vars")
    dark_mass: NDArray[np.float32] = _get_getter("_dark_mass", lambda self: self.snapshot.particles[Species.dark]['mass'][self.dark_in_halo_filter] if self.snapshot and Species.dark in self.snapshot.particles else None, "_dark_vars")

    # Dark2 attributes
    dark2_pos: NDArray[np.float64] = _get_getter("_dark2_pos", lambda self: self.snapshot.particles[Species.dark2]['position'][self.dark2_in_halo_filter] - self.center_pos if self.snapshot and Species.dark2 in self.snapshot.particles else None, "_dark2_vars")
    dark2_x, dark2_y, dark2_z = [property(lambda self: self.dark2_pos[:, i] if self.snapshot and Species.dark2 in self.snapshot.particles else None) for i in range(3)]
    dark2_vel: NDArray[np.float32] = _get_getter("_dark2_vel", lambda self: self.snapshot.particles[Species.dark2]['velocity'][self.dark2_in_halo_filter] - self.center_vel if self.snapshot and Species.dark2 in self.snapshot.particles else None, "_dark2_vars")
    dark2_vx, dark2_vy, dark2_vz = [property(lambda self: self.dark2_vel[:, i] if self.snapshot and Species.dark2 in self.snapshot.particles else None) for i in range(3)]
    dark2_distance: NDArray[np.float64] = _get_getter("_dark2_distance", lambda self: np.sqrt(np.sum(np.square(self.dark2_pos), 1)) if self.snapshot and Species.dark2 in self.snapshot.particles else None, "_dark2_vars")
    dark2_r2d: NDArray[np.float64] = _get_getter("_dark2_r2d", lambda self: np.sqrt(np.sum(np.square(self.dark2_pos[:, [0, 1]]), 1)) if self.snapshot and Species.dark2 in self.snapshot.particles else None, "_dark2_vars")
    dark2_speed: NDArray[np.float32] = _get_getter("_dark2_speed", lambda self: np.sqrt(np.sum(np.square(self.dark2_vel), 1)) if self.snapshot and Species.dark2 in self.snapshot.particles else None, "_dark2_vars")
    dark2_id: NDArray[np.uint32] = _get_getter("_dark2_id", lambda self: self.snapshot.particles[Species.dark2]['id'][self.dark2_in_halo_filter] if self.snapshot and Species.dark2 in self.snapshot.particles else None, "_dark2_vars")
    dark2_id_child: NDArray[np.uint8] = _get_getter("_dark2_id_child", lambda self: self.snapshot.particles[Species.dark2]['id.child'][self.dark2_in_halo_filter] if self.snapshot and Species.dark2 in self.snapshot.particles else None, "_dark2_vars")
    dark2_id_generation: NDArray[np.uint8] = _get_getter("_dark2_id_generation", lambda self: self.snapshot.particles[Species.dark2]['id.generation'][self.dark2_in_halo_filter] if self.snapshot and Species.dark2 in self.snapshot.particles else None, "_dark2_vars")
    dark2_mass: NDArray[np.float32] = _get_getter("_dark2_mass", lambda self: self.snapshot.particles[Species.dark2]['mass'][self.dark2_in_halo_filter] if self.snapshot and Species.dark2 in self.snapshot.particles else None, "_dark2_vars")

    # Lists of individual particles
    stars: list[Particle.Star] = _get_getter("_stars", lambda self: [Particle.Star(self, i) for i in range(self.num_stars_in_filter)] if self.snapshot and Species.star in self.snapshot.particles else [])  # This is dependent on the mask applied to Halo, but that's fine because self._stars is reset every time the mask is changed
    gas: list[Particle.Gas] = _get_getter("_gas", lambda self: [Particle.Gas(self, i) for i in range(self.num_gas_in_filter)] if self.snapshot and Species.gas in self.snapshot.particles else [])
    dark: list[Particle.Dark] = _get_getter("_dark", lambda self: [Particle.ark(self, i) for i in range(self.num_dark_in_filter)] if self.snapshot and Species.dark in self.snapshot.particles else [])
    dark2: list[Particle.Dark2] = _get_getter("_dark2", lambda self: [Particle.Dark2(self, i) for i in range(self.num_dark2_in_filter)] if self.snapshot and Species.dark2 in self.snapshot.particles else [])

    def center_on_value(self, new_pos=None, new_vel=None) -> ParticleGroup:
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

    def center_on_halo(self, other_id, change_velocity=True, change_position=True) -> ParticleGroup:
        """
        Changes the center position (and/or velocity, if specified). 
        Recalculating particles (i.e. using calling_percentage) will still use the old center. This simply changes the reference frame of the positions and/or velocities.
        """
        # TODO: Actually I probably want to recalculate particles with new center. Cause they can just center_on *after* restricting, to use the old center for restricting

        new_pos = new_vel = None
        if change_position:
            new_pos = np.array([self.snapshot.halo_data.field('Xc(6)')[other_id], 
                                self.snapshot.halo_data.field('Yc(7)')[other_id], 
                                self.snapshot.halo_data.field('Zc(8)')[other_id]]) / self.snapshot.h
            
        if change_velocity:
            new_vel = np.array([self.snapshot.halo_data.field('Xc(6)')[other_id], 
                                self.snapshot.halo_data.field('Yc(7)')[other_id], 
                                self.snapshot.halo_data.field('Zc(8)')[other_id]]) / self.snapshot.h
            
        return self.center_on_value(new_pos, new_vel)

    @staticmethod
    def extract_x(array_of_vectors: NDArray) -> NDArray:
        return array_of_vectors[:, 0]
    
    @staticmethod
    def extract_y(array_of_vectors: NDArray) -> NDArray:
        return array_of_vectors[:, 1]
    
    @staticmethod
    def extract_z(array_of_vectors: NDArray) -> NDArray:
        return array_of_vectors[:, 2]

    def save_to_file(self, file_name) -> ParticleGroup:
        temp_sim = self.snapshot
        self.snapshot = None
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
        self.snapshot = temp_sim

        return self

    @staticmethod
    def load_from_file(file_name, sim=None) -> ParticleGroup:
        with open(file_name, 'rb') as file:
            loaded_object = pickle.load(file)
        if sim:
            loaded_object.snapshot = sim
        return loaded_object

    def get_draw_arrays(self, stars=True, gas=True, dark=True, dark2=True):
        if stars is not False and self.star_pos is None: stars = False
        if gas is not False and self.gas_pos is None: gas = False
        if dark is not False and self.dark_pos is None: dark = False
        if dark2 is not False and self.dark2_pos is None: dark2 = False

        if stars is True: stars = [255, 255, 255, 0.7]
        if gas is True: gas = [150, 150, 150, 0.7]
        if dark is True: dark = [50, 50, 255, 0.9]
        if dark2 is True: dark2 = [255, 150, 0, 1.3] 

        star_cols = gas_cols = dark_cols = dark2_cols = np.empty((0, 4))

        if stars is not False: star_cols = np.ones((self.num_stars_in_filter, 4)) * stars if isinstance(stars, Sequence) else np.array([stars(star_particle) for star_particle in self.stars])
        if gas is not False: gas_cols = np.ones((self.num_gas_in_filter, 4)) * gas if isinstance(gas, Sequence) else np.array([gas(gas_particle) for gas_particle in self.gas])
        if dark is not False: dark_cols = np.ones((self.num_dark_in_filter, 4)) * dark if isinstance(dark, Sequence) else np.array([dark(dark_particle) for dark_particle in self.dark])
        if dark2 is not False: dark2_cols = np.ones((self.num_dark2_in_filter, 4)) * dark2 if isinstance(dark2, Sequence) else np.array([dark2(dark2_particle) for dark2_particle in self.dark2])

        cols = np.vstack((star_cols, gas_cols, dark_cols, dark2_cols))
        
        star_pos = gas_pos = dark_pos = dark2_pos = np.empty((0, 3))

        if stars is not False: star_pos = self.star_pos
        if gas is not False: gas_pos = self.gas_pos
        if dark is not False: dark_pos = self.dark_pos
        if dark2 is not False: dark2_pos = self.dark2_pos

        positions = np.vstack((star_pos, gas_pos, dark_pos, dark2_pos))
        
        return (positions, cols)
    
    def get_center_point(self, species=Species.all):
        max_positions = []
        min_positions = []


        if (species == Species.star or species == Species.all) and self.star_pos is not None and len(self.star_pos) > 0:
            max_positions.append(np.max(self.star_pos, 0))
            min_positions.append(np.min(self.star_pos, 0))
        if (species == Species.gas or species == Species.all) and self.gas_pos is not None and len(self.gas_pos) > 0:
            max_positions.append(np.max(self.gas_pos, 0))
            min_positions.append(np.min(self.gas_pos, 0))
        if (species == Species.dark or species == Species.all) and self.dark_pos is not None and len(self.dark_pos) > 0:
            max_positions.append(np.max(self.dark_pos, 0))
            min_positions.append(np.min(self.dark_pos, 0))
        if (species == Species.dark2 or species == Species.all) and self.dark2_pos is not None and len(self.dark2_pos) > 0:
            max_positions.append(np.max(self.dark2_pos, 0))
            min_positions.append(np.min(self.dark2_pos, 0))

        if len(max_positions) > 0 and len(min_positions) > 0:
            return (np.max(max_positions, 0) + np.min(min_positions, 0)) * 0.5
        else:
            return None        
        
    def center_on_center_point(self, species=Species.all) -> ParticleGroup:
        self.center_on_value(self.get_center_point(species))

    def render(self, stars=True, gas=True, dark=True, dark2=True, distance_falloff=1/600, start_pos=(0., 0., 20.)):
        galaxy_3d_renderer.render_points(*self.get_draw_arrays(stars, gas, dark, dark2), distance_falloff, start_pos)

class Halo(ParticleGroup):

    def __init__(self, 
                 sim, 
                 halo_id: int, 
                 restrict_percentage: Union[float, int, None] = 100,
                 species: Sequence[str] = ("all",)):

        self.halo_id = halo_id
        self.halo_radius = sim.get_field('12')[halo_id]

        self.restricted_percentage = restrict_percentage

        super().__init__(sim, species)

        self.center_on_halo(halo_id)

        # The absolute coordinates of the center of this halo. This does not change.
        self.this_halo_center_pos = np.array([self.center_pos[0], self.center_pos[1], self.center_pos[2]])
        # See this_halo_center_pos, but for velocity.
        self.this_halo_center_vel = np.array([self.center_vel[0], self.center_vel[1], self.center_vel[2]])

    host_halo_id: int = _get_getter("_host_halo_id", lambda self: self.snapshot.halo_data.field('hostHalo(2)')[self.halo_id])
    sub_halo_ids = _get_getter("_sub_halo_ids", lambda self: np.where(self.snapshot.halo_data.field('hostHalo(2)') == self.halo_id))
    mass: float = _get_getter("_mass", lambda self: self.snapshot.get_field('4')[self.halo_id])
    r_max: float = _get_getter("_r_max", lambda self: self.snapshot.halo_data.field('Rmax(13)')[self.halo_id] / self.snapshot.h)
    v_max: float = _get_getter("_v_max", lambda self: self.snapshot.halo_data.field('Vmax(17)')[self.halo_id])
    v_esc: float = _get_getter("_v_esc", lambda self: self.snapshot.halo_data.field('v_esc(18)')[self.halo_id])
    num_gas: int = _get_getter("_num_gas", lambda self: self.snapshot.halo_data.field('n_gas(44)')[self.halo_id])
    gas_mass = _get_getter("_gas_mass", lambda self: self.snapshot.halo_data.field('M_gas(45)')[self.halo_id])
    num_stars = _get_getter("_num_stars", lambda self: self.snapshot.halo_data.field('n_star(64)')[self.halo_id])
    star_mass: float = _get_getter("_star_mass", lambda self: self.snapshot.halo_data.field('M_star(65)')[self.halo_id])
    num_particles: int = _get_getter("_num_particles", lambda self: self.snapshot.halo_data.field('npart(5)')[self.halo_id])

    stars_in_halo_filter: NDArray[bool] = _get_getter("_stars_in_halo_filter", lambda self: self.generate_percentage_filter_getter(self.restricted_percentage)(self, Species.star, None))
    gas_in_halo_filter: NDArray[bool] = _get_getter("_gas_in_halo_filter", lambda self: self.generate_percentage_filter_getter(self.restricted_percentage)(self, Species.gas, None))
    dark_in_halo_filter: NDArray[bool] = _get_getter("_dark_in_halo_filter", lambda self: self.generate_percentage_filter_getter(self.restricted_percentage)(self, Species.dark, None))
    dark2_in_halo_filter: NDArray[bool] = _get_getter("_dark2_in_halo_filter", lambda self: self.generate_percentage_filter_getter(self.restricted_percentage)(self, Species.dark2, None))

    def generate_percentage_filter_getter(self, percentage):
        if percentage is None:
            return self.generate_constant_filter_getter(True)
        else:
            return self.generate_radius_filter_getter(self.halo_radius * (percentage / 100))
        
    def restrict_percentage(self, percentage: Union[float, int, None], boolean=None) -> Halo:
        return self.apply_filter(self.generate_percentage_filter_getter(percentage), boolean)
    
    def recenter_on_this_halo(self) -> Halo:
        return self.center_on_halo(self.halo_id)

    def get_host_halo(self, **halo_kwargs) -> Halo:
        if self.host_halo_id == -1:
            return None
        else:
            return Halo(self.snapshot, self.host_halo_id, **halo_kwargs)
        
    def get_sub_halos(self, **halo_kwargs) -> list[Halo]:
        return [Halo(self.snapshot, child_id, **halo_kwargs) for child_id in self.child_halo_ids]
