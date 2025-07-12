import os, gizmo_analysis as gizmo, astropy.io.ascii as ascii, Halo, abc
from typing import Union, Sequence, Literal

# TODO: Make all the parameter and variable names make intuitive sense. They suck in my implementation right now.
#   HaloData
#   Plurality
#   Simulation file path and simulation file name can be combined, probably.
#     Just do a simple os.file exists() on f"../data/{name}", otherwise treat it as a path. Return an error if it doesnt exist at the path either

# TODO: If multiple snapshots are passed in, let the user also pass in multiple halo finder types

# TODO: Need to make getting fields from HaloData more abstract. 
#   Like make some number of fields that are in most halo finder data files available as attributes(?)

def get_all_ints_in_string(string):
    """
    Returns all full integers that appear in the passed in string. I.e. passing in "abc123def456" returns [123, 456]. If no number is found, returns an empty array
    """
    nums = []
    current_num = ""
    for char in string:
        if char.isdigit():
            current_num += char
        elif len(current_num) > 0:
            nums.append(int(current_num))
            current_num = ""

    return nums

def find_file_containing_number(filenames, number, file_ending):
    """
    For all the filenames passed in, finds the first one that has a given number in it (and ends in the string file_ending)
    """
    # snapshot_name = f"snapshot_{snapshot_value}" # TODO: Fill with zeros
    for filename in filenames:
        file_num = get_all_ints_in_string(filename)
        file_num = file_num[0] if len(file_num) > 0 else None
        if file_num == number and filename.endswith(file_ending): # TODO: This function
            return filename

class GeneratorDict(dict):
    def __init__(self, set_value_function, all_keys=None):
        self._has_been_filled_out = False
        self.all_keys = all_keys
        self.set_value_function = set_value_function # Takes in this dict and key to set, and sets the value in this dict at the key to the desired value (and can set other vals). If key is None, sets all possible values
                                                     # Returns True if key was successfully set, returns False if it wasn't

    def __getitem__(self, key):
        if key not in self:
            self.set_value_function(self, key)

        if key not in self:
            raise KeyError(f"Unable to find trace merger tree to snapshot with value {key}")

        return super().__getitem__(key)

    def values(self):
        if not self._has_been_filled_out: 
            self.set_value_function(self, None)
            self._has_been_filled_out = True
        return super().keys()
    
    def keys(self):
        if not self._has_been_filled_out: 
            self.set_value_function(self, None)
            self._has_been_filled_out = True
        return super().keys()

    def generated_keys(self):
        return super().keys()

    def generated_values(self):
        return super().values()


class HaloData(abc.ABC):
    """
    An abstract class who's children are meant to find, load, store, and provide halo data corresponding to some halo finder type (i.e. Amiga, Rockstar)
    """

    @abc.abstractmethod
    def __init__(self, snapshot, path, snapshot_value): ...
    """
    Loads the halo data at the given path and snapshot_value
    """

    @abc.abstractmethod
    def get_field(self, field, divide_by_h = True): ...
    """
    Returns the requested field in the loaded halo data corresponding
    """

    @staticmethod
    @abc.abstractmethod
    def get_younger_snapshot_value(snapshot_value):
        return snapshot_value - 1

    @staticmethod
    @abc.abstractmethod
    def get_older_snapshot_value(snapshot_value):
        return snapshot_value + 1

    @staticmethod
    @abc.abstractmethod
    def find_halo_data_file_path(parent_directory_path, snapshot_value, snapshot_value_kind='index'): ...
    """
    Returns the path of the halo data corresponding to the given snapshot_value, in the given parent directory
    """

    @staticmethod
    def find_merger_tree_file_path(parent_directory_path, snapshot_value_1, snapshot_value_2): raise NotImplementedError
    """
    Returns the path of the merger tree data file linking the given snapshot values, in the given directory. The order of the snapshot values should not matter
    """

    # _get_progenitor_ids need not be overwritten if get_progenitor_tree is overwritten to not call it
    @staticmethod
    def get_progenitor_ids(halo_id, file_path): raise NotImplementedError
    """
    Returns the ids of all immediate progenitors of the halo with id halo_id, given the path of a merger tree file linking the given halo's snapshot to its progenitor's
    """

    # _get_main_progenitor_id need not be overwritten if get_main_progenitor_line is overwritten to not call it
    @staticmethod
    def get_main_progenitor_id(halo_id, file_path): raise NotImplementedError
    """
    Returns the id of the immediate main progenitor of the halo with id halo_id, given the path of a merger tree file linking the given halo's snapshot to its progenitor's
    """

    # _get_descendant_id need not be overwritten if get_descendant_line is overwritten to not call it
    @staticmethod
    def get_descendant_id(halo_id, file_path, prioritize_most_contribution=False): raise NotImplementedError
    """
    Returns the id of the immediate descendant of the halo with id halo_id, given the path of a merger tree file linking the given halo's snapshot to its descendant's
    """

    # @staticmethod
    # def get_progenitor_tree(snapshot_of_halo, halo_id): raise NotImplementedError

    @classmethod
    def get_main_progenitor_line(cls, snapshot_of_halo, halo_id, merger_files_parent_directory_path):

        def set_value(this_dict, key):
            prev_snapshot = None
            cur_snapshot = snapshot_of_halo
            while True:
                prev_snapshot = cur_snapshot
                cur_snapshot = cls.get_younger_snapshot_value(cur_snapshot)
                if cur_snapshot not in this_dict:
                    prev_halo_id = this_dict[prev_snapshot]
                    merger_tree_file = cls.find_merger_tree_file_path(merger_files_parent_directory_path, cur_snapshot, prev_snapshot)

                    if merger_tree_file is None: return False

                    cur_halo_id = cls._get_main_progenitor_id(prev_halo_id, merger_tree_file)
                    this_dict[cur_snapshot] = cur_halo_id

                if cur_snapshot == key:
                    return True
             
        generator_dict = GeneratorDict(set_value)
        generator_dict[snapshot_of_halo] = halo_id

        return generator_dict
        
    @classmethod
    def get_descendant_line(cls, snapshot_of_halo, halo_id, merger_files_parent_directory_path, prioritize_most_contribution=False):
        
        def set_value(this_dict, key):
            prev_snapshot = None
            cur_snapshot = snapshot_of_halo
            while True:
                prev_snapshot = cur_snapshot
                cur_snapshot = cls.get_older_snapshot_value(cur_snapshot)
                if cur_snapshot not in this_dict:
                    prev_halo_id = this_dict[prev_snapshot]
                    merger_tree_file = cls.find_merger_tree_file_path(merger_files_parent_directory_path, cur_snapshot, prev_snapshot)

                    if merger_tree_file is None: return False

                    cur_halo_id = cls._get_main_progenitor_id(prev_halo_id, merger_tree_file)
                    this_dict[cur_snapshot] = cur_halo_id

                if cur_snapshot == key:
                    return True
             
        generator_dict = GeneratorDict(set_value)
        generator_dict[snapshot_of_halo] = halo_id

        return generator_dict

class AHFData(HaloData):
    def __init__(self, snapshot, path, snapshot_value):
        if not os.path.isfile(path):
            raise Exception(f"Could not find halo data file affiliated with snapshot {snapshot_value}")
        self.data = ascii.read(path)
        self.snapshot = snapshot

        self.h_fields = ['xc(6)', 'yc(7)', 'zc(8)', 'rvir(12)', 'mvir(4)', 'rmax(13)', 'mstar(65)', 'ngas(45)']

    def get_field(self, field, divide_by_h = True):
        """
        Get the values in the column of the specified field from the .AHF_halos file.

        Parameters:
        ----------
        field : string
            The name of the field.

        divide_by_h : Boolean
            If you would like to divide by h (Hubble Constant).
            Automatically set to True.
        
        Returns
        -------
        The list of values in that field.
        """
        
        # Get the correct name of the field
        field_name = str(field).lower()  # Convert field to string if it's an integer
        field_name = field_name.replace('_','')
        # Loop through all the field names
        for item in self.data.dtype.names:
            string = item.lower().replace('_','')
            if field_name in string:
                field_name = item
                break 

        if divide_by_h == True:
            h_query = str(field).lower() # Convert field to string if its an integer
            for h_field in self.h_fields:
                string = h_field.lower().replace('_','')
                if h_query in string:
                    # Desired field is one where we must divide by h before returning
                    # Perform division by h
                    column = self.data.field(field_name) / self.snapshot.h
                    return column
                
            # True was passed but the field is one where we do not divide by h
            column = self.data.field(field_name)
            return column
        else: 
            # Store all the field data in a list called column
            column = self.data.field(field_name) 
            # Return the column
            return column

    def field(self, field):
        return self.data.field(field)

    @staticmethod
    def find_halo_data_file_path(parent_directory_path, snapshot_value, snapshot_value_kind='index') -> Union[str, None]:
        parent_directory_contents = os.listdir(parent_directory_path)
        for filename in parent_directory_contents:
            file_num = get_all_ints_in_string(filename)
            file_num = file_num[0] if len(file_num) > 0 else None

            if file_num == snapshot_value and filename.endswith('.AHF_halos'): # TODO: This function
                return os.path.join(parent_directory_path, filename)
            
        return None
    
        # raise Exception(f"Could not find halo data file affiliated with snapshot {snapshot_value} (filename must contain {snapshot_name})")

    @staticmethod
    def find_merger_tree_file_path(parent_directory_path, snapshot_value_1, snapshot_value_2):
        parent_directory_contents = os.listdir(parent_directory_path)
        for filename in parent_directory_contents:
            nums_in_filename = get_all_ints_in_string(filename)

            if snapshot_value_1 in nums_in_filename and snapshot_value_2 in nums_in_filename:
                return os.path.join(parent_directory_path, filename)
            
        return None

    @staticmethod
    def get_progenitor_ids(halo_id, file_path):
        with open(file_path, "r") as file:
            file_contents = file.read()
            
        progenitor_ids = []
        saving_progenitors = False
        for line in file_contents.split("\n"):
            if len(line) == 0 or line[0] == "#":
                continue
            if line[0] != " ":
                if saving_progenitors:
                    break
                if int(line.split(" ")[0]) == halo_id:
                    saving_progenitors = True
            elif saving_progenitors:
                progenitor_ids.append(int(line.split("  ")[2]))

        return progenitor_ids

    @staticmethod
    def get_main_progenitor_id(halo_id, file_path):
        with open(file_path, "r") as file:
            file_contents = file.read()

        for line in file_contents.split("\n"):
            if len(line) == 0 or line[0] == "#":
                continue

            if int(line.strip().split(" ")[0]) == halo_id:
                return line.strip().split(" ")[-1]

    @staticmethod
    def get_descendant_id(halo_id, file_path, prioritize_most_contribution=False):
        with open(file_path, "r") as file:
            file_contents = file.read()

        child_ids = []
        current_child_id = None
        for line in file_contents.split("\n"):
            if len(line) == 0 or line[0] == "#":
                continue

            if line[0] != " ":
                current_child_id = int(line.split(" ")[0])
            else:
                if int(line.split("  ")[2]) == halo_id:
                    parts_contributed = int(line.split(" ")[2])
                    child_ids.append((current_child_id, parts_contributed))
                    if not prioritize_most_contribution:
                        break

        if len(child_ids) == 1:
            return child_ids[0][0]
        elif len(child_ids) == 0:
            return None
        else:
            # Return child ID with highest corresponding parts_contributed
            return max(child_ids, key=lambda child: child[1])[0]

    @staticmethod
    def get_descendant_line():
        ...

    @staticmethod
    def get_main_progenitor_line():
        ...

class RockstarData(HaloData):
    ...

class HaloFinderTypes:
    ahf = AHFData
    rockstar = RockstarData


class Snapshot:
    """
    A class containing the information for one snapshot of a simulation, including the snapshot's particles and halo data (if specified)
    """

    # CLASS_TYPE is defined so that code may be designed that takes in either a Snapshot or a Simulation, and it can check which one is passed in
    CLASS_TYPE = Halo.ClassType.SNAPSHOT

    def __init__(self, 
                 sim, 
                 snapshot_value, 
                 simulation_directory, 
                 snapshot_directory, 
                 species, 
                 snapshot_value_kind="index", 
                 halo_finder_type: HaloData = AHFData, 
                 halo_data_file_path=None,
                 merger_tree_file_path=None):
        # The following private values are initially None, and only loaded once they are accessed through their getters.
        self._particles = None
        self._halo_data = None
        self._h = None

        self.sim = sim
        self.snapshot_value = snapshot_value
        self.simulation_directory = simulation_directory
        self.snapshot_directory = snapshot_directory
        self.species = species
        self.snapshot_value_kind = snapshot_value_kind
        self.halo_finder_type = halo_finder_type
        self.halo_data_file_path = halo_data_file_path
        self.merger_tree_file_path = merger_tree_file_path

    def get_next_snapshot_value(self):
        return self.snapshot_value + 1
    
    def get_prev_snapshot_value(self):
        return self.snapshot_value - 1

    @property
    def particles(self):
        # If particles have not yet been loaded, load them in and return the value. Otherwise, return the loaded value.
        if self._particles is None:
            self._particles = gizmo.io.Read.read_snapshots(
                simulation_directory = self.simulation_directory,
                snapshot_directory = self.snapshot_directory, 
                species=self.species, 
                snapshot_value_kind=self.snapshot_value_kind,
                snapshot_values=self.snapshot_value
            )

        return self._particles
    
    @property
    def halo_data(self) -> HaloData:
        # If halo data has not yet been loaded, load them in and return the value. Otherwise, return the loaded value.
        if self._halo_data is None:
            if self.halo_data_file_path is None:
                raise ValueError("Halo data file for this snapshot was either not found or not inputted.")
            if self.halo_finder_type is None:
                raise ValueError("No halo finder type was defined.")
            self._halo_data = self.halo_finder_type(self, self.halo_data_file_path, self.snapshot_value)

        return self._halo_data
    
    def get_field(self, field):
        # Get the requested field from this snapshot's halo data
        return self.halo_data.get_field(field)
    
    def get_halo(self, halo_id, restrict_percentage: Union[float, int, None] = 100):
        # Get the halo from this snapshot with ID halo_id
        return Halo.Halo(self, halo_id, restrict_percentage)
    
    def get_particle_group(self):
        # Get a ParticleGroup containing all particles 
        return Halo.ParticleGroup(self)

    @property
    def h(self):
        # If h has not yet been loaded, load them in and return the value. Otherwise, return the loaded value.
        if self._h is None:
            self._h = gizmo.io.Read.read_header(
                simulation_directory = self.simulation_directory,
                snapshot_directory = self.snapshot_directory,
                snapshot_value_kind = self.snapshot_value_kind,
                snapshot_value = self.snapshot_value
            )['hubble']

        return self._h

class Simulation:
    """
    A class containing one or more Snapshots, with methods corresponding to those in Snapshot to access their information and functionality
    If only one snapshot is defined, snapshot-specific methods of this class may be called just like they would be on a snapshot (without specifying a snapshot value)
    But if more than one snapshots are defined, the ID of the relevant snapshot must be passed in or else an error will be raised
    """

    # CLASS_TYPE is defined so that code may be designed that takes in either a Snapshot or a Simulation, and it can check which one is passed in
    CLASS_TYPE = Halo.ClassType.SIMULATION

    def __init__(self,
                 simulation_directory_name: str = None,
                 simulation_directory_path: str = None,
                 species: Union[str, list[str]] = ['star'],
                 snapshot_directory_path_in_simulation: str = "output",
                 halo_data_directory_path_in_simulation: str = None,
                 halo_data_file_path: Union[str, list[str]] = None,
                 halo_finder_type: HaloData = HaloFinderTypes.ahf,
                 snapshot_value_kind: Literal['index', 'redshift', 'scalefactor', 'time'] = 'index',
                 snapshot_values: Union[int, list[int]] = 600):

        self.snapshots = []
        self.halo_datas = []

        if isinstance(snapshot_values, int):
            snapshot_values = [snapshot_values]

        # If a simulation name has been given, we can assume the user is using the conventional locations
        if simulation_directory_name is not None:
            simulation_directory_path = f'../data/{simulation_directory_name}'
        elif simulation_directory_path is None:
            raise ValueError("Must define simulation_name or simulation_directory_path.")
        
        halo_data_file_paths = {}
        # Look for the file that ends with '.AHF_halos'.
        if halo_data_file_path is not None:
            # If halo_data_file_path is given, then just directly use that/those path(s)
            if isinstance(halo_data_file_path, str):
                halo_data_file_path = [halo_data_file_path]

            if isinstance(halo_data_file_path, list):
                if len(halo_data_file_path) != len(snapshot_values):
                    raise ValueError("Must input the same amount of halo data file paths as you do snapshot values if inputting an array of halo data file paths")
                for i, snapshot_value in enumerate(snapshot_values):
                    halo_data_file_paths[snapshot_value] = halo_data_file_path[i]
            elif isinstance(halo_data_file_path, dict):
                halo_data_file_paths = halo_data_file_path
            else:
                raise TypeError("halo_data_file_path must be of type str, list, dict, or NoneType")

        else:
            # Otherwise, use the specified halo finder's class to find the path(s) of the halo data file(s) affiliated with the specified snapshot value
            halo_data_directory = simulation_directory_path if halo_data_directory_path_in_simulation is None else os.path.join(simulation_directory_path, halo_data_directory_path_in_simulation)
            for snapshot_value in snapshot_values:
                found_path = halo_finder_type.find_halo_data_file_path(halo_data_directory, snapshot_value, snapshot_value_kind)
                if found_path is not None: halo_data_file_paths[snapshot_value] = found_path

        self.snapshots = {snapshot_value: Snapshot(
            self,
            snapshot_value,
            simulation_directory_path,
            snapshot_directory_path_in_simulation,
            species,
            snapshot_value_kind,
            halo_finder_type,
            halo_data_file_paths[snapshot_value] if snapshot_value in halo_data_file_paths else None
        ) for snapshot_value in snapshot_values}
            
        """
        if simulation_directory is None or ahf_path is None:
            print('Cannot read files. Either:\n')
            print('\t1) Provide a simulation_name while adhering to the proper folder structure.')
            print('\t\tExample:  sim = Simulation("m10r_res250md")')
            print('\t2) Manually specify: simulation_directory and ahf_directory. Also, specify the snapshot directory if it is not output.')
            print('\t\tExample:  sim = Simulation(simulation_directory="path", ahf_path="path")')
            print('\t\tExample:  sim = Simulation(simulation_directory="path", ahf_path="path", snapshot_directory="path")\n')
            if simulation_directory is None:
                print('Missing simulation directory.') 
            if ahf_path is None:
                print('Missing ahf_path.') 
            return
        """

    def get_snapshot(self, snapshot_value=None):
        # Return the snapshot with the given ID. If no snapshot value is passed in and there is only one snapshot defined, return that snapshot
        # If no snapshot value is passed in and there are multiple snapshots defined, raise an error
        if snapshot_value is None:
            if len(self.snapshots) == 1:
                return list(self.snapshots.values())[0]
            else:
                raise IndexError("Multiple snapshots are stored in simulation; unable to access singular attribute.")
        else:
            return self.snapshots[snapshot_value]

    @property
    def snapshot(self):
        # If only one snapshot is defined, return that snapshot. Else, raise an error
        return self.get_snapshot()
    
    def get_particles(self, snapshot_value=None):
        # Return the particles in the snapshot corresponding to the ID passed in. If none is passed in and only one snapshot is defined, return the particles from that snapshot (else raise an error)
        return self.get_snapshot(snapshot_value).particles
    
    @property
    def particles(self):
        # If only one snapshot is defined, return the particles of that snapshot. Else, raise an error
        return self.snapshot.particles
    
    def get_halo_data(self, snapshot_value=None):
        # Return the halo data in the snapshot corresponding to the ID passed in. If none is passed in and only one snapshot is defined, return the halo data from that snapshot (else raise an error)
        return self.get_snapshot(snapshot_value).halo_data
    
    @property
    def halo_data(self):
        return self.snapshot.halo_data

    def get_halo(self, halo_id, snapshot_value=None, restrict_percentage: Union[float, int, None] = 100):
        # Return the halo of ID halo_id in the snapshot corresponding to the ID passed in. If none is passed in and only one snapshot is defined, return the halo from that snapshot (else raise an error)
        return self.get_snapshot(snapshot_value).get_halo(halo_id, restrict_percentage)

    def get_particle_group(self, snapshot_value=None):
        # Return a particle group containing all particles in the snapshot corresponding to the ID passed in. If none is passed in and only one snapshot is defined, return a particle group from that snapshot (else raise an error)
        return self.get_snapshot(snapshot_value).get_particle_group()

    def get_field(self, field, snapshot_value=None):
        # Return the requested field in the snapshot corresponding to the ID passed in. If none is passed in and only one snapshot is defined, return the field from that snapshot (else raise an error)
        return self.get_snapshot(snapshot_value).get_field(field)