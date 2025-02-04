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

def get_first_full_int_in_string(string):
    """
    Returns the first full integer that appears in the passed in string. I.e. passing in "abc123def456" returns 123. If no number is found, returns None
    """
    num = ""
    for char in string:
        if char.isdigit():
            num += char
        else:
            if len(num) > 0:
                break

    return int(num) if len(num) > 0 else None

def find_file_containing_number(filenames, number, file_ending):
    """
    For all the filenames passed in, finds the first one that has a given number in it (and ends in the string file_ending)
    """
    # snapshot_name = f"snapshot_{snapshot_value}" # TODO: Fill with zeros
    for filename in filenames:
        file_num = get_first_full_int_in_string(filename)
        if file_num == number and filename.endswith(file_ending): # TODO: This function
            return filename

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

    @classmethod
    @abc.abstractmethod
    def find_halo_data_file_path(cls, directory, snapshot_value, snapshot_value_kind='index'): ...
    """
    Returns the path of the halo data corresponding to the given snapshot_value, in the given directory
    """

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

    @classmethod
    def find_halo_data_file_path(cls, parent_directory_path, snapshot_value, snapshot_value_kind='index') -> Union[str, None]:
        parent_directory_contents = os.listdir(parent_directory_path)
        filename = find_file_containing_number(parent_directory_contents, snapshot_value, '.AHF_halos')
        return os.path.join(parent_directory_path, filename) if filename is not None else None
    
        # raise Exception(f"Could not find halo data file affiliated with snapshot {snapshot_value} (filename must contain {snapshot_name})")

class HaloFinderTypes:
    ahf = AHFData
    # rockstar = 


class Snapshot:
    """
    A class containing the information for one snapshot of a simulation, including the snapshot's particles and halo data (if specified)
    """

    # CLASS_TYPE is defined so that code may be designed that takes in either a Snapshot or a Simulation, and it can check which one is passed in
    CLASS_TYPE = Halo.ClassType.SNAPSHOT

    def __init__(self, sim, snapshot_value, simulation_directory, snapshot_directory, species, snapshot_value_kind="index", halo_finder_type: HaloData = AHFData, halo_data_file_path=None):
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