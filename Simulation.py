import os, gizmo_analysis as gizmo, astropy.io.ascii as ascii, numpy as np, Halo, Particle

class Simulation:

    def __init__(
            self, 
            simulation_name = None,
            simulation_directory = None, 
            snapshot_directory = 'output',
            ahf_path = None, 
            species = ['star'], 
            snapshot_value_kind='index',
            snapshot_values = 600
        ):
        """
        Initialize a new Simulation object.

        Parameters:
        ----------
        simulation_name : string
            The name of the simulation. 
            By giving the name, it will look for simulation_directory/snapshot_directory/ahf_directory in '../data/{simulation_name}'
        simulation_directory : string
            The path to the .hdf5 file. 
        snapshot_directory : string
            The path to the snapshot_times.txt. 
        ahf_path : string
            The path to the .AHF_halos file.
        species : list
            name[s] of particle species:
                'all' = all species in file
                'dark' = dark matter at highest resolution
                'dark2' = dark matter at lower resolution
                'gas' = gas
                'star' = stars
                'blackhole' = black holes, if snapshot contains them
        snapshot_values : int or float or list
            index[s] or redshift[s] or scale-factor[s] of snapshot[s]

        Attributes:
        -----------
        h : float
            The hubble constant.
        h_fields : list
            List containing the fields that require division by h (Hubble constant)
        particles : float
            The data for all the indicated particles in the simulation.
        ahf_data : float
            The data within the .AHF_halos file.
        """
        # If a simulation name has been given, we can assume the user is using the conventional locations
        if simulation_name is not None:
            simulation_directory = f'../data/{simulation_name}'
            # Look for the file that ends with '.AHF_halos'.
            items = os.listdir(simulation_directory)
            for item in items:
                file_path = os.path.join(simulation_directory, item)
                if not os.path.isdir(file_path) and item.endswith('.AHF_halos'):
                    print(file_path)
                    ahf_path = file_path
            if ahf_path is None:
                print(f'Could not find an ahf_directory in: {simulation_directory}')
                return
        elif simulation_directory is not None and ahf_path is None:
            # Look for the file that ends with '.AHF_halos'.
            items = os.listdir(simulation_directory)
            for item in items:
                file_path = os.path.join(simulation_directory, item)
                if not os.path.isdir(file_path) and item.endswith('.AHF_halos'):
                    ahf_path = file_path
                    print('Found AHF file here: ' + ahf_path)
                    break
            if ahf_path is None:
                print(f'Could not find an ahf_directory in: {simulation_directory}')
                return
        else:
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
            
        # Snpashot value is used to get the hubble constant, it will always be a subset of the snapshot_values
        snapshot_value = snapshot_values[0] if type(snapshot_values) is list else snapshot_values
        # Get the hubble constant from gizmo_analysis
        self.h = gizmo.io.Read.read_header(
            simulation_directory = simulation_directory,
            snapshot_directory = snapshot_directory,
            snapshot_value_kind = snapshot_value_kind,
            snapshot_value = snapshot_value
        )['hubble']
        # Initialize fields that require division by h (Hubble constant)
        self.h_fields = ['xc(6)', 'yc(7)', 'zc(8)', '12', '4', 'rmax(13)', 'mstar(65)', 'ngas(45)']
        # Get the particles from gizmo_analysis
        self.particles = gizmo.io.Read.read_snapshots(
            simulation_directory = simulation_directory,
            snapshot_directory = snapshot_directory,
            species=species, 
            snapshot_value_kind=snapshot_value_kind,
            snapshot_values=snapshot_values
        )
        # Get the AHF data from the halo file
        self.ahf_data = ascii.read(ahf_path)
        # Filter the AHF data
        self.ahf_data = self.ahf_data[(self.ahf_data.field('fMhires(38)') > 0.99)]
    
    def get_halo(self, index = 0):
        # Get the center of the indicated dark matter halo
        xc = self.ahf_data.field('Xc(6)')[index] / self.h
        yc = self.ahf_data.field('Yc(7)')[index] / self.h
        zc = self.ahf_data.field('Zc(8)')[index] / self.h
        halo_center = np.array([xc, yc, zc])
        # Get the peculiar velocity of the indicated dark matter halo
        vxc = self.ahf_data.field('VXc(9)')[index] / self.h
        vyc = self.ahf_data.field('VYc(10)')[index] / self.h
        vzc = self.ahf_data.field('VZc(11)')[index] / self.h
        halo_vel = np.array([vxc, vyc, vzc])
        # Get the particles
        stars = Particle.get_particles(self,'star', index, halo_center, halo_vel)
        gasses = Particle.get_particles(self,'gas', index, halo_center, halo_vel)
        darks = Particle.get_particles(self,'dark', index, halo_center, halo_vel)

        # Return the indicated dark matter halo
        return Halo(self, index, stars, gasses, darks, halo_center, halo_vel)

    def get_field(self, field, divide_h = True):
        """
        Get the values in the column of the specified field from the .AHF_halos file.

        Parameters:
        ----------
        field : string
            The name of the field.

        divide_h : Boolean
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
        for item in self.ahf_data.dtype.names:
            string = item.lower().replace('_','')
            if field_name in string:
                field_name = item
                break 

        if divide_h == True:
            h_query = str(field).lower() # Convert field to string if its an integer
            for h_field in self.h_fields:
                string = h_field.lower().replace('_','')
                if h_query in string:
                    # Desired field is one where we must divide by h before returning
                    # Perform division by h
                    column = self.ahf_data.field(field_name) / self.h
                    return column
                
            # True was passed but the field is one where we do not divide by h
            column = self.ahf_data.field(field_name)
            return column
        else: 
            # Store all the field data in a list called column
            column = self.ahf_data.field(field_name) 
            # Return the column
            return column
    