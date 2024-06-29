class Halo:

    def __init__(self, simulation, index, stars, gasses, darks, center, velocity):
        """
        Initialize a new Halo object.

        Parameters:
        ----------
        simulation : Simulation
            The simulation the halo comes from.
        id : integer
            The id of the halo.
        stars : Stars list
            The list of stars in the halo.
        xc : float
            The center x position.
        yc : float
            The center y position.
        zc : float
            The center z position.
        hostID : integer
            The id of the parent halo.
        mass : float
            The mass of the halo.
        radius : float
            The radius of the halo.
        rMax : float
            The max radius of the halo.
        vMax : float
            The max velocity of the halo.
        vEsc : float
            The escape velocity of the halo.
        numGas : integer
            The number of gas particles.
        gasMass : float
            The mass of gas particles.
        numStars : integer
            The number of star particles.
        gasMass : float
            The mass of star particles.
        """
        self.simulation = simulation
        self.index = index
        self.stars = stars
        self.gasses = gasses
        self.darks = darks
        self.xc = center[0]
        self.yc = center[1]
        self.zc = center[2]
        self.vxc = velocity[0]
        self.vyc = velocity[1]
        self.vzc = velocity[2]
        # Grab some more metadata for the halo
        self.hostID = simulation.ahf_data.field('hostHalo(2)')[index]
        self.mass = simulation.get_field('4')[index]
        self.rMax = simulation.ahf_data.field('Rmax(13)')[index] / simulation.h
        self.vMax = simulation.ahf_data.field('Vmax(17)')[index]
        self.vEsc = simulation.ahf_data.field('v_esc(18)')[index]
        self.numGas = simulation.ahf_data.field('n_gas(44)')[index]
        self.gasMass = simulation.ahf_data.field('M_gas(45)')[index]
        self.numStars = simulation.ahf_data.field('n_star(64)')[index]
        self.starMass = simulation.ahf_data.field('M_star(65)')[index]

    def restrict_percentage(self, percentage = 15):
        # Get the radius of the galaxy that can actually hold stars
        # Rhalo, Mhalo, Vhalo <-> Rvir, Mvir, Vvir
        rgal = (percentage / 100.0) * self.simulation.get_field('12')[self.id]
        # Get all the stars and center on the given halo
        x = self.simulation.particles['star']['position'][:,0] - self.xc
        y = self.simulation.particles['star']['position'][:,1] - self.yc
        z = self.simulation.particles['star']['position'][:,2] - self.zc
        a = self.simulation.particles['star']['form.scalefactor']
        m = self.simulation.particles['star']['mass']
        vx = self.simulation.particles['star']['velocity'][:,0] - self.vxc
        vy = self.simulation.particles['star']['velocity'][:,1] - self.vyc
        vz = self.simulation.particles['star']['velocity'][:,2] - self.vzc
        # Get the distance of each star from the center of the indicated dark matter halo
        distances =  np.sqrt(np.square(x) + np.square(y) + np.square(z))
        # Filter out all stars that are too far away 
        x_gal = x[distances < rgal]
        y_gal = y[distances < rgal]
        z_gal = z[distances < rgal]
        a_gal = a[distances < rgal]
        m_gal = m[distances < rgal]
        vx_gal = vx[distances < rgal]
        vy_gal = vy[distances < rgal]
        vz_gal = vz[distances < rgal]
        # Create a new stars list
        new_stars = []
        for i in range(len(x_gal)):
            star = Star(x_gal[i], y_gal[i], z_gal[i], m_gal[i], a_gal[i], vx_gal[i], vy_gal[i], vz_gal[i])
            new_stars.append(star)
        # Update the halos star list
        self.stars = new_stars

    def restrict_slice(self, face = 'xy', proj_distance = 1, thickness = 1):
        x = self.simulation.particles['star']['position'][:,0] - self.xc
        y = self.simulation.particles['star']['position'][:,1] - self.yc
        z = self.simulation.particles['star']['position'][:,2] - self.zc
        a = self.simulation.particles['star']['form.scalefactor']
        m = self.simulation.particles['star']['mass']
        vx = self.simulation.particles['star']['velocity'][:,0] - self.vxc
        vy = self.simulation.particles['star']['velocity'][:,1] - self.vyc
        vz = self.simulation.particles['star']['velocity'][:,2] - self.vzc
        # Get sqrt of the given face
        roots = []
        for i in range(len(x)):
            if face == 'xy' or face == 'yx': roots.append( (x**2 + y**2) ** .5 )
            elif face == 'xz' or face == 'zx': roots.append( (x**2 + z**2) ** .5 )
            elif face == 'yz' or face == 'zy': roots.append( (y**2 + z**2) ** .5 )
        # restrict all stars whose: sqrt(×^2 + y^2) < proj_distance
        x_gal = x[roots < proj_distance]
        y_gal = y[roots < proj_distance]
        z_gal = z[roots < proj_distance]
        a_gal = a[roots < proj_distance]
        m_gal = m[roots < proj_distance]
        vx_gal = vx[roots < proj_distance]
        vy_gal = vy[roots < proj_distance]
        vz_gal = vz[roots < proj_distance]
        # Create a new stars list
        new_stars = []
        for i in range(len(x_gal)):
            star = Star(x_gal[i], y_gal[i], z_gal[i], m_gal[i], a_gal[i], vx_gal[i], vy_gal[i], vz_gal[i])
            # restrict all stars whose |z pos| < thickness
            if face == 'xy' or face == 'yx':
                if abs(z_gal[i]) < thickness: new_stars.append(star)
            elif face == 'xz' or face == 'zx':
                if abs(y_gal[i]) < thickness: new_stars.append(star)
            elif face == 'yz' or face == 'zy':
                if abs(x_gal[i]) < thickness: new_stars.append(star)
        # Update the halos star list
        self.stars = new_stars

    def center_on(self, otherID):
        # Get the center relative to the halo at the given index
        xc = (self.simulation.get_field('Xc(6)')[self.id]) - (self.simulation.get_field('Xc(6)')[otherID])
        yc = (self.simulation.get_field('Yc(7)')[self.id]) - (self.simulation.get_field('Yc(7)')[otherID])
        zc = (self.simulation.get_field('Zc(8)')[self.id]) - (self.simulation.get_field('Zc(8)')[otherID])
        # Recenter each star in the list
        for star in self.stars:
            star.x -= xc
            star.y -= yc
            star.z -= zc