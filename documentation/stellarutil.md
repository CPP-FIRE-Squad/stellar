# Table of Contents
- Halo Class
    - [center_on](#center_on)
    - [init Halo](#init-halo): Initialize a new Halo object.
    - [restrict_percentage](#restrict_percentage)
    - [restrict_slice](#restrict_slice)
- Simulation Class
    - [get_field](#get_field): Get the values in the column of the specified field from the .AHF_halos file.
    - [get_halo](#get_halo): Get the indicated dark matter halo.
    - [help](#help): Recieve help.
    - [init Simulation](#init-simulation): Initialize a new Simulation object.
    - [print_halo_fields](#print_halo_fields)
    - [print_menu](#print_menu)
- Star Class
    - [get_2DR](#get_2dr): Get the 2d radius of the star from the center of the halo.
    - [get_3DR](#get_3dr): Get the 3d radius of the star from the center of the halo.
    - [get_velocity](#get_velocity): Get the velocity of the star by calculating the magnitude of the velocity vector.
    - [init Star](#init-star): Initialize a new Star object.
---
## center_on
---
## init Halo

### Description:
Initialize a new Halo object.


### Parameters:


| Name | Type | Description |
| --- | --- | --- |
| simulation | Simulation |     The simulation the halo comes from. | required |
| id | integer |     The id of the halo. | required |
| stars | Stars list |     The list of stars in the halo. | required |
| xc | float |     The center x position. | required |
| yc | float |     The center y position. | required |
| zc | float |     The center z position. | required |
| hostID | integer |     The id of the parent halo. | required |
| mass | float |     The mass of the halo. | required |
| radius | float |     The radius of the halo. | required |
| rMax | float |     The max radius of the halo. | required |
| vMax | float |     The max velocity of the halo. | required |
| vEsc | float |     The escape velocity of the halo. | required |
| numGas | integer |     The number of gas particles. | required |
| gasMass | float |     The mass of gas particles. | required |
| numStars | integer |     The number of star particles. | required |
| gasMass | float |     The mass of gas particles. | required |
---
## restrict_percentage
---
## restrict_slice
---
## get_field

### Description:
Get the values in the column of the specified field from the .AHF_halos file.


### Parameters:


| Name | Type | Description |
| --- | --- | --- |
| field | string |     The name of the field. | required |
---
## get_halo

### Description:
Get the indicated dark matter halo.


### Parameters:


| Name | Type | Description |
| --- | --- | --- |
| id | int |     The index of the dark matter halo. Default is 0. | required |
---
## help

### Description:
Recieve help.

---
## init Simulation

### Description:
Initialize a new Simulation object.


### Parameters:


| Name | Type | Description |
| --- | --- | --- |
| simulation_name | string |     The name of the simulation. <br />    By giving the name, it will look for simulation_directory/snapshot_directory/ahf_directory in '../data/{simulation_name}' | required |
| simulation_directory | string |     The path to the .hdf5 file.  | required |
| snapshot_directory | string |     The path to the snapshot_times.txt.  | required |
| ahf_path | string |     The path to the .AHF_halos file. | required |
| species | list |     name[s] of particle species:<br />        'all' = all species in file<br />        'dark' = dark matter at highest resolution<br />        'dark2' = dark matter at lower resolution<br />        'gas' = gas<br />        'star' = stars<br />        'blackhole' = black holes, if snapshot contains them | required |
| snapshot_values | int or float or list |     index[s] or redshift[s] or scale-factor[s] of snapshot[s] | required |
---
## print_halo_fields
---
## print_menu
---
## get_2DR

### Description:
Get the 2d radius of the star from the center of the halo.

---
## get_3DR

### Description:
Get the 3d radius of the star from the center of the halo.

---
## get_velocity

### Description:
Get the velocity of the star by calculating the magnitude of the velocity vector.

---
## init Star

### Description:
Initialize a new Star object.


### Parameters:


| Name | Type | Description |
| --- | --- | --- |
| x | float |     The x position of the star. | required |
| y | float |     The y position of the star. | required |
| z | float |     The z position of the star. | required |
| m | float |     The mass of the star. | required |
| a | float |     The scale factor of the star. | required |
| vx | float |     The x velocity of the star. | required |
| vy | float |     The y velocity of the star. | required |
| vz | float |     The z velocity of the star. | required |
