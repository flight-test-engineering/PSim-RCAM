    
import srtm
import numpy as np

# 1. Initialize the data handler (Do this ONCE, before the loop)
elevation_data = srtm.get_data()

# 2. Get elevation (Do this inside the loop)
# Returns meters. Returns None if data is missing (e.g., over ocean)

INIT_LATLON_DEG = np.array([47.2548, 11.2963]) 
latitude = INIT_LATLON_DEG[0]
longitude = INIT_LATLON_DEG[1]
ground_alt = elevation_data.get_elevation(latitude, longitude)

if ground_alt is None:
    ground_alt = 0.0 # Default to Sea Level over oceans
    print('DEBUG NO DEM DATA')

print(f'{ground_alt=}')
