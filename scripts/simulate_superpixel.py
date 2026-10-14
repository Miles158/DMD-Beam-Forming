import numpy as np
import matplotlib.pyplot as plt


from src import superpixel as px
from src import propagation as prop

wavelength = 473e-9 # m


# DMD parameters 
mirror_pitch = 7.56e-6

mirror_size_px = (1080, 1920)  # (ny, nx) = (vert, horiz)
mirror_size_m = (mirror_pitch * mirror_size_px[0], mirror_pitch * mirror_size_px[1])  # (y, x) in meters


superpixel_dimension = 3 
superpixel_size_m = (superpixel_dimension * mirror_pitch, superpixel_dimension * mirror_pitch)

# relay parameters
f1 = 0.3  # m
f2 = 0.1  # m

# Note on Lens 1 clear aperture:
# Light from a mirror at x0 reaching the aperture crosses lens 1 at x0 + aperture_position.
# From the furthest DMD corner, this is up to ~16 mm off-axis, which exceeds a 1-inch lens
# radius (12.7 mm). Use None (unclipped) or >= 2-inch lens (25.4e-3 m) to avoid corner dimming.
lens1_clear_radius = None  # 12.7e-3 m (1-inch) will clip DMD corners

L = wavelength * f1 / mirror_pitch  # m

aperture_x = -L / (superpixel_dimension**2)  
aperture_y = -L / superpixel_dimension       
aperture_position_xy = (aperture_x, aperture_y)  


max_aperture_radius = L / (4 * superpixel_dimension)  # m
real_aperture_radius = 0.75 * max_aperture_radius     # chosen arbitrarily

print(px.pixel_phases(superpixel_dimension))


samples_per_mirror = 2
num_pix = 2 * mirror_size_px[1] * samples_per_mirror  
grid_shape = (num_pix, num_pix)  # (ny, nx)

dy = mirror_pitch / samples_per_mirror
dx = mirror_pitch / samples_per_mirror
grid_spacing = (dy, dx)  # (dy, dx) in meters for propagation.py
grid_extent_m = (num_pix * dy, num_pix * dx)  # (y, x) in meters