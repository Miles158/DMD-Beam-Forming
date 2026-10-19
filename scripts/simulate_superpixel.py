import numpy as np
import matplotlib.pyplot as plt
import torch

from src import superpixel as px
from src import propagation as prop
from src import helpers
from src import models 
from src import plotting

device = "mps"

wavelength = 473e-9 # m


# DMD parameters 
mirror_pitch = 7.56e-6

mirror_size_px = (1080, 1920)  # (ny, nx) = (vert, horiz)
mirror_size_m = (mirror_pitch * mirror_size_px[0], mirror_pitch * mirror_size_px[1])  # (y, x) in meters


superpixel_dimension = 3 
superpixel_size_m = (superpixel_dimension * mirror_pitch, superpixel_dimension * mirror_pitch)

# relay parameters
f1 = 0.3  # m
f2 = 0.150  # m

# Note on Lens 1 clear aperture:
# Light from a mirror at x0 reaching the aperture crosses lens 1 at x0 + aperture_position.
# From the furthest DMD corner, this is up to ~16 mm off-axis, which exceeds a 1-inch lens
# radius (12.7 mm). Use None (unclipped) or >= 2-inch lens (25.4e-3 m) to avoid corner dimming.
lens1_clear_radius = None  # 12.7e-3 m (1-inch) will clip DMD corners

L = wavelength * f1 / mirror_pitch  # m

aperture_y = -L / superpixel_dimension       
aperture_x = -L / (superpixel_dimension**2)  
aperture_position = (aperture_y, aperture_x) 


max_aperture_radius = L / (4 * superpixel_dimension)  # m
real_aperture_radius = 0.75 * max_aperture_radius     # chosen arbitrarily

print(max_aperture_radius)
print(px.pixel_phases(superpixel_dimension))


samples_per_mirror = 2
num_pix = 2 * mirror_size_px[1] * samples_per_mirror  
grid_shape = (num_pix, num_pix)  # (ny, nx)

dy = mirror_pitch / samples_per_mirror
dx = mirror_pitch / samples_per_mirror
grid_spacing = (dy, dx)  # (dy, dx) in meters for propagation.py
grid_extent_m = (num_pix * dy, num_pix * dx)  # (y, x) in meters





# generate a sample target function
target = helpers.generate_LG10_donut(mirror_size_px, waist=60, superpixel_dimension=superpixel_dimension)


# encode
commands, fields = px.build_superpixel_fields(superpixel_dimension, px.pixel_phases(superpixel_dimension))
dmd_commands = px.encode(target, superpixel_dimension, commands, fields)         # (360, 640, 9)
mirror_image = px.to_image(dmd_commands, superpixel_dimension)                   # (1080, 1920), 0/1

print(dmd_commands.shape)

# ---------------------------------------------------------------- reference field
# Ideal modulator: the target value on every mirror of its superpixel, times a phase gradient that
# steers all its light into the aperture. It goes through the same optics, so flip, magnification,
# position and aperture filtering match automatically.
# Sign: light from a mirror at x arrives through the aperture weighted by exp(+i*phase(x)) (that is
# pixel_phases), so a field must carry exp(-i*phase(x)) to come out with the target's own phase.
ideal_per_mirror = np.repeat(np.repeat(target, superpixel_dimension, axis=0), superpixel_dimension, axis=1)   # (1080, 1920)

nx, ny = grid_shape
dy, dx = grid_spacing
y = (torch.arange(ny, device=device) - (ny - 1) / 2) * dy
x = (torch.arange(nx, device=device) - (nx - 1) / 2) * dx 
yy, xx = torch.meshgrid(y, x, indexing="ij")

phase_gradient = torch.exp(-2j * torch.pi * (xx / (superpixel_dimension**2 * mirror_pitch) + yy / (superpixel_dimension * mirror_pitch)))
del yy, xx


# simulate
model = models.Relay4f(grid_spacing, wavelength, f1, f2,
                      aperture_center=aperture_position, aperture_radius=real_aperture_radius,
                      lens1_clear_radius=lens1_clear_radius, lens2_clear_radius=lens1_clear_radius)

with torch.no_grad():
    out_dmd = model(helpers.to_propagation_grid(mirror_image, samples_per_mirror, grid_shape, device))
    out_ref = model(helpers.to_propagation_grid(ideal_per_mirror, samples_per_mirror, grid_shape, device) * phase_gradient)
del phase_gradient

print(f"fidelity vs bandwidth-limited target: {models.fidelity(out_ref, out_dmd):.5f}")


#  plots
plotting.plot_ideal_vs_dmd(
    out_ref=out_ref,
    out_dmd=out_dmd,
    aperture_position=aperture_position,
    grid_spacing=grid_spacing,
    mirror_size_m=mirror_size_m,
    f1=f1,
    f2=f2,
    save_path="ideal_vs_dmd.png",
)

plotting.plot_dmd_actuators(
    actuators=dmd_commands, 
    mirror_pitch=mirror_pitch,
    superpixel_dimension=superpixel_dimension,
    unit="mm",
    save_path="actuator_commands",
)

