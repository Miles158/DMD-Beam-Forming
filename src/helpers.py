import numpy as np
import torch

def generate_LG10_donut(mirror_size_px, superpixel_dimension, waist=60):

    sp_rows, sp_cols = mirror_size_px[0] // superpixel_dimension, mirror_size_px[1] // superpixel_dimension   # (360, 640) superpixels
    yy_sp, xx_sp = np.mgrid[0:sp_rows, 0:sp_cols].astype(float)
    yy_sp -= (sp_rows - 1) / 2
    xx_sp -= (sp_cols - 1) / 2
    waist = 60.0                                                       # in superpixels
    r_sp = np.hypot(xx_sp, yy_sp)
    target = (r_sp / waist) * np.exp(-(r_sp / waist)**2) * np.exp(1j * np.arctan2(yy_sp, xx_sp))

    return target


def generate_plane_wave(mirror_size_px, superpixel_dimension, phase=0.0):
    """Uniform complex plane-wave target at the superpixel resolution.

    When encoded with the superpixel method, this produces a regular grating on the DMD
    whose first diffraction order defines the exact aperture centre.
    """
    sp_rows = mirror_size_px[0] // superpixel_dimension
    sp_cols = mirror_size_px[1] // superpixel_dimension
    return np.full((sp_rows, sp_cols), np.exp(1j * phase), dtype=np.complex128)


def generate_band_edge_grating(
    mirror_size_px,
    superpixel_dimension,
    orientation="horizontal",
    bandwidth_fraction=0.75,
    cycles_per_superpixel=None,
):
    """Cosine grating at the chosen bandwidth-limit spatial frequency.

    Produces a +/-1 diffraction order pair sitting on opposite rims of the aperture:
    - max_aperture_radius = L / (4 * n) corresponds to 1/4 cycle per superpixel (bandwidth_fraction=1.0).
    - real_aperture_radius = bandwidth_fraction * max_aperture_radius corresponds to
      cycles_per_superpixel = 0.25 * bandwidth_fraction.

    Args:
        mirror_size_px (tuple): (ny, nx) DMD mirror dimensions, e.g. (1080, 1920).
        superpixel_dimension (int): Superpixel size n (e.g. 3).
        orientation (str): 'horizontal' ('x'), 'vertical' ('y'), or 'both'.
        bandwidth_fraction (float): Fraction of max_aperture_radius (default 0.75).
        cycles_per_superpixel (float, optional): Explicit spatial frequency in cycles/superpixel.
            If None, computed as 0.25 * bandwidth_fraction.
    """
    sp_rows = mirror_size_px[0] // superpixel_dimension
    sp_cols = mirror_size_px[1] // superpixel_dimension
    yy_sp, xx_sp = np.mgrid[0:sp_rows, 0:sp_cols].astype(float)
    yy_sp -= (sp_rows - 1) / 2
    xx_sp -= (sp_cols - 1) / 2

    if cycles_per_superpixel is None:
        cycles_per_superpixel = 0.25 * bandwidth_fraction

    ori = orientation.lower()
    if ori in ("horizontal", "x", "col", "cols"):
        target = np.cos(2 * np.pi * cycles_per_superpixel * xx_sp)
    elif ori in ("vertical", "y", "row", "rows"):
        target = np.cos(2 * np.pi * cycles_per_superpixel * yy_sp)
    elif ori == "both":
        target = 0.5 * (
            np.cos(2 * np.pi * cycles_per_superpixel * xx_sp)
            + np.cos(2 * np.pi * cycles_per_superpixel * yy_sp)
        )
    else:
        raise ValueError("orientation must be 'horizontal' ('x'), 'vertical' ('y'), or 'both'")

    return target.astype(np.complex128)



def to_propagation_grid(mirror_array, samples_per_mirror, grid_shape, device):
    """(1080, 1920) array with one value per mirror -> complex tensor on the simulation grid.

    Each mirror becomes samples_per_mirror x samples_per_mirror samples, and the chip is
    centred on the grid (same origin as apply_thin_lens).
    """
    sampled = np.kron(mirror_array, np.ones((samples_per_mirror, samples_per_mirror)))
    grid = np.zeros(grid_shape, dtype=np.complex64)
    center_hort = (grid_shape[0] - sampled.shape[0]) // 2
    center_vert = (grid_shape[1] - sampled.shape[1]) // 2
    grid[center_hort:center_hort + sampled.shape[0], center_vert:center_vert + sampled.shape[1]] = sampled
    return torch.from_numpy(grid).to(device)
