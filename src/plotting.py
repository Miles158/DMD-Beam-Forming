"""
plotting.py

Visualization tools for DMD actuator patterns and optical planes.
"""

import numpy as np
import matplotlib.pyplot as plt


def unpack_superpixel_commands(commands, superpixel_dimension=None):
    """
    Unpacks a grid of superpixel commands into a 2D array of individual mirrors.

    Args:
        commands (np.ndarray): 
            Either shape (n_rows, n_cols, n**2) or (n_rows, n_cols, n, n).
        superpixel_dimension (int, optional): 
            Size n of the superpixel (n x n mirrors). If None, inferred from shape.

    Returns:
        np.ndarray: 2D array of shape (n_rows * n, n_cols * n).
    """
    commands = np.asarray(commands)
    if commands.ndim == 2:
        return commands

    if commands.ndim == 3:
        n_rows, n_cols, n_sq = commands.shape
        if superpixel_dimension is None:
            n = int(np.round(np.sqrt(n_sq)))
            if n * n != n_sq:
                raise ValueError(f"Last dimension {n_sq} is not a perfect square.")
        else:
            n = superpixel_dimension
        commands = commands.reshape(n_rows, n_cols, n, n)
    elif commands.ndim == 4:
        n_rows, n_cols, n_y, n_x = commands.shape
        n = n_y
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D array, got shape {commands.shape}")

    # (n_rows, n_cols, n, n) -> (n_rows, n, n_cols, n) -> (n_rows * n, n_cols * n)
    unpacked = commands.transpose(0, 2, 1, 3).reshape(n_rows * n, n_cols * n)
    return unpacked


def plot_dmd_actuators(
    actuators,
    mirror_pitch=7.56e-6,
    superpixel_dimension=None,
    unit="mm",
    centered=True,
    ax=None,
    cmap="gray",
    title="DMD Actuator Settings",
    origin="lower",
    show_colorbar=False,
    save_path=None,
    dpi=300,
):
    """
    Plots the DMD mirror actuator settings on the physical DMD plane with correct spatial scaling.

    Args:
        actuators (np.ndarray or torch.Tensor): 
            2D array of mirror states (ny, nx), or 3D/4D superpixel commands.
        mirror_pitch (float or tuple): 
            Physical mirror pitch in meters. Can be a single float (square mirrors)
            or (pitch_y, pitch_x). Default is 7.56e-6 m (7.56 µm).
        superpixel_dimension (int, optional): 
            Dimension n of superpixel if passing encoded superpixel commands.
        unit (str): 
            Units for axis coordinates: 'mm' (default), 'um' or 'µm', or 'm'.
        centered (bool): 
            If True (default), center the coordinate system at (0, 0) (optical axis).
            If False, place (0, 0) at the bottom-left corner of the mirror array.
        ax (matplotlib.axes.Axes, optional): 
            Existing Matplotlib axis to draw on. If None, a new figure is created.
        cmap (str): 
            Colormap to use (default: 'gray', where 0=OFF=black, 1=ON=white).
        title (str): 
            Plot title.
        origin (str): 
            'lower' (default) or 'upper' for imshow.
        show_colorbar (bool): 
            Whether to display a colorbar.
        save_path (str, optional): 
            File path to save the figure (e.g., 'dmd_pattern.png'). If provided,
            fig.savefig(...) is executed automatically.
        dpi (int): 
            Resolution in dots-per-inch when saving the figure (default: 300).

    Returns:
        tuple: (fig, ax)
    """
    # Convert PyTorch tensors to NumPy if necessary
    if hasattr(actuators, "detach"):
        actuators = actuators.detach().cpu().numpy()
    actuators = np.asarray(actuators)

    # Unpack superpixel commands if 3D or 4D
    if actuators.ndim in (3, 4):
        mirror_array = unpack_superpixel_commands(actuators, superpixel_dimension=superpixel_dimension)
    elif actuators.ndim == 2:
        mirror_array = actuators
    else:
        raise ValueError(f"Actuator array must have 2, 3, or 4 dimensions, got shape {actuators.shape}")

    ny, nx = mirror_array.shape

    # Handle mirror pitch
    if isinstance(mirror_pitch, (tuple, list)):
        pitch_y, pitch_x = mirror_pitch
    else:
        pitch_y = pitch_x = float(mirror_pitch)

    # Unit scaling
    unit_lower = unit.lower()
    if unit_lower in ("mm", "millimeter", "millimeters"):
        scale = 1e3
        unit_label = "mm"
    elif unit_lower in ("um", "µm", "micrometer", "micrometers", "micron", "microns"):
        scale = 1e6
        unit_label = "µm"
    elif unit_lower in ("m", "meter", "meters"):
        scale = 1.0
        unit_label = "m"
    else:
        raise ValueError(f"Unknown unit: '{unit}'. Choose 'mm', 'um', or 'm'.")

    # Physical dimensions in selected units
    width = nx * pitch_x * scale
    height = ny * pitch_y * scale

    if centered:
        extent = [-width / 2.0, width / 2.0, -height / 2.0, height / 2.0]
    else:
        extent = [0.0, width, 0.0, height]

    if origin == "upper" and not centered:
        extent = [0.0, width, height, 0.0]
    elif origin == "upper" and centered:
        extent = [-width / 2.0, width / 2.0, height / 2.0, -height / 2.0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8 * (height / width) if width > 0 else 6))
    else:
        fig = ax.figure

    # Determine display range (vmin/vmax)
    is_binary = np.all(np.isin(mirror_array, [0, 1]))
    vmin, vmax = (0, 1) if is_binary else (None, None)

    im = ax.imshow(
        mirror_array,
        extent=extent,
        origin=origin,
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel(f"x ({unit_label})")
    ax.set_ylabel(f"y ({unit_label})")
    ax.set_title(f"{title}\n({ny}×{nx} mirrors, {width:.2f}×{height:.2f} {unit_label})")

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if is_binary:
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(["OFF (0)", "ON (1)"])

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")



def crop_image(
    field,
    aperture_position,
    grid_spacing,
    mirror_size_m,
    f1,
    f2,
    margin=0.55,
):
    """
    Window around the DMD image in the target plane: 
    cropped around lens 2's optical axis with the optical relay magnification (f2 / f1).

    Args:
        field (torch.Tensor or np.ndarray): 
            Complex field at the target plane (shape [..., ny, nx]).
        aperture_position (tuple): 
            (aperture_y, aperture_x) in meters.
        grid_spacing (tuple or float): 
            (dy, dx) grid spacing in meters.
        mirror_size_m (tuple): 
            (height_m, width_m) physical size of DMD mirror active area in meters.
        f1 (float): 
            Focal length of lens 1 in meters.
        f2 (float): 
            Focal length of lens 2 in meters.
        margin (float): 
            Fractional expansion factor around the geometrically demagnified image (default: 0.55).

    Returns:
        np.ndarray: Cropped complex field.
    """
    if hasattr(field, "detach"):
        field = field.detach()

    aperture_y, aperture_x = aperture_position
    if isinstance(grid_spacing, (tuple, list)):
        dy, dx = grid_spacing
    else:
        dy = dx = float(grid_spacing)

    ny, nx = field.shape[-2], field.shape[-1]
    cy = int(round(aperture_y / dy + (ny - 1) / 2))
    cx = int(round(aperture_x / dx + (nx - 1) / 2))

    hy = int(margin * mirror_size_m[0] * f2 / f1 / dy)
    hx = int(margin * mirror_size_m[1] * f2 / f1 / dx)

    cropped = field[..., cy - hy : cy + hy, cx - hx : cx + hx]
    if hasattr(cropped, "cpu"):
        cropped = cropped.cpu().numpy()
    return np.asarray(cropped)


def plot_ideal_vs_dmd(
    out_ref,
    out_dmd,
    aperture_position=None,
    grid_spacing=None,
    mirror_size_m=None,
    f1=None,
    f2=None,
    save_path="ideal_vs_dmd.png",
    dpi=300,
    figsize=(10, 6),
    titles=("reference", "DMD"),
    intensity_cmap="gray",
    phase_cmap="twilight",
):
    """
    Plots a 2x2 comparison (intensity and phase) between reference/ideal field and DMD field.
    If optical parameters (aperture_position, grid_spacing, mirror_size_m, f1, f2) are provided,
    it automatically crops both fields using crop_image().

    Args:
        out_ref (torch.Tensor or np.ndarray): Reference output field (full grid or pre-cropped).
        out_dmd (torch.Tensor or np.ndarray): DMD output field (full grid or pre-cropped).
        aperture_position (tuple, optional): (aperture_y, aperture_x) in meters.
        grid_spacing (tuple, optional): (dy, dx) in meters.
        mirror_size_m (tuple, optional): (height_m, width_m) in meters.
        f1 (float, optional): Focal length of lens 1 in meters.
        f2 (float, optional): Focal length of lens 2 in meters.
        save_path (str, optional): File path to save figure (default: 'ideal_vs_dmd.png').
        dpi (int): Resolution for saved figure.
        figsize (tuple): Figure size (default: (10, 6)).
        titles (tuple): Titles for row 0 and row 1 (default: ('reference', 'DMD')).
        intensity_cmap (str): Colormap for intensity (default: 'gray').
        phase_cmap (str): Colormap for phase (default: 'twilight').

    Returns:
        tuple: (fig, ax)
    """
    # If optical parameters are given, crop both fields
    if all(p is not None for p in (aperture_position, grid_spacing, mirror_size_m, f1, f2)):
        ref_img = crop_image(out_ref, aperture_position, grid_spacing, mirror_size_m, f1, f2)
        dmd_img = crop_image(out_dmd, aperture_position, grid_spacing, mirror_size_m, f1, f2)
    else:
        ref_img = out_ref.detach().cpu().numpy() if hasattr(out_ref, "detach") else np.asarray(out_ref)
        dmd_img = out_dmd.detach().cpu().numpy() if hasattr(out_dmd, "detach") else np.asarray(out_dmd)

    fig, ax = plt.subplots(2, 2, figsize=figsize)
    for i, (img, name) in enumerate([(ref_img, titles[0]), (dmd_img, titles[1])]):
        ax[i, 0].imshow(np.abs(img)**2, cmap=intensity_cmap)
        ax[i, 0].set_title(f"{name}: intensity")
        ax[i, 1].imshow(np.angle(img), cmap=phase_cmap)
        ax[i, 1].set_title(f"{name}: phase")

    for a in ax.ravel():
        a.axis("off")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, ax


# Convenient alias
plot_reference_vs_dmd = plot_ideal_vs_dmd


