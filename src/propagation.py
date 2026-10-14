"""
propagation.py
Written by Miles Houston

Pytorch functions for optical propagation used in imaging simulations.
"""
import torch.nn.functional as functional
import torch
import numpy as np

import torch


# this function is responsible for the fraunhofer propagation of the wavefront from the sample plane to the detector plane.
def fraunhofer_propagate(
    wavefront,
    z_dist,
    grid_spacing_object,
    wavelength,
):
    """Propagate a wavefront using the Fraunhofer (far-field) diffraction. 

    Args: 
        wavefront (torch.Tensor): Complex-valued wavefront to be propagated.
        z_dist (float): propagation distance from object to sample. 
        grid_spacing_object (float or tuple): Spacing (in meters) between points in the wavefront tensor. 

    Returns: 
        torch.Tensor: Complex valued propagated wavefront. 
    """
    
    ny, nx = wavefront.shape
    device = wavefront.device
    wavelength = torch.tensor(wavelength, dtype=torch.float32, device=device)  # or 'cpu'
    k = 2 * torch.pi / wavelength



    
    if isinstance(grid_spacing_object, (int, float)):
        dy = dx = grid_spacing_object
    else:
        dy, dx = grid_spacing_object

    # Get the spatial frequencies mapped by the FFT
    fx = torch.fft.fftshift(torch.fft.fftfreq(nx, d=dx, device=device))
    fy = torch.fft.fftshift(torch.fft.fftfreq(ny, d=dy, device=device))

    # Map spatial frequencies to physical coordinates at the detector plane
    x_detector = fx * wavelength * z_dist
    y_detector = fy * wavelength * z_dist
    X_det, Y_det = torch.meshgrid(x_detector, y_detector, indexing="xy")

    # FFT
    wavefront_ft = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(wavefront))
    ) * (dx * dy)

    # The quadratic phase term must use the detector coordinates
    quadratic_phase = torch.exp(1j * k * (X_det**2 + Y_det**2) / (2 * z_dist)) / (
        1j * wavelength * z_dist
    )
    phase_factor = torch.exp(torch.tensor(1j * k * z_dist, device=device))

    detector_wave = wavefront_ft * quadratic_phase * phase_factor

    # Note that the detector wave now has a different grid spacing / size than the input wave front.
    return detector_wave


def crop_wavefront(wavefront, grid_spacing_detector, detector_size):

    n_h = int(detector_size[0] / grid_spacing_detector[0])
    n_w = int(detector_size[1] / grid_spacing_detector[1])
    
    sh = (wavefront.shape[-2] - n_h) // 2
    sw = (wavefront.shape[-1] - n_w) // 2
    return wavefront[..., sh : sh + n_h, sw : sw + n_w]


def capture_diffraction(wavefront, grid_spacing_detector, detector_size):
    cropped_wavefront = crop_wavefront(wavefront, grid_spacing_detector, detector_size)
    return torch.abs(cropped_wavefront) ** 2


def capture_intensity(wavefront, grid_spacing_detector, detector_size):
    return capture_diffraction(wavefront, grid_spacing_detector, detector_size).sum()

def downsample_area_exact(intensity, detector_pixels):
    # intensity: [H, W] real tensor
    H, W = intensity.shape[-2], intensity.shape[-1]
    assert H % detector_pixels == 0 and W % detector_pixels == 0, \
        f"{H}x{W} not divisible by {detector_pixels}"

    sh = H // detector_pixels
    sw = W // detector_pixels

    # reshape into bins and sum
    return intensity.reshape(detector_pixels, sh, detector_pixels, sw).sum(dim=(1, 3))


def capture_diffraction_downsample(
    wavefront,
    grid_spacing_detector,
    detector_size,
    detector_pixels=None,
    reduction="sum",
):
    """
    Returns |cropped_wavefront|^2 downsampled to (detector_pixels, detector_pixels).

    ``reduction="sum"`` preserves the historical detector-binning behavior.
    Use ``reduction="mean"`` when the measured detector data were downsampled
    by averaging pixels.
    """
    cropped = crop_wavefront(wavefront, grid_spacing_detector, detector_size)
    intensity = torch.abs(cropped) ** 2

    intensity_downsampled = functional.interpolate(intensity.unsqueeze(0).unsqueeze(1).cpu(), size=detector_pixels, mode="area")
    
    if reduction == "sum":
        # Area interpolation returns a mean. Scale it to integrated bin counts.
        bin_area = (
            (intensity.shape[-2] / detector_pixels[0])
            * (intensity.shape[-1] / detector_pixels[1])
        )
        intensity_downsampled = intensity_downsampled * bin_area
    elif reduction != "mean":
        raise ValueError("reduction must be 'sum' or 'mean'")

    return intensity_downsampled.squeeze().squeeze().to(intensity.device)


def apply_thin_lens(wavefront, grid_spacing, wavelength, focal_length, clear_radius=None):
    """
    Applies the phase transformation of a thin lens to a wavefront.

    Args:
        wavefront (torch.Tensor): Complex-valued wavefront tensor [H, W].
        grid_spacing (tuple): (dy, dx) grid spacing in meters.
        wavelength (float): Wavelength of the light in meters.
        focal_length (float): Focal length of the lens in meters.
        clear_radius (float, optional): Clear aperture radius of the lens in meters.
                                        If provided, limits transmission to within this radius.

    Returns:
        torch.Tensor: The wavefront after passing through the lens.
    """
    device = wavefront.device
    ny, nx = wavefront.shape
    dy, dx = grid_spacing

    y = (torch.arange(ny, device=device) - (ny - 1) / 2) * dy
    x = (torch.arange(nx, device=device) - (nx - 1) / 2) * dx
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Quadratic phase factor of the ideal thin lens
    ideal_phase = -torch.pi / (wavelength * focal_length) * (xx.square() + yy.square())
    lens_transmission = torch.exp(1j * ideal_phase.to(torch.complex64))

    # Apply clear aperture pupil mask if specified
    if clear_radius is not None:
        pupil = (xx.square() + yy.square()) <= clear_radius**2
        lens_transmission = lens_transmission * pupil

    return wavefront * lens_transmission


def apply_quadratic_curvature(wavefront, grid_spacing, wavelength, curvature):
    """Apply quadratic curvature at the current plane.

    Positive curvature is converging and has units of inverse meters.
    """
    device = wavefront.device
    ny, nx = wavefront.shape
    dy, dx = grid_spacing

    y = (torch.arange(ny, device=device) - (ny - 1) / 2) * dy
    x = (torch.arange(nx, device=device) - (nx - 1) / 2) * dx
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    phase = -torch.pi * curvature * (xx.square() + yy.square()) / wavelength
    return wavefront * torch.exp(1j * phase.to(torch.complex64))





def fresnel_propagate_transfer(wavefront, z_dist, grid_spacing, wavelength):
    """Fresnel transfer-function propagation. grid_spacing = (dy, dx) in metres."""
    device = wavefront.device
    dtype = torch.float32
    dy = torch.tensor(grid_spacing[0], dtype=dtype, device=device)
    dx = torch.tensor(grid_spacing[1], dtype=dtype, device=device)
    wl = torch.tensor(wavelength, dtype=dtype, device=device)
    ny, nx = wavefront.shape

    Fx = torch.fft.fftfreq(nx, d=dx, device=device)
    Fy = torch.fft.fftfreq(ny, d=dy, device=device)
    Vx = Fx.unsqueeze(0).expand(ny, nx)
    Vy = Fy.unsqueeze(1).expand(ny, nx)

    transfer_func = torch.exp((-1j * torch.pi * wl * z_dist * (Vx**2 + Vy**2)).to(torch.complex64))

    Wx, Wy = nx * dx, ny * dy                  # window size (m)
    z = abs(z_dist)                            # also valid for backward propagation
    fx_lim = 1 / (wl * torch.sqrt((2 * z / Wx)**2 + 1))
    fy_lim = 1 / (wl * torch.sqrt((2 * z / Wy)**2 + 1))
    keep = (Vx.abs() <= fx_lim) & (Vy.abs() <= fy_lim)
    transfer_func = transfer_func * keep.to(transfer_func.dtype)

    return torch.fft.ifft2(torch.fft.fft2(wavefront) * transfer_func)