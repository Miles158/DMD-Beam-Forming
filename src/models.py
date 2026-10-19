import torch
import torch.nn as nn

from src import propagation as prop


class Relay4f(nn.Module):
    """DMD -> f1 -> lens 1 -> f1 -> aperture -> f2 -> lens 2 -> f2 -> target plane.

    Lens 2 is centred on the aperture (the paper's off-axis lens placement).
    All positions are (y, x) in metres, measured from the grid centre.
    """

    def __init__(self, grid_spacing, wavelength, f1, f2, aperture_center, aperture_radius,
                 lens1_clear_radius=None, lens2_clear_radius=None):
        super().__init__()
        self.grid_spacing = grid_spacing
        self.wavelength = wavelength
        self.f1 = f1
        self.f2 = f2
        self.aperture_center = aperture_center
        self.aperture_radius = aperture_radius
        self.lens1_clear_radius = lens1_clear_radius
        self.lens2_clear_radius = lens2_clear_radius


    def forward(self, field):
        """Complex field at the DMD (grid_shape) -> complex field at the target plane (grid_shape)."""
        wavelength = self.grid_spacing, self.wavelength

        field = prop.fresnel_propagate_transfer(field, self.f1, self.grid_spacing, self.wavelength)                    # DMD -> lens 1
        field = prop.apply_thin_lens(field, self.grid_spacing, self.wavelength, self.f1, clear_radius=self.lens1_clear_radius)
        fourier_plane = prop.fresnel_propagate_transfer(field, self.f1, self.grid_spacing, self.wavelength)            # lens 1 -> Fourier plane

        field = prop.apply_circular_aperture(fourier_plane, self.grid_spacing, self.aperture_center, self.aperture_radius)
        field = prop.fresnel_propagate_transfer(field, self.f2, self.grid_spacing, self.wavelength)                    # aperture -> lens 2
        field = prop.apply_thin_lens(field, self.grid_spacing, self.wavelength, self.f2, clear_radius=self.lens2_clear_radius,
                                     center=self.aperture_center)
        target_plane = prop.fresnel_propagate_transfer(field, self.f2, self.grid_spacing, self.wavelength)             # lens 2 -> target plane

        return target_plane


def fidelity(reference, obtained):

    overlap = torch.sum(reference.conj() * obtained)
    norm = torch.sum(reference.abs().square()) * torch.sum(obtained.abs().square())
    
    return (overlap.abs().square() / norm).item()