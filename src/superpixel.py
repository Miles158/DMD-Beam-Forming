
import numpy as np
import itertools

def pixel_phases(pixel_dimension):
    """Using the ideal aperture position, calculate the phase of each mirror in a superpixel."""

    row, col = np.mgrid[0:pixel_dimension, 0: pixel_dimension]
    return 2 * np.pi * (col / pixel_dimension**2 + row / pixel_dimension)
 


 
def build_superpixel_fields(superpixel_dimension, phase_array):
    """Generate the look-up table for output fields corresponding to all the possible mirror configurations."""
    pixels_per_superpixel = superpixel_dimension**2
    dmd_commands = []
    fields = []
    for command_vec in itertools.product([0, 1], repeat=pixels_per_superpixel):      # every 0/1 vector of length m
        dmd_commands.append(command_vec)

        field = np.sum((np.array(command_vec) * np.exp(1j * phase_array.ravel()) / pixels_per_superpixel))

        fields.append(field)

    return np.array(dmd_commands), np.array(fields)



def find_best_command(target, commands, fields):
    """Finds the best output field with the lookup table using L2 norm."""
    index = np.argmin(np.abs(fields - target))
    return commands[index]



def encode(target_complex_field, superpixel_dimension, commands, fields):
    n = superpixel_dimension
    n_rows, n_cols = target_complex_field.shape
    normalized_target = target_complex_field / np.abs(target_complex_field).max()
    scaled_target = normalized_target * np.abs(fields).max()

    dmd_commands = np.zeros((n_rows, n_cols, n**2), dtype=np.uint16)
    for row in range(n_rows):
        for col in range(n_cols):
            target = scaled_target[row, col] * np.exp(-2j * np.pi * col / n)   
            dmd_commands[row, col] = find_best_command(target, commands, fields)

    return dmd_commands

