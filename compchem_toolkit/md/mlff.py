import pymlff
from pymatgen.core.structure import Structure

def get_structure_from_config(config: pymlff.core.Configuration):
    # Get list of species in configuration
    species = []
    for element in config.atom_types_numbers:
        # Add the element as many times as its dict value
        species += [element] * config.atom_types_numbers[element]
    s = Structure(
        lattice=config.lattice,
        coords=config.coords,
        species=species,
        coords_are_cartesian=True, # Cartesian coords!
    )
    return s