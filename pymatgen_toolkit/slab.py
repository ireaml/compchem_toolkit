"""Collection of functions to work with pymatgen Slab object"""
import pymatgen
from pymatgen.core.structure import Structure
from pymatgen.core.surface import Slab

def get_layer_sites(
    structure: Structure,
    epsilon: float,
):
    """
    Returns sites in top and bottom layer of slab. These are selected as those whose
    z coordinate lies between the top/bottom of slab and top-epsilon/bottom+epsilon.

    Args:
        structure (pymatgen.core.structure.Structure):
            pymatgen Structure object of slab
        epsilon (float):
            Length from top/bottom of slab to select the sites that lie within
            (i.e. sites within top/bottom surface of slab +- epsilon)

    Returns:
        [dict]: dict with top_layer and bottom_layer sites
    """
    max_z = max(site.coords[2] for site in structure)
    min_z = min(site.coords[2] for site in structure)
    top_layer = [ site for site in structure if site.coords[2] >= max_z - epsilon]
    bottom_layer = [ site for site in structure if site.coords[2] <= min_z + epsilon]
    # print(f"Number of sites in top layer: {len(top_layer)}")
    return {'top_layer': top_layer, 'bottom_layer': bottom_layer}


def get_structure_from_slab(
    slab: Slab,
) -> Structure:
    """
    Transform pymatgen slab to pymatgen structure

    Args:
        slab (Slab):
            pymatgen.core.surface.Slab object

    Returns:
        Structure: pymatgen.core.structures.Structure object
    """
    struct = Structure(
        lattice = slab.lattice,
        species = slab.species,
        coords = [site.frac_coords for site in slab],
        coords_are_cartesian = False,
    )
    return struct