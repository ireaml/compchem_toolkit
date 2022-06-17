# Collection of useful functions for vasp relaxations
from copy import deepcopy
import pymatgen
from pymatgen.core.structure import Structure
from aiida.orm.nodes.data.list import List


def get_selective_dynamics_from_layers(
    structure: Structure,
    number_of_layers_in_slab: int,
    epsilon: float = 0.01
    ):
    """
    Returns selective dynamics dictionary to parse to aiida vasp calcjob, 
    allowing to relax atoms in outer layers of slab

    Args:
        number_of_layers_in_slab (int): 
            number of layers in slab 
        structure (pymatgen.core.structure.Structure): 
            Structure
        epsilon (float, optional): 
            Tolerance to select sites to relax. The larger, the more sites will be 
            considered in the outer layers and hence relaxed. 
            Defaults to 0.01.

    Returns:
        dict: dictionary matching positions_dof to a list which indicates where site 
        in structure should be relaxed ([True,True,True]) or fixed ([False,False,False]).
        Follows same ordering as sites in the input structure.
    """
    structure_copy = deepcopy(structure) # sanity safety

    # Calculate max and min z of slab
    z_coords = [atom.coords[2] for atom in structure_copy]
    max_z = max(z_coords) ; print(f"Max z of site in slab: {max_z:.4f}")
    min_z = min(z_coords) ; print(f"Min z of site in slab: {min_z:.4f}")
    # Calculate average height of a layer
    height_per_layer = (max_z - min_z) / number_of_layers_in_slab
    print(f"Height of each layer: {height_per_layer:.4f} A")

    # Flags signaling whether the respective coordinate(s) of this atom will be allowed to change during the ionic relaxation
    # i.e. True enables relaxation of that coordinate
    selective_dynamics = {'positions_dof' : List(),} 
    atoms_in_outer_layers = []
    for atom in structure_copy:
        # select atoms in top or bottom layer
        if atom.coords[2] > (max_z - ( height_per_layer + epsilon) ) or atom.coords[2] < (min_z + (height_per_layer + epsilon) ):
            atoms_in_outer_layers.append(atom)
            selective_dynamics['positions_dof'].append([ True,True,True ] )# allow to relax
        else:
            selective_dynamics['positions_dof'].append([ False,False,False ] )  # dont relax. Fix their coordinates

    print(f"Total number of atoms in structure: {len(structure_copy)}")
    print(f"Number of atoms in outer layers (will be relaxed): {len(atoms_in_outer_layers)}")

    assert len(selective_dynamics['positions_dof']) == len(structure_copy)
    return selective_dynamics

def get_selective_dynamics_from_index(
    structure: Structure,
    indexes_sites_to_relax: list,
) -> dict:
    """
    Returns selective dynamics dictionary to parse to aiida vasp calcjob, \
    setting the atoms whose indexes are given in `indexes_sites_to_relax`
    to relax.
    """
    selective_dynamics = {'positions_dof' : List(),} # format required by aiida
    for index, site in enumerate(structure):
        if index in indexes_sites_to_relax:
            selective_dynamics['positions_dof'].append([True, True, True]) # sites to relax
        else:
            selective_dynamics['positions_dof'].append([False, False, False]) # fixed sites
    return selective_dynamics