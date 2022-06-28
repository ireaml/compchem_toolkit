# Useful functions to analyse Oh distortions in perovskites
from typing import Optional
from copy import deepcopy
import numpy as np

# Pymatgen stuff
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor 
from pymatgen.analysis.local_env import CrystalNN

aaa = AseAtomsAdaptor()

def get_tilting_angles(
    struct: Structure,
    b_cation: str = 'Pb',
    x_anion: str = 'I',
    distance_between_b_cations: float = 6.6,
    distance_between_b_x: float = 3.8, 
    algorithm: str='neighbors',
    verbose: bool = False,
):
    """
    Calculates tilting angles (between B-X-B) for a given structure. 
    It returns the average B-X-B angle (in degrees) and can print all calculated B-X-B angles 

    Args:
        struct (Structure): pymatgen structure of your material.
        b_cation (str, optional): symbol of the B cation (in a perovskite with general formula ABX3). \
            Defaults to 'Pb'.
        x_anion (str, optional): symbol of the X anion in the perovskite. Defaults to 'I'.
        distance_between_b_cations: (float): distance between 2 neighbouring B cations (the centre of the octahedra), in A.
        distance_between_b_x (float): distance between bonded B cation and X anion, in A.
        algorithm (str, optional): Algorithm used to find the x anions bonded to the B cations. \
            This can be 'crystal_nn' (more reliable but slower) or 'neighbors' (faster).
            Defaults to 'neighbors'.
        verbose (bool, optional): Print all calculated B-X-B angles. \
            Defaults to False.

    Returns:
       float: Average B-X-B angle.
    """
    # Get ase atoms object
    atoms = aaa.get_atoms(struct)
    # Initialize CrystalNN
    cn = CrystalNN(search_cutoff = distance_between_b_x)

    # Get all B cation sites in structure
    b_sites = [index for index, site in enumerate(struct) if site.species_string == b_cation]
    b_sites_copy = deepcopy(b_sites) # just for sanity
    angles_b_x_b = []
    # Loop for each B cation site
    for b_site_1 in b_sites_copy:
        b_sites_copy.remove(b_site_1) # Remove site from list to avoid double counting
        # Get neighbouring B cation sites (distance between them < 6.4 A)
        for b_site_2 in b_sites_copy:
            distance = struct.get_distance(b_site_1, b_site_2)
            if distance < distance_between_b_cations: 
                # Get the anions surrounding each B cation
                if algorithm == 'neighbors':
                    # Initially, was using the get_neighbors method, but think CrystalNN is more reliable, yet slower.
                    x_neighbors_of_b_1 = [ site.index for site in struct.get_neighbors(struct[b_site_1], r = distance_between_b_x) if site.species_string == x_anion ]
                    x_neighbors_of_b_2 = [ site.index for site in struct.get_neighbors(struct[b_site_2], r = distance_between_b_x) if site.species_string == x_anion ]
                elif algorithm == 'crystal_nn':
                    x_neighbors_of_b_1 = [ site_info['site_index'] for site_info in cn.get_nn_data(struct, b_site_1).all_nninfo if site_info['site'].species_string == x_anion ] 
                    x_neighbors_of_b_2 = [ site_info['site_index'] for site_info in cn.get_nn_data(struct, b_site_2).all_nninfo if site_info['site'].species_string == x_anion ] 
    
                # Make sure we find 6 X anions neighbouring the B cation
                assert len(x_neighbors_of_b_1) == 6, f"I find {len(x_neighbors_of_b_1)} {x_anion} surrounding the {b_cation}. This number should be 6!"
                assert len(x_neighbors_of_b_2) == 6, f"I find {len(x_neighbors_of_b_2)} {x_anion} surrounding the {b_cation}. This number should be 6!"

                # Get the X anion connecting the two octahedra (the edge that the octahedra share)
                common_x_anions = list(set(x_neighbors_of_b_1).intersection(x_neighbors_of_b_2))
                if not common_x_anions : # no common X anion
                    print(f'No common {x_anion} between the {b_cation}')
                    break
                for x_site in common_x_anions:
                    # pymatgen_angle = round(struct.get_angle(b_site_1, x_site, b_site_2), 3) # There is a bug in this pymatgen fucntion and think it struggles with pbc?
                    # Use ASE instead
                    ase_angle = atoms.get_angle(b_site_1, x_site, b_site_2 , mic=True)
                    # print("Sites ", b_site_1, x_anion, b_site_2, angle, ase_angle)
                    angles_b_x_b.append(round(ase_angle, 3))
    if verbose:
        print("Tilting angles: ", angles_b_x_b) # in degrees
    return round(np.mean(angles_b_x_b), 3) # in degrees