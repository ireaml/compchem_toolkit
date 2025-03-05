"""Useful functions to analyse Oh distortions in perovskites."""
from copy import deepcopy
from typing import Optional
import warnings

import numpy as np
from pymatgen.analysis.local_env import CrystalNN

# Pymatgen stuff
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

aaa = AseAtomsAdaptor()


def get_tilting_angles(
    struct: Structure,
    b_cation: str = "Pb",
    x_anion: str = "I",
    distance_between_b_cations: float = 6.6,
    distance_between_b_x: float = 3.8,
    algorithm: str = "neighbors",
    verbose: bool = False,
    average: bool = True,
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
       float: Average B-X-B angle (in degrees) if average=True or list of all calculated B-X-B angles if average=False.
    """
    # Get ase atoms object
    atoms = aaa.get_atoms(struct)
    # Initialize CrystalNN
    cn = CrystalNN(search_cutoff=distance_between_b_x)

    # Get all B cation sites in structure
    b_sites = [
        index for index, site in enumerate(struct) if site.species_string == b_cation
    ]
    b_sites_copy = deepcopy(b_sites)  # just for sanity
    angles_b_x_b = []
    # Loop for each B cation site
    for b_site_1 in b_sites_copy:
        b_sites_copy.remove(b_site_1)  # Remove site from list to avoid double counting
        # Get neighbouring B cation sites (distance between them < 6.4 A)
        for b_site_2 in b_sites_copy:
            distance = struct.get_distance(b_site_1, b_site_2)
            if distance < distance_between_b_cations:
                # Get the anions surrounding each B cation
                if algorithm == "neighbors":
                    # Initially, was using the get_neighbors method, but think CrystalNN is more reliable, yet slower.
                    x_neighbors_of_b_1 = [
                        site.index
                        for site in struct.get_neighbors(
                            struct[b_site_1], r=distance_between_b_x
                        )
                        if site.species_string == x_anion
                    ]
                    x_neighbors_of_b_2 = [
                        site.index
                        for site in struct.get_neighbors(
                            struct[b_site_2], r=distance_between_b_x
                        )
                        if site.species_string == x_anion
                    ]
                elif algorithm == "crystal_nn":
                    x_neighbors_of_b_1 = [
                        site_info["site_index"]
                        for site_info in cn.get_nn_data(struct, b_site_1).all_nninfo
                        if site_info["site"].species_string == x_anion
                    ]
                    x_neighbors_of_b_2 = [
                        site_info["site_index"]
                        for site_info in cn.get_nn_data(struct, b_site_2).all_nninfo
                        if site_info["site"].species_string == x_anion
                    ]

                # Make sure we find 6 X anions neighbouring the B cation
                if not len(x_neighbors_of_b_1) == 6:
                 warnings.warn(
                     f"I find {len(x_neighbors_of_b_1)} {x_anion} surrounding the {b_cation}. This number should be 6!"
                 )
                if not len(x_neighbors_of_b_2) == 6:
                    warnings.warn(
                        f"I find {len(x_neighbors_of_b_2)} {x_anion} surrounding the {b_cation}. This number should be 6!"
                        )

                # Get the X anion connecting the two octahedra (the edge that the octahedra share)
                common_x_anions = list(
                    set(x_neighbors_of_b_1).intersection(x_neighbors_of_b_2)
                )
                if not common_x_anions:  # no common X anion
                    print(f"No common {x_anion} between the {b_cation}")
                    break
                for x_site in common_x_anions:
                    # pymatgen_angle = round(struct.get_angle(b_site_1, x_site, b_site_2), 3) # There is a bug in this pymatgen fucntion and think it struggles with pbc?
                    # Use ASE instead
                    ase_angle = atoms.get_angle(b_site_1, x_site, b_site_2, mic=True)
                    # print("Sites ", b_site_1, x_anion, b_site_2, angle, ase_angle)
                    angles_b_x_b.append(round(ase_angle, 3))
    if verbose:
        print("Tilting angles: ", angles_b_x_b)  # in degrees
    if average:
        return round(np.mean(angles_b_x_b), 3)  # in degrees
    return angles_b_x_b


def get_tilting_angles_xbx(
    struct: Structure,
    b_cation: str = "Pb",
    x_anion: str = "I",
    distance_between_b_cations: float = 6.6,
    distance_between_b_x: float = 3.8,
    algorithm: str = "neighbors",
    verbose: bool = False,
    average: bool = True,
):
    """
    Calculates tilting angles (between X-B-X) for a given structure.
    It returns the average X-B-X angle (in degrees) and can print all calculated X-B-X angles

    Args:
        struct (Structure): pymatgen structure of your material.
        b_cation (str, optional): symbol of the B cation (in a perovskite with general formula ABX3). \
            Defaults to 'Pb'.
        x_anion (str, optional): symbol of the X anion in the perovskite. Defaults to 'I'.
        distance_between_b_cations (float): distance between 2 neighbouring B cations (the centre of the octahedra), in A.
        distance_between_b_x (float): distance between bonded B cation and X anion, in A.
        algorithm (str, optional): Algorithm used to find the x anions bonded to the B cations. \
            This can be 'crystal_nn' (more reliable but slower) or 'neighbors' (faster).
            Defaults to 'neighbors'.
        verbose (bool, optional): Print all calculated X-B-X angles. \
            Defaults to False.

    Returns:
       float: Average X-B-X angle (in degrees) if average=True or list of all calculated X-B-X angles if average=False.
    """
    # Get ase atoms object
    atoms = aaa.get_atoms(struct)
    # Initialize CrystalNN
    cn = CrystalNN(search_cutoff=distance_between_b_x)

    # Get all X anion sites in structure
    x_sites = [
        index for index, site in enumerate(struct) if site.species_string == x_anion
    ]
    angles_x_b_x = []
    # Loop for each X anion site
    for x_site in x_sites:
        # Get the neighboring B cations
        if algorithm == "neighbors":
            b_neighbors = [
                site.index
                for site in struct.get_neighbors(struct[x_site], r=distance_between_b_x)
                if site.species_string == b_cation
            ]
        elif algorithm == "crystal_nn":
            b_neighbors = [
                site_info["site_index"]
                for site_info in cn.get_nn_data(struct, x_site).all_nninfo
                if site_info["site"].species_string == b_cation
            ]

        # Ensure we have exactly 2 B cations to form an X-B-X angle
        if len(b_neighbors) != 2:
            continue

        b_site_1, b_site_2 = b_neighbors

        # Compute X-B-X angle
        ase_angle = atoms.get_angle(b_site_1, x_site, b_site_2, mic=True)
        angles_x_b_x.append(round(ase_angle, 3))

    if verbose:
        print("Tilting angles: ", angles_x_b_x)  # in degrees
    if average:
        return round(np.mean(angles_x_b_x), 3)  # in degrees
    return angles_x_b_x