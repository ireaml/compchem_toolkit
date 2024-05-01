import numpy as np
from pymatgen.core.structure import Structure, Molecule
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element


def get_min_bond_distance(s: Structure, mode="fast", cutoff=4, verbose=True):
    """Calculate minimum and average distance in structure

    Args:
        s (Structure): pymatgen Structure object
        verbose (bool, optional): Print min and mean distances. Defaults to True.

    Returns: min distance, mean distance
    """
    distances = []
    pairs = [] # to avoid double counting
    if mode == "accurate":
        nn = CrystalNN()
        for i in range(len(s)):
            nns = nn.get_nn_info(s, i)
            for n in nns:
                if (i, n) not in pairs and (n, i) not in pairs:
                    distances.append(s.get_distance(i, n["site_index"]))
                    pairs.append((i, n["site_index"]))
    elif mode == "fast":
        for i, s1 in enumerate(s):
            # Get sites within 4 A
            nn = s.get_neighbors(site=s1, r=cutoff)
            # Get distances
            for n in nn:
                if (n.index, i) not in pairs: # avoid duplicates
                    distances.append(s.get_distance(i, n.index))
                    pairs.append((i, n.index))
    if verbose:
        print(f"Min distance: {min(distances):.2f} A, Mean distance: {np.mean(distances):.2f} A")
    return min(distances), np.mean(distances)


def get_density_atoms_per_A3(structure, verbose=True):
    """Calculate density in atoms/A^3"""
    Na = 6.022 * 10**23
    if isinstance(structure, Molecule):
        molecular_mass = structure.composition.weight
    else: # mass of atom
        molecular_mass = 0
        for el, n_el in structure.composition.reduced_composition.as_dict().items():
            molecular_mass += Element(el).atomic_mass * n_el
    print(f"Molecular mass: {molecular_mass}")
    density = structure.density # g/cm^3
    # Tranform from g/cm^3 to atoms/A^3
    density_atoms_A3 = density  * (Na / molecular_mass) * (1 / 10**8)**3
    if verbose:
        print(f"Density in atoms/A^3: {density_atoms_A3:.2e}")
    return density_atoms_A3

def get_coordination_number(s, verbose=True):
    nn = CrystalNN()
    coordination_numbers = []
    for i in range(len(s)):
        nns = nn.get_nn_info(s, i)
        coordination_numbers.append(len(nns))
    mean_coord_num = np.mean(coordination_numbers)
    if verbose:
        print(f"Mean coordination number: {mean_coord_num:.2f}. Max: {max(coordination_numbers)}, Min: {min(coordination_numbers)}")
    return coordination_numbers