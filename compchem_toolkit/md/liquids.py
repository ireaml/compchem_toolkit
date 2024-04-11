import numpy as np
from pymatgen.core.structure import Structure, Molecule
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element

def get_min_bond_distance(s: Structure, verbose=True):
    """Calculate minimum and average distance in structure

    Args:
        s (Structure): pymatgen Structure object
        verbose (bool, optional): Print min and mean distances. Defaults to True.

    Returns: min distance, mean distance
    """
    nn = CrystalNN()
    distances = []
    pairs = [] # to avoid double counting
    for i in range(len(s)):
        nns = nn.get_nn_info(s, i)
        for n in nns:
            if (i, n) not in pairs and (n, i) not in pairs:
                distances.append(s.get_distance(i, n["site_index"]))
                pairs.append((i, n["site_index"]))
    if verbose:
        print(f"Min distance: {min(distances):.2f} A, Mean distance: {np.mean(distances):.2f} A")
    return min(distances), np.mean(distances)


def get_density_atoms_per_A3(structure):
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
    return density  * (Na / molecular_mass) * (1 / 10**8)**3