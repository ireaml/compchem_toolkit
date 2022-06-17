""" Collection of functions to work with pymatgen Molecule object"""

import pymatgen

def parse_molecule_from_sdf(
    path: str
) -> pymatgen.core.structure.Molecule:
    """
    Parse a sdf file into a pymatgen Molecule object.

    Args:
        path (str): path to the `sdf` file containing your molecule.

    Returns:
        pymatgen.core.structure.Molecule: a pymatgen Molecule object.
    """
    with open(path) as ff:
        file = ff.read()
    atoms = file.splitlines()[0:] # read all lines
    species = [] ; coords = []
    for line in atoms:
        splitted = line.split()[0:4]
        coords.append([float(i) for i in splitted[0:3]])
        species.append(splitted[3])

    molecule = pymatgen.core.structure.Molecule(
        species = species,
        coords = coords,
    )
    return molecule