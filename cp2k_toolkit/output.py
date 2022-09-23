import os
import warnings
from typing import Union

import ase
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def read_cp2k_structure(
    filename: str,
) -> Union[Structure, str]:
    """
    Reads a structure from CP2K restart file and returns it as a pymatgen
    Structure.
    Args:
        filename (:obj:`str`):
            Path to the cp2k restart file.
    Returns:
        :obj:`Structure`:
            `pymatgen` Structure object
    """
    if os.path.exists(filename):
        try:
            atoms = ase.io.read(
                filename=filename,
                format="cp2k-restart",
            )
        except Exception:
            warnings.warn(
                f"Problem parsing structure from CP2K restart file {filename}. Storing as 'Not "
                f"converged'. Check file & relaxation"
            )
            return "Not converged"
        try:
            aaa = AseAtomsAdaptor()
            structure = aaa.get_structure(atoms)
            structure = structure.get_sorted_structure()  # Sort sites by
            # electronegativity
        except Exception:
            warnings.warn(f"Problem converting ase Atoms object to pymatgen structure.")
            return "Not converged"

    else:
        raise FileNotFoundError(f"File {filename} does not exist!")
    return structure
