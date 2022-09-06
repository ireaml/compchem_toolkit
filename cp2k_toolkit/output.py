import os
import warnings
from typing import Union
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
import ase

aaa = AseAtomsAdaptor()

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
            aaa = AseAtomsAdaptor()
            atoms = ase.io.read(
                filename=filename,
                format="cp2k-restart",
            )
            structure = aaa.get_structure(atoms)
            structure = structure.get_sorted_structure()  # Sort sites by
            # electronegativity
        except Exception:
            warnings.warn(
                f"Problem parsing structure from: {filename}, storing as 'Not "
                f"converged'. Check file & relaxation"
            )
            structure = "Not converged"
    else:
        raise FileNotFoundError(f"File {filename} does not exist!")
    return structure