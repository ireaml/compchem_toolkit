"""Collection of functions to work with pymatgen Structure object"""

import pymatgen
from shakenbreak import analysis 
import numpy as np


def get_atomic_disp(
    ref_structure: pymatgen.core.structure.Structure,
    struct2: pymatgen.core.structure.Structure,
    stol: float = 0.4,
    min_dist: float = 0.1,
):
    """
    Calculates sum of atomic displacements between paired sites of 2 structures.
    Only displacements above a threshold (`min_dist`) are considered.
    """
    normalization = (len(ref_structure) / ref_structure.volume) ** (1 / 3)
    norm_rms_disp, norm_dist = analysis._calculate_atomic_disp(
        struct1=ref_structure,
        struct2=struct2,
        stol=stol,
    )
    disp =  (
        np.sum(norm_dist[norm_dist > min_dist * normalization])
        / normalization
    )  # Only include displacements above min_dist threshold, and remove
    # normalization
    return disp