"""Collection of functions to work with pymatgen Structure object"""

import numpy as np
import pymatgen
from pymatgen.analysis.structure_matcher import StructureMatcher

def _calculate_atomic_disp(
    struct1: Structure,
    struct2: Structure,
    stol: float = 0.5,
) -> tuple:
    """
    Calculate root mean square displacement and atomic displacements,
    normalized by the free length per atom ((Vol/Nsites)^(1/3)) between
    two structures.

    Args:
        struct1 (:obj:`Structure`):
            Structure to compare to struct2.
        struct2 (:obj:`Structure`):
            Structure to compare to struct1.
        stol (:obj:`float`):
            Site tolerance used for structural comparison (via
            `pymatgen`'s `StructureMatcher`), as a fraction of the
            average free length per atom := ( V / Nsites ) ** (1/3). If
            output contains too many 'NaN' values, this likely needs to
            be increased.
            (Default: 0.5)

    Returns:
        :obj:`tuple`:
            Tuple of normalized root mean squared displacements and
            normalized displacements between the two structures.
    """
    sm = StructureMatcher(
        ltol=0.3, stol=stol, angle_tol=5, primitive_cell=False, scale=True
    )
    struct1, struct2 = sm._process_species([struct1, struct2])
    struct1, struct2, fu, s1_supercell = sm._preprocess(struct1, struct2)
    match = sm._match(
        struct1, struct2, fu, s1_supercell, use_rms=True, break_on_match=False
    )

    if match is None:
        return None
    return match[0], match[1]


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
    try:
        norm_rms_disp, norm_dist = _calculate_atomic_disp(
            struct1=ref_structure,
            struct2=struct2,
            stol=stol,
        )
        disp = (
            np.sum(norm_dist[norm_dist > min_dist * normalization]) / normalization
        )  # Only include displacements above min_dist threshold, and remove
        # normalization
    except TypeError:  # small tolerances - couldn't match lattices
        stol *= 2
        print("Initial attempt could not match lattices. Trying with " f"stol {stol}")
        try:
            norm_rms_disp, norm_dist = _calculate_atomic_disp(
                struct1=ref_structure,
                struct2=struct2,
                stol=stol,
            )
            disp = (
                np.sum(norm_dist[norm_dist > min_dist * normalization]) / normalization
            )  # Only include displacements above min_dist threshold, and remove
            # normalization
        except TypeError:
            pass

    return disp
