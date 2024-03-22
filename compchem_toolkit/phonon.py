"""
From calorine.tools.phonons
"""
from typing import Any, Dict

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

try:
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
except ModuleNotFoundError:  # pragma: no cover
    raise ModuleNotFoundError('phonopy (https://pypi.org/project/phonopy/) is '
                              'required in order to use the functionality '
                              'in the phonons module.'
                              )  # pragma: no cover


def get_force_constants(structure: Atoms,
                        calculator: SinglePointCalculator,
                        supercell_matrix: np.ndarray,
                        kwargs_phonopy: Dict[str, Any] = {},
                        kwargs_generate_displacements: Dict[str, Any] = {}) -> Phonopy:
    """
    Calculates the force constants for a given structure using
    `phonopy <https://phonopy.github.io/phonopy/>`_, which needs to be cited if this function
    is used for generating data for publication.
    The function returns a `Phonopy` object that can be used to calculate, e.g.,
    the phonon dispersion, the phonon density of states as well as related quantities such
    as the thermal displacements and the free energy.

    Parameters
    ----------
    structure
        structure for which to compute the phonon dispersion; usually this is a primitive cell
    calculator
        ASE calculator to use for the calculation of forces
    supercell_matrix
        specification of supercell size handed over to phonopy;
        should be a tuple of three values or a matrix
    kwargs_phonopy
        *Expert option*:
        keyword arguments used when initializing the `Phonopy` object;
        this includes, e.g., the tolerance used when determining the symmetry (`symprec`) and
        `parameters for the non-analytical corrections
        <https://phonopy.github.io/phonopy/phonopy-module.html#non-analytical-term-correction>`_
        (`nac_params`)
    kwargs_generate_displacements
        *Expert option*:
        keyword arguments to be handed over to the `generate_displacements` method;
        this includes in particular the `distance` keyword, which specifies the
        magnitude of the atomic displacement imposed when calculating the force constant matrix
    """

    # prepare primitive unit cell for phonopy
    structure_ph = PhonopyAtoms(symbols=structure.symbols,
                                cell=structure.cell,
                                scaled_positions=structure.get_scaled_positions(),
                                pbc=structure.pbc,
                                **kwargs_phonopy)

    # prepare supercells
    phonon = Phonopy(structure_ph, supercell_matrix)
    phonon.generate_displacements(**kwargs_generate_displacements)

    # compute force constant matrix
    forces = []
    for structure_ph in phonon.supercells_with_displacements:
        structure_ase = Atoms(symbols=structure_ph.symbols,
                              cell=structure_ph.cell,
                              scaled_positions=structure_ph.scaled_positions,
                              pbc=structure.pbc)
        structure_ase.calc = calculator
        forces.append(structure_ase.get_forces().copy())

    phonon.forces = forces
    phonon.produce_force_constants()

    return phonon