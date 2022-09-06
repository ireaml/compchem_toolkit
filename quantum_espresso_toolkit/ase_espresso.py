"""
Parse structure from Quantum Espresso output file
Much of this has been adapted from ase.io.espresso.
"""

import warnings
from copy import copy

# ase
from ase.cell import  Cell
from ase import Atoms
from ase.io.espresso import *

# pymatgen
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor


def read_espresso_structure(
    filename: str,
) -> Structure:
    """
    Reads a structure from Quantum Espresso output file and returns it
    as a pymatgen Structure. Units must be in Angstrom.
    Args:
        filename (str):
            Path to your file
    Returns:
        pymatgen.core.structure.Structure:
    """
    # ase.io.espresso functions seem a bit buggy, so we use the following implementation
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            file_content = f.read()

    if "Begin final coordinates" in file_content:
        file_content = file_content.split("Begin final coordinates")[-1] # last geometry
    if "End final coordinates" in file_content:
        file_content = file_content.split("End final coordinates")[0] # last geometry
    try:
        cell_lines = [
            line for line in
            file_content.split("CELL_PARAMETERS (angstrom)")[1].split(
                'ATOMIC_POSITIONS (angstrom)')[0].split("\n")
            if line != "" and line != " " and line != "   "
        ]
        atomic_positions = file_content.split("ATOMIC_POSITIONS (angstrom)")[1]
    except:
        print("Problem parsing 'ATOMIC_POSITIONS' and/or 'CELL_PARAMETERS' fields.")
        return "Not converged"

    # CELL parameters
    cell_lines_processed = [
        [
            float(number) for number in line.split()
        ] for line in cell_lines if len(line.split()) == 3
    ]
    # ATOMIC POSITIONS
    atomic_positions_processed = [
        [
            entry for entry in line.split()
        ] for line in atomic_positions.split("\n") if len(line.split()) >= 4
    ]
    coordinates = [
        [
            float(entry) for entry in line[1:4]
        ] for line in atomic_positions_processed
    ]
    symbols = [
        entry[0] for entry in atomic_positions_processed
        if entry != "" and entry != " " and entry != "  "
    ]
    # Check parsing is ok
    for entry in coordinates:
        if len(entry) != 3:
            # Ensure 3 numbers (xyz) are parsed from coordinates section
            print("Wrond coordinate", entry)
            warnings.warn("Problem parsing atomic coordinates.")
            return "Not converged"
    if not len(symbols) == len(coordinates):
        # Same number of atoms and coordinates
        warnings.warn(
                "Problem parsing atomic coordinates or symbols."
            )
        return "Not converged"
    try:
        atoms = Atoms(
            symbols=symbols,
            positions=coordinates,
            cell=cell_lines_processed,
            pbc=True,
        )
    except:
        warnings.warn(
            "Problem creating Atoms oject from parsed parameters, symbols and coordinates."
            " Check file & relaxation"
        )
        return "Not converged"
    try:
        aaa = AseAtomsAdaptor()
        structure = aaa.get_structure(atoms)
        structure = structure.get_sorted_structure()
    except:
        warnings.warn(
            "Problem converting ase Atoms object to pymatgen Structure."
        )
        return "Not converged"
    return structure


