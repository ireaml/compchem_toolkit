"""
Parse structure/output information from Quantum Espresso output file
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

# Section identifiers
_PW_START = 'Program PWSCF'
_PW_END = 'End of self-consistent calculation'
_PW_CELL = 'CELL_PARAMETERS'
_PW_POS = 'ATOMIC_POSITIONS'
_PW_MAGMOM = 'Magnetic moment per site'
_PW_FORCE = 'Forces acting on atoms'
_PW_TOTEN = '!    total energy'
_PW_STRESS = 'total   stress'
_PW_FERMI = 'the Fermi energy is'
_PW_HIGHEST_OCCUPIED = 'highest occupied level'
_PW_HIGHEST_OCCUPIED_LOWEST_FREE = 'highest occupied, lowest unoccupied level'
_PW_KPTS = 'number of k points='
_PW_BANDS = _PW_END
_PW_BANDSTRUCTURE = 'End of band structure calculation'


def read_espresso_structure(
    filename: str,
) -> Structure:
    """
    Reads a structure from Quantum Espresso output and returns it as a pymatgen Structure.
    Args:
        filename (str): 
            Path to your file
    Returns:
        pymatgen.core.structure.Structure: 
    """
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            file_content = f.read()
    try:
        if "Begin final coordinates" in file_content:
            file_content = file_content.split("Begin final coordinates")[-1] # last geometry
        cell_lines = [line for line in file_content.split("CELL_PARAMETERS")[1].split('ATOMIC_POSITIONS')[0].split("\n") if line != "" and line != " "]
        atomic_positions = file_content.split("ATOMIC_POSITIONS")[1]
        # CELL parameters
        cell_lines_processed = [
            [float(number) for number in line.split()] for line in cell_lines if len(line.split()) == 3
        ]
        # ATOMIC POSITIONS
        atomic_positions_processed = [
            [entry for entry in line.split()] for line in atomic_positions.split("\n") if len(line.split()) >= 4
        ]
        coordinates = [
            [float(entry) for entry in line[1:4]] for line in atomic_positions_processed
        ]
        symbols = [entry[0] for entry in atomic_positions_processed]
        # Check parsing is ok
        for entry in coordinates:
            assert len(entry) == 3 # Encure 3 numbers (xyz) are parsed from coordinates section
        assert not len(symbols) == len(coordinates) # Same number of atoms and coordinates
        atoms = Atoms(
            symbols=symbols,
            positions=coordinates,
            cell=cell_lines_processed,
            pbc=True,
            
        )
        aaa = AseAtomsAdaptor()
        structure = aaa.get_structure(atoms)
    except:
        warnings.warn(
                f"Problem parsing structure from: {filename}, storing as 'Not "
                f"converged'. Check file & relaxation"
        )
        structure = "Not converged"
    return structure
    
def get_indexes(
    pwo_lines,
) -> dict:
    """
    Parse Quantum Espresso output file and return a dictionary with the indexes
    of the lines where the different information is printed.
    Adapted from ase.io.espresso.
    
    Args:
        pwo_lines (list): 
            list of lines from Quantum Espresso output file

    Returns:
        dict: dict mapping section identifier to index of line
    """
    indexes = {
        _PW_START: [],
        _PW_END: [],
        _PW_CELL: [],
        _PW_POS: [],
        _PW_MAGMOM: [],
        _PW_FORCE: [],
        _PW_TOTEN: [],
        _PW_STRESS: [],
        _PW_FERMI: [],
        _PW_HIGHEST_OCCUPIED: [],
        _PW_HIGHEST_OCCUPIED_LOWEST_FREE: [],
        _PW_KPTS: [],
        _PW_BANDS: [],
        _PW_BANDSTRUCTURE: [],
    }
    for idx, line in enumerate(pwo_lines):
        for identifier in indexes:
            if identifier in line:
                indexes[identifier].append(idx)
    return indexes


def parse_bands(
    fileobj,
) -> dict:
    """
    Parse Quantum Espresso output file and return a dictionary with the bands.

    Args:
        fileobj (fileobj): fileobj of the Quantum Espresso output file

    Returns:
        dict: dictionary mapping kpoint to list of tuples (energy, occupation)
    """
    # Get indexes of the lines where the different information is printed in the qe.out
    lines = fileobj.read().splitlines() # remove \n from lines
    fileobj.close() # back to beggining of file
    indexes = get_indexes(lines)
    # print(indexes)
    # Select lines with band info
    try:
        lines_with_band_energies = lines[indexes['End of self-consistent calculation'][0]:indexes['highest occupied, lowest unoccupied level'][0]]
    except:
        lines_with_band_energies = lines[indexes['End of self-consistent calculation'][0]:indexes['highest occupied level'][0]]
    bands = {}
    kpoint_indexes = {}
    occupation_indexes = {}
    number_of_kpoint = 0
    for index, line in enumerate(lines_with_band_energies):
        # If bands' in line, its the line with the kpoint
        if 'bands' in line:
            kpoint = line.split('(')[0].split('=')[1]
            kpoint = tuple(kpoint.split()) # format kpoint into tuple rather than str
            # print(kpoint)
            bands[kpoint] = {}
            kpoint_indexes.update({copy(number_of_kpoint): [deepcopy(kpoint), copy(index)] })
            number_of_kpoint += 1
        # Occupation string separates eigenvalues from occupations
        elif 'occupation' in line:
            occupation_indexes.update({copy(number_of_kpoint-1): copy(index)})
    
    for number_of_kpoint, kpoint_info in kpoint_indexes.items():
        kpoint = kpoint_info[0]
        kpoint_index = kpoint_info[1]
        try:
            index_of_next_kpoint = kpoint_indexes[number_of_kpoint+1][1]
        except KeyError: # for the last kpoint
            index_of_next_kpoint = -1 # last line
        bands[kpoint] = {
            'energies': lines_with_band_energies[kpoint_index+2:occupation_indexes[number_of_kpoint]],
            'occupations': lines_with_band_energies[occupation_indexes[number_of_kpoint]+1:index_of_next_kpoint],
            }
        # print(bands)
    return bands


def format_bands(
    bands: dict,
) -> dict:
    formatted_bands = {kpoint: {} for kpoint in list(bands.keys())}
    # Loop over kpoints
    for kpoint, value in bands.items():
        energies = []
        # Loop over all energy and occupation 
        # print("Len of energies", len(value['energies']))
        # print("Len of occupations", len(value['occupations']))
        for energy_line, occupation_line in zip(value['energies'], value['occupations']):
            # print(energy_line)
            # print(occupation_line)
            if energy_line != '' and occupation_line != '':
                energy_line_splited = [float(energy) for energy in energy_line.split()]
                occupation_line_splited = [float(occupation) for occupation in occupation_line.split()]
                # print(energy_line_splited, occupation_line_splited)
                for energy, occupation in zip(energy_line_splited, occupation_line_splited):
                    energies.append((energy, occupation))
        formatted_bands[kpoint] = energies
        # break
    return formatted_bands


def read_bands(
    fileobj,
) -> dict:
    """
    Parse electronic bands from a Quantum Espresso output file.

    Args:
        fileobj (fileobj): fileobj of the Quantum Espresso output file

    Returns:
        dict: dict mapping kpoint to list of tuples (energy, occupation)
    """
    bands = parse_bands(fileobj)
    return format_bands(bands)