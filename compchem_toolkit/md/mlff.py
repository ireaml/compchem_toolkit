import pymlff
from pymatgen.core.structure import Structure
import numpy as np


def get_structure_from_config(config: pymlff.core.Configuration):
    # Get list of species in configuration
    species = []
    for element in config.atom_types_numbers:
        # Add the element as many times as its dict value
        species += [element] * config.atom_types_numbers[element]
    s = Structure(
        lattice=config.lattice,
        coords=config.coords,
        species=species,
        coords_are_cartesian=True, # Cartesian coords!
    )
    return s


def parse_free_energy_from_outcar(path_to_outcar: str="OUTCAR"):
    """Parse TOTEN energies from OUTCAR file"""
    str_finished_ionic_step = "aborting loop because EDIFF is reached"
    str_free = "FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)"
    str_toten = "free  energy   TOTEN  =" #str_toten = "free energy    TOTEN  ="

    # Read the OUTCAR file
    with open(path_to_outcar, "r") as f:
        lines = f.readlines()

    # Get first line containing the string str_token after a line containing str_finished_ionic_step
    totens = []
    for i, line in enumerate(lines):
        if str_finished_ionic_step in line:
            found_toten = False
            for j in range(i, len(lines)):
                if str_toten in lines[j]: # str_free in lines[j] and
                    toten = lines[j].split(str_toten)[1].split("eV")[0]
                    #print(i, j, lines[j])
                    totens.append(float(toten))
                    found_toten = True
                    break
            if not found_toten:
                print(f"TOTEN not found after line {i}!") #break
            #break
    # Save totens to file
    arr = np.array(totens)
    np.save("totens.npy", arr)