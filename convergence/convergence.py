# Collection of useful functions for vasp convergence testing

# Imports
import os
from six import b
import yaml
import numpy as np
from copy import deepcopy
import math
from monty.serialization import dumpfn, loadfn

# ase
import ase.io 
import ase.atoms

# kgrid
from kgrid.series import cutoff_series
from kgrid import calc_kpt_tuple

# Pymatgen
from pymatgen.io.vasp.inputs import Potcar, VaspInput, Incar, Poscar, Kpoints
from pymatgen.core.periodic_table import Element
from pymatgen.electronic_structure.core import Spin
import pymatgen.core.structure 
from pymatgen.core.structure  import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import UnknownPotcarWarning

import warnings
#warnings.filterwarnings('ignore') # ignore potcar warnings
warnings.filterwarnings(
    "ignore", category=UnknownPotcarWarning
)  # Ignore pymatgen POTCAR warnings

# inhouse scripts
from potcar import get_potcar_from_structure

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_kgrids_cutoffs(
    structure: pymatgen.core.structure.Structure,
    kmin: int=4, 
    kmax: int=20,
    emin: float=300, #in eV
    emax: float=900, #in eV
    ) -> list:
    """ Generate a series of kgrids for your lattice between a real-space cutoff range of `kmin` and `kmax` (in A). 
    For semiconductors, the default values (kmin: 4; kmax: 20) are generally good. 
    For metals you might consider increasing a bit the cutoff (kmax~30).
    Will also generate a series of energy cutoffs, between emin and emax, in eV.
    Returns a list of kmeshes and energy cutoffs (in eV).
    Args:
        atoms (ase.atoms.Atoms): _description_
        kmin (int, optional): _description_. Defaults to 4.
        kmax (int, optional): _description_. Defaults to 20.
        emin (float, optional): _description_. Defaults to 300.

    Returns:
        _type_: _description_
    """
    # Transform struct to atoms
    aaa = AseAtomsAdaptor()
    atoms = aaa.get_atoms(structure = structure)
    # Calculate kgrid samples for the given material
    kpoint_cutoffs = cutoff_series(atoms = atoms,
                                l_min= kmin, 
                                l_max= kmax, 
                                )
    kspacing = [np.pi / c for c in kpoint_cutoffs]
    kgrid_samples = [calc_kpt_tuple(
            atoms, cutoff_length=(cutoff - 1e-4)) for cutoff in kpoint_cutoffs]
    print(f"Kgrid samples: {kgrid_samples}")
    
    # And energy cutoffs
    cutoffs = range(emin,emax,50)

    return kgrid_samples, cutoffs

def write_config(
    kgrids : list,
    config_directory : str,
    emin: int= 300,
    emax: int= 900,
    name: str= None,
) -> None:
    # Read base config file
    file = '/home/ireaml/Python_Modules/Scripts/convergence/CONFIG'
    with open(file, 'r') as f:
        config = f.read()
    # Tranform kgrids to string
    strings_list = []
    for kgrid in kgrids:
        mystring = ' '.join([str(number) for number in kgrid])
        strings_list.append(mystring)
    krgids_string = ",".join(strings_list)

    if emin:
        config = config.replace("300", str(emin))
    if emax:
        config = config.replace("900", str(emax))
    if name:
        config = config.replace("name", name)
    # Add kgrids to config file and write it
    config = config.replace("kgrids", krgids_string)
    with open(f'{config_directory}/CONFIG', 'w') as f:
        f.write(config)

def setup_convergence(
    structure: Structure,
    directory: str, 
    kmin: int=4, 
    kmax: int=20,
    emin: float=300, #in eV
    emax: float=900, #in eV
    potcar_mapping: dict = None,
    write_input: bool = True,
) -> VaspInput :
    """
    Writes vasp input to then perform convergence testing with vaspup2.0.

    Args:
        structure (Structure): _description_
        directory (str): _description_
        kmin (int, optional): _description_. Defaults to 4.
        kmax (int, optional): _description_. Defaults to 20.
        emin (float, optional): _description_. Defaults to 300.
        write_input (bool, optional): _description_. Defaults to True.

    Returns:
        VaspInput: _description_
    """
    # Get kgrids
    kgrids = generate_kgrids_cutoffs(
        structure, 
        kmin = kmin, 
        kmax = kmax,
        emin = emin, #in eV
        emax = emax, #in eV
        )[0]
    # Poscar, potcar, kpoints
    poscar = Poscar(structure=structure)
    potcar = get_potcar_from_structure(
        structure, 
        potcar_mapping
        )
    kpoints = Kpoints(kpts=[[1,1,1]])

    # Default incar settings
    default_incar_settings = loadfn(os.path.join(MODULE_DIR, "incar_energy.yaml"))
    incar = Incar(default_incar_settings)

    # Vaspinput
    vaspinput = VaspInput(incar = incar, kpoints = kpoints, poscar = poscar, potcar = potcar)
    if write_input:
        vaspinput.write_input(output_dir = directory)

    # Write config
    assert type(kgrids) == list
    write_config(
        kgrids = kgrids,
        config_directory = directory,
        emin = emin, #in eV
        emax = emax, #in eV
        name = structure.composition.to_pretty_string()
    )
    return vaspinput