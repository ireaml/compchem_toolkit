"""
Collection of useful functions to analyse vasp output
"""

# Imports
from collections import defaultdict
import os
import numpy as np
from copy import deepcopy
from typing import Optional
from monty.io import zopen
from monty.serialization import dumpfn, loadfn
import pandas as pd
from pandas.core.frame import DataFrame

# pymatgen
import pymatgen
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Wavecar, Poscar, Structure
# from pymatgen.io.ase import AseAtomsAdaptor
# from pymatgen.io.vasp.inputs import VaspInput, Incar, Poscar, Potcar, Kpoints
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import BSVasprun

# SUMO
from sumo.electronic_structure.dos import load_dos, get_pdos
from sumo.plotting.dos_plotter import SDOSPlotter

# Matplotlib
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

# In-house stuff
from vasp_toolkit.potcar import get_potcar_from_structure, get_valence_orbitals_from_potcar

def analyse_procar(
    bsvasprun: BSVasprun,
    list_of_atom_indexes: list,
    threshold: float = 0.06,
    verbose: bool= True,
) -> DataFrame:
    """ 
    Determines the energy states localized in the atoms speficied in list_of_atom_indexes.
    
    Args:
        bsvasprun (BSVasprun):
            pymatgen.io.vasp.outputs.BSVasprun object
        list_of_atom_indexes (list):
            list of site indexes for which to determine their energy states
        threshold (float, optional):
            threshold for the site projection of the orbital. Only orbitals with \
            higher projections on all sites in `list_of_atom_indexes` will be returned. 
            Defaults to 0.06.
        verbose (bool, optional):
            Whether to print the energy states. 
            Defaults to True.
    Returns:
        DataFrame
    """
    # bsvasprun.projected_eigenvalues[spin][kpoint][band][orbital]
    number_of_bands = len(bsvasprun.projected_eigenvalues[Spin(1)][0])
    number_of_kpoints = len(bsvasprun.projected_eigenvalues[Spin(1)])
    projections = {}
    selected_bands = {}
    print("Analysing PROCAR. All indexing starts at 0 (pythonic), so add 1 to change to VASP indexing!")
    for band in range(0, number_of_bands):
        projections[band] = {} # store projections on selected sites
        selected_bands[band] = {} # bands with significant projection on sites, for output
        for kpoint in range(0, number_of_kpoints):
            projections[band][kpoint] = {}
            atom_projections = {}
            for atom_index in list_of_atom_indexes: # Filter projections of selected sites
                atom_projections[atom_index] = round(
                    sum(bsvasprun.projected_eigenvalues[Spin(1)][kpoint][band][atom_index]), 
                    3,
                )                                         

            if all(
                atom_projection > threshold # projection higher than threshold for all sites
                for atom_projection in atom_projections.values()
            ):
                selected_bands[band][kpoint] = {
                    index: round(sum(value), 3) for index, value in enumerate(bsvasprun.projected_eigenvalues[Spin(1)][kpoint][band]) 
                     if sum(value) > threshold
                } # add sites with significant contributions
                if verbose:
                    print("Band: ", band, "Kpoint: ", kpoint)
                    print(selected_bands[band][kpoint])
                    print("-------")
    df = pd.DataFrame.from_dict(selected_bands)
    return df


def plot_dos(
    vasprun_path: str,
    elements_orbitals: Optional[dict] = None,
    structure: Optional[pymatgen.core.structure.Structure] = None,
    gaussian: Optional[float] = 0.06,
    xmin: Optional[float] = -3.0,
    xmax: Optional[float] = 3.0,
) -> matplotlib.axes.Axes:
    """
    Quickly plot orbital projected DOS using SUMO.

    Args:
        vasprun_path (str): _description_
        elements_orbitals (Optional[dict]): _description_
        structure (Optional[pymatgen.core.structure.Structure]): _description_
        gaussian (Optional[float], optional): _description_. Defaults to 0.06.
        xmin (Optional[float], optional): _description_. Defaults to -3.0.
        xmax (Optional[float], optional): _description_. Defaults to 3.0.

    Returns:
        matplotlib.axes.Axes
    """
    # If user didnt specify orbitals, use valence orbitals of POTCAR
    if not elements_orbitals:
        elements_orbitals = get_valence_orbitals_from_potcar( 
            potcar = get_potcar_from_structure(structure = structure)
        )
    # Load DOS and plot 
    dos, pdos = load_dos(
        vasprun = vasprun_path,
        elements = elements_orbitals,
        gaussian = gaussian,
     )

    sdosplotter = SDOSPlotter(
        dos = dos,
        pdos = pdos,
    )
    myplot = sdosplotter.get_plot( 
        xmin= xmin,
        xmax= xmax,
        fonts = ['Whitney Light', 'Whitney Book']
        )
    return myplot


def make_parchg(
    wavecar: str, 
    contcar: str, 
    vasp_band_index: int, 
    parchg_filename: Optional[str] = None, 
    spin: int = 0, 
    kpoint: int=0, 
    phase: bool=False,
    vasp_type: str='std', # or ncl or gam
) -> None:
    if not parchg_filename:
        parchg_filename = f'./PARCHG_band_{vasp_band_index}'
    my_wavecar = Wavecar(filename=wavecar, vasp_type=vasp_type) # or std or gam
    my_poscar = Poscar(Structure.from_file(contcar))
    my_parchg = my_wavecar.get_parchg(
        my_poscar, 
        kpoint=kpoint, # first kpoint (Gamma in this case) 
        band=vasp_band_index-1, # convert from python to vasp indexing
        spin=spin, # 0 = up, 1 = down
        phase=phase # show phase changes in density if present
        ) 
    my_parchg.write_file(f"{parchg_filename}.vasp")