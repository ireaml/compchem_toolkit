"""
Useful imports and functions to submit VASP relaxation workchains with aiida
"""

from aiida import load_profile
from aiida.engine.processes.workchains.workchain import WorkChain
load_profile('aiida-vasp')

# Imports
import os
import numpy as np
from copy import deepcopy
from typing import Optional
from monty.io import zopen
from monty.serialization import dumpfn, loadfn

# aiida
import aiida
from aiida.orm.nodes.data.structure import StructureData
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Code, Dict, Bool, WorkChainNode
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import run, submit
from aiida_vasp.data.chargedensity import ChargedensityData
from aiida_vasp.data.wavefun import WavefunData
from aiida.orm.nodes.data.remote.base import RemoteData
from aiida.tools.groups import GroupPath
path = GroupPath()
FolderData = DataFactory('folder')

# pymatgen
import pymatgen
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import VaspInput, Incar, Poscar, Potcar, Kpoints
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.electronic_structure.core import Spin

# in house vasp functions
from vasp_toolkit.input import get_potcar_mapping

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_incar_settings = loadfn(os.path.join(MODULE_DIR, "yaml_files/vasp/relax_incar.yaml"))
default_potcar_dict = loadfn(os.path.join(MODULE_DIR, "yaml_files/vasp/default_POTCARs.yaml"))

# Potcar family
potcar_family = 'PAW_PBE_54'

def submit_vasp_relax(
    structure: Structure,
    code_string: str,
    kmesh: tuple,
    incar_settings: dict,
    label: str,
    options: dict,
    algo: str = 'cg', # Optimization algorithm, default to conjugate gradient
    use_default_incar_settings: bool = False,
    positions: bool = True,
    shape: bool = False,
    volume: bool = False,
    ionic_steps: int = 300,
    potential_mapping: Optional[dict] = None,
    settings: Optional[dict] = None,
    metadata: Optional[dict] = None,
    chgcar: Optional[ChargedensityData] = None,
    wavecar: Optional[WavefunData] = None,
    remote_folder: Optional[RemoteData] = None,
    clean_workdir: Optional[bool] = False,
    dynamics: Optional[dict] = None,
    group_label: Optional[str] = None,
) -> WorkChain:
    """
    Submit vasp relaxation.

    Args:
        structure (Structure):
            _description_
        code_string (str):
            _description_
        kmesh (tuple):
            _description_
        incar_options (dict):
            _description_
        algo (str, optional):
            Algorithm for ionic minimization: conjugate gradient (cg) or pseudo Newton(). Defaults to 'cg'.
        defaulttoconjugategradientpositions (bool, optional):
            _description_. Defaults to True.
        shape (bool, optional):
            _description_. Defaults to False.
        volume (bool, optional):
            _description_. Defaults to False.
        ionic_steps (int, optional):
            _description_. Defaults to 300.
        potential_mapping (Optional[dict], optional):
            _description_. Defaults to None.
        settings (Optional[dict], optional):
            _description_. Defaults to None.
        remote_folder (Optional[RemoteData], optional):
            _description_. Defaults to None.
        group_label (Optional[str], optional):
            Group label to save the node to.
    Returns:
        WorkChain: aiida Workchain object
    """
    # We set the workchain you would like to call
    workchain = WorkflowFactory('vasp.relax')

    # And finally, we declare the options, settings and input containers
    settings = AttributeDict()
    inputs = AttributeDict()

    # Organize settings
    settings.parser_settings = {}

    # Set inputs for the following WorkChain execution

    # Set code
    inputs.code = Code.get_from_string(code_string)

    # Set structure
    if not isinstance(structure, pymatgen.core.structure.Structure): # in case it's pmg interface or slab object
        structure = Structure(
            species = structure.species,
            lattice = structure.lattice,
            coords= structure.frac_coords,
            coords_are_cartesian= False,
        )
    sorted_structure = structure.get_sorted_structure()
    if sorted_structure != structure:
        print("Structure is not sorted, sorting now. \n Quit if you had set MAGMOM's or selective dynamics for the unsorted structure!")
    inputs.structure = StructureData(pymatgen =structure)

    # Set k-points grid density
    kpoints = DataFactory('array.kpoints')()
    kpoints.set_kpoints_mesh(kmesh)
    inputs.kpoints = kpoints

    # Set INCAR parameters
    default_incar_settings_copy = deepcopy(default_incar_settings)
    if use_default_incar_settings:
        default_incar_settings_copy.update(incar_settings)
    else:
        default_incar_settings_copy = deepcopy(incar_settings)
    # Check no typos in keys
    incar = Incar(default_incar_settings_copy)
    incar.check_params() # check keys
    incar_dict = incar.as_dict() ; del incar_dict['@class'] ; del incar_dict['@module']
    # Check ICHARG tag
    if incar_dict.get('ICHARG') != 2:
        if (incar_dict.get('ICHARG') in [1,11]) and (not remote_folder): # needs CHGCAR as input, so make sure we're giving it
            assert chgcar is not None, "CHGCAR is required for ICHARG = 1 or 11"
        if (incar_dict.get('ICHARG') == 0) and (not remote_folder): # needs WAVECAR as input, so make sure we're giving it
            assert wavecar is not None, "WAVECAR is required for ICHARG = 0"
    inputs.parameters = DataFactory('dict')(dict= {'incar': incar_dict})

    # Set potentials and their mapping
    inputs.potential_family = DataFactory('str')(potcar_family)
    if potential_mapping:
        inputs.potential_mapping = DataFactory('dict')(dict=potential_mapping)
    else:
        inputs.potential_mapping = DataFactory('dict')(dict= get_potcar_mapping(structure = structure))

    # Set options
    inputs.options = DataFactory('dict')(dict=options)

    # Set settings
    default_settings = {
        'parser_settings': {
            'misc': ['total_energies', 'maximum_force', 'maximum_stress', 'run_status', 'run_stats', 'notifications'],
            'add_structure': True, # retrieve structure and kpoints
            # 'add_kpoints': True,
            # 'add_forces' : True,
            'add_energies': True,
            # 'add_bands': True,
            # 'add_trajectory': True,
            },
            }
    if settings:
        default_settings.update(settings)
    inputs.settings = DataFactory('dict')(dict=default_settings)

    # Metadata
    if metadata:
        inputs.metadata = metadata
    else:
        if not label:
            formula = structure.composition.to_pretty_string()
            label = f'relax_{formula}'
        inputs.metadata = {'label': label}

    # dynamics
    if dynamics:
        inputs.dynamics = dynamics

    # Set workchain related inputs, in this case, give more explicit output to report
    inputs.verbose = DataFactory('bool')(True)

    # Relaxation related parameters that is passed to the relax workchain
    relax = AttributeDict()

    # Turn on relaxation
    relax.perform = DataFactory('bool')(True)

    # Select relaxation algorithm
    relax.algo = DataFactory('str')(algo)

    # Set force cutoff limit (EDIFFG, but no sign needed)
    try:
        force_cutoff = abs(default_incar_settings_copy['EDIFFG'])
        relax.force_cutoff = DataFactory('float')(force_cutoff)
    except KeyError:
        relax.force_cutoff = DataFactory('float')(0.01)

    # Turn on relaxation of positions (strictly not needed as the default is on)
    # The three next parameters correspond to the well known ISIF=3 setting
    relax.positions = DataFactory('bool')(positions)
    # Relaxation of the cell shape (defaults to False)
    relax.shape = DataFactory('bool')(shape)
    # Relaxation of the volume (defaults to False)
    relax.volume = DataFactory('bool')(volume)
    # Set maximum number of ionic steps
    relax.steps = DataFactory('int')(ionic_steps)
    # Set the relaxation parameters on the inputs
    inputs.relax = relax

    # Chgcar and wavecar
    if chgcar:
        inputs.chgcar = chgcar
    if wavecar:
        inputs.wavecar = wavecar
    if remote_folder:
        inputs.restart_folder = remote_folder

    # Clean Workdir
    # If True, clean the work dir upon the completion of a successfull calculation.
    inputs.clean_workdir = Bool(clean_workdir)

    # Submit the requested workchain with the supplied inputs
    workchain = submit(workchain, **inputs)

    if group_label:
        group = path[group_label].get_or_create_group()
        group = path[group_label].get_group()
        group.add_nodes(workchain)
        print(
            f"Submitted relax workchain with pk: {workchain.pk} and label {workchain.label}, "
            f"stored in group with label {group_label}"
        )
    else:
        print(f"Submitted relax workchain with pk: {workchain.pk} and label {workchain.label}")

    return workchain