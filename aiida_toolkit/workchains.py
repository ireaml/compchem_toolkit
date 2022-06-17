# Imports
import os
import numpy as np
from monty.serialization import dumpfn, loadfn

# aiida
import aiida
from aiida import load_profile
load_profile('aiida-vasp')
from aiida.orm.nodes.data.structure import StructureData
from aiida.orm import load_node, load_code, load_group, Code, Dict, Bool, QueryBuilder, WorkChainNode
from aiida.plugins import DataFactory
from aiida.engine import run, submit
from aiida.tools import delete_nodes
from aiida_vasp.data.chargedensity import ChargedensityData
from aiida_vasp.data.wavefun import WavefunData
from aiida.orm.nodes.data.remote.base import RemoteData
FolderData = DataFactory('folder')

# pymatgen
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import VaspInput, Incar, Poscar, Potcar, Kpoints
from pymatgen.io.vasp.outputs import Outcar, Vasprun
from pymatgen.electronic_structure.core import Spin

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_vasp_workchain(
    structure: aiida.orm.nodes.data.structure.StructureData, 
    incar: dict, 
    kgrid: tuple,
    potential_mapping: dict,
    code_string: str,
    options: dict ,
    metadata : dict = None,
    settings: dict = None,
    dynamics: dict = None,
    chgcar: ChargedensityData = None,
    wavecar: WavefunData = None,
    restart_folder: RemoteData = None,
    clean_workdir: bool = False,
    ):
    """
    Setup the inputs for a VaspWorkChain.
    """
    from aiida.orm import Code, Dict, Str, Bool
    from aiida.orm.nodes.data.list import List
    from aiida_vasp.utils.aiida_utils import get_data_node
    from aiida.common.extendeddicts import AttributeDict
    from aiida.plugins import DataFactory, WorkflowFactory
    
    basevasp = WorkflowFactory('vasp.vasp')
    # Then, we set the workchain you would like to call
    workchain = WorkflowFactory('vasp.verify')

    inputs = AttributeDict()

    # Code
    code = Code.get_from_string(code_string)
    inputs.code = code

    # Structure
    inputs.structure = structure
    formula = structure.get_pymatgen().formula.replace(' ', '') # Avoid spaces in labels in formula's name

    # Parameters
    inputs.parameters = Dict(dict={'incar': incar})

    # Kpoints
    kpoints = get_data_node('array.kpoints')
    kpoints.set_kpoints_mesh(kgrid)
    inputs.kpoints = kpoints

    # Pseudos
    inputs.potential_family = Str('PBE_msc')
    inputs.potential_mapping = Dict(dict=potential_mapping)

    # Options
    inputs.options = Dict(dict = options)
    # Parser Settings
    if not settings:
        settings = {
            'parser_settings': {
                'misc': [
                    'total_energies', 'maximum_force', 'maximum_stress', 'run_status', 'run_stats', 'notifications'
                    ],
                'add_structure': True, # retrieve structure and kpoints
                'add_kpoints': True,
                'add_forces' : True,
                'add_energies': True,
                'add_trajectory': True,
                }
            }
    inputs.settings = Dict(dict = settings)

    # Metadata
    if not metadata:
        metadata = {
                'label': f'Relax_{formula}_slab',
                'description': 'PBEsol-D3 VASP relaxation of FASI slab',
                }
    inputs.metadata = metadata
    
    # Clean Workdir
    # If True, clean the work dir upon the completion of a successfull calculation.
    inputs.clean_workdir = Bool(clean_workdir)

    # Dynamics for constrained relaxation
    if dynamics: 
        selective_dynamics = {'positions_dof' : [] }
        assert type(dynamics['positions_dof']) == list
        selective_dynamics['positions_dof'] = List( list=dynamics['positions_dof']) 
        inputs.dynamics = selective_dynamics # Dict(dict = selective_dynamics )

    # Chgcar and wavecar
    if chgcar:
        inputs.chgcar = chgcar
    if wavecar:
        inputs.wavecar = wavecar
    if restart_folder:
        inputs.restart_folder = restart_folder

    return workchain, inputs