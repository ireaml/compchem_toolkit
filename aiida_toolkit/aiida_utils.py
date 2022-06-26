### Useful imports and functions to work with aiida

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

def get_options_dict(
    computer: str,
):
    """
    Get the options for a given computer.
    """
    options = loadfn(os.path.join(MODULE_DIR, "yaml_files/options.yaml"))
    assert computer in options.keys()
    return options[computer]


def get_struct(
    pk: int
) -> Structure:
    """
    Get the output structure of a given node.
    """
    node = load_node(pk)
    try:
        struct = node.outputs.structure.get_pymatgen()
        return struct
    except:
        print("No output structure for that node!")
        return None