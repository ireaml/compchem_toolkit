"""
Useful functions to work with aiida
"""

# Imports
import os

# aiida
import aiida
import numpy as np
from aiida import load_profile
from monty.serialization import dumpfn, loadfn

load_profile("aiida-vasp")
from aiida.engine import run, submit
from aiida.orm import (
    Bool,
    Code,
    Dict,
    QueryBuilder,
    WorkChainNode,
    load_code,
    load_group,
    load_node,
)
from aiida.orm.nodes.data.remote.base import RemoteData
from aiida.orm.nodes.data.structure import StructureData
from aiida.plugins import DataFactory
from aiida.tools import delete_nodes
from aiida_vasp.data.chargedensity import ChargedensityData
from aiida_vasp.data.wavefun import WavefunData

FolderData = DataFactory("folder")

# pymatgen
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Outcar, Vasprun

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
    pk: int,
    output: bool = True,
) -> Structure:
    """
    Get the output or input structure of a given node.
    """
    node = load_node(pk)
    try:
        if output:
            struct = node.outputs.structure.get_pymatgen()
        else:
            struct = node.inputs.structure.get_pymatgen()
        return struct
    except:
        if output:
            print("No output structure for that node!")
        else:
            print("No input structure for that node!")
        return None
