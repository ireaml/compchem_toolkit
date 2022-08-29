"""Common imports."""

from aiida import load_profile
load_profile('aiida-vasp')
# ! reentry scan

# Imports
import yaml
from copy import deepcopy
import os
import sys
import numpy as np

# aiida stuff
from aiida.orm.nodes.data.structure import StructureData
from aiida.orm import load_node, load_code, load_group, Code, Dict, Bool, QueryBuilder, WorkChainNode, Int
from aiida.plugins import DataFactory
from aiida.tools.groups import GroupPath
path = GroupPath()
FolderData = DataFactory('folder')

# pymatgen
from pymatgen.core.structure import Structure, Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import VaspInput, Incar, Poscar, Potcar, Kpoints
from  pymatgen.io.vasp.outputs import  Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Compchem toolkit stuff
from vasp_toolkit.input import get_default_number_of_bands, check_paralellization
from aiida_toolkit import relax, singleshot, parsing
from vasp_toolkit import output