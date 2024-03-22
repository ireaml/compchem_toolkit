"""Common imports."""

from aiida import load_profile

load_profile("aiida-vasp")

import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

# Imports
import yaml
from aiida.orm import (
    Bool,
    Code,
    Dict,
    Int,
    QueryBuilder,
    WorkChainNode,
    load_code,
    load_group,
    load_node,
)

# aiida stuff
from aiida.orm.nodes.data.structure import StructureData
from aiida.plugins import DataFactory
from aiida.tools.groups import GroupPath

path = GroupPath()
FolderData = DataFactory("folder")

# pymatgen
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Vasprun
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Compchem toolkit stuff
from aiida_tools import aiida_utils, parsing, relax, singleshot
from vasp import input, output, potcar
from vasp.input import check_paralellization, get_default_number_of_bands

# Mpl style sheet
file = "/home/ireaml/Python_Modules/mpl_style/publication_style.mplstyle"
if os.path.exists(file):
    plt.style.use(file)
