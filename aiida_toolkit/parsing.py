import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import subprocess
from typing import Optional
import warnings

### Useful imports and functions to work with aiida
from aiida import load_profile
from aiida.engine.processes.workchains.workchain import WorkChain
load_profile('aiida-vasp')

# aiida
import aiida
from aiida.orm import load_node
from aiida.orm.nodes.data.structure import StructureData
from aiida.common.extendeddicts import AttributeDict
from aiida.orm import Code, Dict, Bool, WorkChainNode
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine import submit
FolderData = DataFactory('folder')
from aiida_vasp.data.chargedensity import ChargedensityData
from aiida_vasp.data.wavefun import WavefunData
from aiida.orm.nodes.data.remote.base import RemoteData

# pymatgen
import pymatgen
from pymatgen.core.structure import Structure
# from pymatgen.io.ase import AseAtomsAdaptor
# from pymatgen.io.vasp.inputs import VaspInput, Incar, Poscar, Potcar, Kpoints
from pymatgen.io.vasp.outputs import Outcar, Vasprun, BSVasprun, Chgcar
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import BSVasprun
from pymatgen.electronic_structure.core import Spin

# In-house stuff
from vasp_toolkit.potcar import get_potcar_from_structure, get_valence_orbitals_from_potcar
from vasp_toolkit.output import plot_dos


def get_vasprun_from_pk(
    pk: int,
    remove_file: bool = True,
    vasprun_type: str = 'Vasprun' 
) -> pymatgen.io.vasp.outputs.Vasprun:
    """_summary_

    Args:
        pk (int): pk of aiida node
        remove_file (bool, optional): whether to remove the file after reading it. Defaults to True.
        vasprun_type (str, optional): either 'Vasprun' or 'BSVasprun'. Defaults to 'Vasprun'.
    Returns:
        vasprun: Vasprun object
    """
    node = load_node(pk)
    label = node.label
    # Read vasprun
    assert 'vasprun.xml' in node.outputs.retrieved.list_object_names()
    vasprun_content = node.outputs.retrieved.get_object_content('vasprun.xml')
    vasprun_path = f'vasprun_{label}.xml'
    with open(vasprun_path, 'w') as ff:
        ff.write(vasprun_content)
    if vasprun_type == 'Vasprun':
        vasprun = Vasprun(vasprun_path)
    elif vasprun_type == 'BSVasprun':
        vasprun = BSVasprun(vasprun_path)
    if remove_file:
        os.remove(vasprun_path)
    else:
        print(f"Vasprun saved at {vasprun_path}")
    return vasprun


def get_outcar_from_pk(
    pk: int,
    remove_file: bool = True,
    folder_name: str = None,
) -> pymatgen.io.vasp.outputs.Outcar:
    """
    From pk, returns the associated Outcar object

    Args:
        pk (int): 
            pk of aiida node
        remove_file (bool, optional): 
            Whether to remove the file after reading it. Defaults to True.
        folder_name (str, optional):
            Name of folder to store the file. 
            Defaults to './outcars'
    Returns:
        outcar: Outcar object
    """
    node = load_node(pk)
    if not node.outputs.misc['run_status']['finished']:
        warnings.warn( "Calculation didnt finished OK!")
    label = node.label
    # Read 
    assert 'OUTCAR' in node.outputs.retrieved.list_object_names()
    outcar_content = node.outputs.retrieved.get_object_content('OUTCAR')
    if not folder_name:
        folder_name = f"./outcars"
        outcar_path = f'{folder_name}/OUTCAR_{label}'
    else:
        folder_name = f"./{folder_name}"
        outcar_path = f'{folder_name}/OUTCAR'
    if not os.path.exists(outcar_path) and not os.path.exists(folder_name):
        os.mkdir(folder_name)
    if not os.path.exists(outcar_path):
        with open(outcar_path, 'w') as ff:
            ff.write(outcar_content)
    outcar = Outcar(outcar_path)
    if remove_file:
        os.remove(outcar_path)
    return outcar


def get_dos_from_pk(
    pk: int,
    elements_orbitals: dict = None,
    gaussian: float = 0.06,
    xmin: float = -3.0,
    xmax: float = 3.0,
    remove_file: bool = True,
    computer: Optional[str] = None,
    **kwargs, # Other keyword arguments accepted by SDOSPlotter.get_plot()
) -> mpl.axes.Axes:
    """_summary_

    Args:
        pk (int): pk of aiida node
        elements_orbitals (dict, optional): dict matching element to valence orbitals to plot in projected DOS. Defaults to None.
        gaussian (float, optional): _description_. Defaults to 0.06.
        xmin (float, optional): _description_. Defaults to -3.0.
        xmax (float, optional): _description_. Defaults to 3.0.
        kaargs (dict, optional): _description_. Defaults to {}.

    Returns:
        _type_: _description_
    """
    node = load_node(pk)
    label = node.label
    remote_path = node.outputs.remote_folder.attributes['remote_path']
    vasprun_path = f'./vaspruns/vasprun_{label}.xml'
    if not os.path.exists(vasprun_path) and not os.path.exists('./vaspruns'):
        os.mkdir('./vaspruns')
    if computer:
        subprocess.run(["rsync", "-azvu", f"{computer}:{remote_path}/vasprun.xml", f"./vaspruns/vasprun_{label}.xml"])
    else: # else parse from aiida outputs
        # Read vasprun # This can crash the kernel, so lets transfer with rsync
        assert 'vasprun.xml' in node.outputs.retrieved.list_object_names()
        vasprun_content = node.outputs.retrieved.get_object_content('vasprun.xml')
        if not os.path.exists(vasprun_path):
            with open(vasprun_path, 'w') as ff:
                ff.write(vasprun_content)
                
    # If user didnt specify orbitals, use valence orbitals of POTCAR
    if not elements_orbitals:
        struct = node.inputs.structure.get_pymatgen()
        elements_orbitals = get_valence_orbitals_from_potcar( 
            potcar = get_potcar_from_structure(structure = struct)
        )
    myplot = plot_dos(
        vasprun_path = vasprun_path,
        elements_orbitals=elements_orbitals,
        gaussian = gaussian,
        xmin= xmin,
        xmax= xmax,
    )
    if remove_file and os.file.exists(vasprun_path):
        os.remove(vasprun_path) # delete vasprun file after plotting DOS
    return myplot


def transfer_chgcar(
    pk: int,
    remote_computer: str = None,
) -> str:
    """
    Transfer the CHGCAR file from the remote computer where the aiida calculation was run. 
    """
    node = load_node(pk)
    assert node.outputs.remote_folder.attributes["remote_path"], f"No remote path for pk {pk}"
    remote_path = node.outputs.remote_folder.attributes["remote_path"]
    node_label = node.label
    abs_path = os.getcwd()
    if not os.path.exists('./CHGCARS'):
        os.mkdir('./CHGCARS')
    shell_output = subprocess.run([
        "rsync", "-azvus", f"{remote_computer}:{remote_path}/CHGCAR", f"{abs_path}/CHGCARS/{node_label}_CHGCAR"
    ])
    if shell_output.returncode == 0:
        print(f"Saved CHGCAR as {abs_path}/CHGCARS/{node_label}_CHGCAR")
    else:
        print(f"Error in tranfer_chgcar: {shell_output}")
    return f"{abs_path}/CHGCARS/{node_label}_CHGCAR"


def parse_chgcar(
    pk: int,
    remote_computer: str = None,
) -> Chgcar:
    filename = transfer_chgcar(pk = pk, remote_computer = remote_computer)
    chgcar = Chgcar.from_file(filename)
    return chgcar


def get_charge_density_data_from_pk(
    pk: int,
    remote_computer: str = None,
) -> ChargedensityData:
    """
    Returns the ChargedensityData object from the pk of aiida node.
    """
    filename = transfer_chgcar(pk = pk, remote_computer = remote_computer)
    return ChargedensityData(filename)


def transfer_vasp_files(
    pk: int,
    remote_computer: str,
    folder_name: str = None,
    list_of_files: list = None,
) -> list:
    """Parses files from remote computer and transfers them to local computer."""
    node = load_node(pk)
    # Remote path
    assert node.outputs.remote_folder.attributes["remote_path"], f"No remote path for pk {pk}"
    remote_path = node.outputs.remote_folder.attributes["remote_path"]
    node_label = node.label
    paths_to_return = []
    
    # Local path
    abs_path = os.getcwd()
    if not folder_name:
        folder_name = f"{abs_path}/VASP_files"
    else:
        folder_name = f"{abs_path}/{folder_name}"
    if not os.path.exists(f'{folder_name}'):
        os.mkdir(f"{folder_name}")
    # Transfer
    for vasp_file in list_of_files: 
        if folder_name == "VASP_files":
            file_path = f"{folder_name}/{node_label}_{vasp_file}"
        else:
            file_path = f"{folder_name}/{vasp_file}"
        
        shell_output = subprocess.run([
            "rsync", "-azvus", f"{remote_computer}:{remote_path}/{vasp_file}", f"{file_path}"
        ])
        if shell_output.returncode == 0:
            print(f"Saved {vasp_file} as {file_path}")
            paths_to_return.append(f"{file_path}")
        else:
            print(f"Error in tranfer: {shell_output}")
    return paths_to_return