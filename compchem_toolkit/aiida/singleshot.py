"""Useful imports and functions to submit VASP singleshot workchains with aiida."""

from email.policy import default

from aiida import load_profile
from aiida.engine.processes.workchains.workchain import WorkChain

load_profile("aiida-vasp")

# Imports
import os
from copy import deepcopy
from typing import Optional

# aiida
import aiida
import numpy as np
from aiida.common.extendeddicts import AttributeDict
from aiida.engine import run, submit
from aiida.orm import Bool, Code, Dict, WorkChainNode, load_node
from aiida.orm.nodes.data.remote.base import RemoteData
from aiida.orm.nodes.data.structure import StructureData
from aiida.plugins import CalculationFactory, DataFactory, WorkflowFactory
from aiida.tools.groups import GroupPath
from aiida_vasp.data.chargedensity import ChargedensityData
from aiida_vasp.data.wavefun import WavefunData
from monty.io import zopen
from monty.serialization import dumpfn, loadfn

path = GroupPath()
FolderData = DataFactory("folder")

# pymatgen
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar, Potcar, VaspInput
from pymatgen.io.vasp.outputs import Outcar, Vasprun

from aiida_toolkit import parsing
# in house vasp functions
from vasp_toolkit.input import get_default_number_of_bands, get_potcar_mapping

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
incar_settings_vasp_ncl = loadfn(
    os.path.join(MODULE_DIR, "yaml_files/vasp/incar_vasp_ncl_singleshot.yaml")
)
incar_settings_vasp_std = loadfn(
    os.path.join(MODULE_DIR, "yaml_files/vasp/incar_vasp_std_singleshot.yaml")
)
default_potcar_dict = loadfn(
    os.path.join(MODULE_DIR, "yaml_files/vasp/default_POTCARs.yaml")
)

# Potcar family
potcar_family = "PAW_PBE_54"


def submit_vasp_singleshot(
    structure: Structure,
    code_string: str,
    kmesh: tuple,
    incar_settings: dict,
    label: str,
    options: dict,
    use_default_incar_settings: bool = False,
    spin_orbit: bool = False,
    functional: str = "HSE06",
    potential_mapping: Optional[dict] = None,
    settings: Optional[dict] = None,
    chgcar: Optional[ChargedensityData] = None,
    wavecar: Optional[WavefunData] = None,
    remote_folder: Optional[RemoteData] = None,
    clean_workdir: Optional[bool] = False,
    group_label: Optional[str] = None,
) -> WorkChain:
    """
    Submit VASP singleshot workchain with aiida.

    Args:
        structure (Structure):
            Input structure as a pymatgen.core.structure.Structure object.
        code_string (str):
            _description_
        kmesh (tuple):
            _description_
        incar_options (dict):
            _description_
        use_default_incar_settings (bool):
            whether to update the default INCAR settings with the user defined ones
        potential_mapping (Optional[dict], optional):
            _description_. Defaults to None.
        settings (Optional[dict], optional):
            _description_. Defaults to None.
        label (str):
            label for the workchain node
        spin_orbit (bool):
            whether to include spin-orbit coupling
        group_label (Optional[str], optional):
            Group label to save the node to.
    Returns:
        WorkChain: _description_
    """

    def check_number_of_bands(
        structure,
        default_incar_settings_copy,
    ) -> dict:
        """Check if bands have been doubled for spin orbit coupling"""
        if default_incar_settings_copy.get("NELECT"):
            nelect = default_incar_settings_copy.get("NELECT")
        else:
            nelect = None

        if default_incar_settings_copy.get("LSORBIT") == True:
            default_nbands = get_default_number_of_bands(
                structure=structure,
                number_of_electrons=nelect,
            )
            if (
                not default_incar_settings_copy.get("NBANDS")
                or default_incar_settings_copy.get("NBANDS") < 2 * default_nbands
            ):
                default_incar_settings_copy.update({"NBANDS": 2 * default_nbands + 5})
            if not "ncl" in code_string:
                raise ValueError(
                    f"The LSORBIT keyword is set to True, but you're using {code_string}! Skipping!"
                )

        else:
            default_nbands = get_default_number_of_bands(
                structure=structure,
                number_of_electrons=nelect,
            )
            if (
                not default_incar_settings_copy.get("NBANDS")
                or default_incar_settings_copy.get("NBANDS") <= default_nbands
            ):
                default_incar_settings_copy["NBANDS"] = default_nbands + 5

        return default_incar_settings_copy

    def check_vdw_parameters(
        default_incar_settings_copy,
    ) -> dict:
        """Check IVDW parameters for HSE06/PBE0"""
        if default_incar_settings_copy.get("LHFCALC") == True:
            if default_incar_settings_copy.get("AEXX") != 0.0:
                default_incar_settings_copy.update(
                    {
                        "VDW_S8": 0.928,
                        "VDW_SR": 1.287,
                    }
                )  # From https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/dft-d3/functional
        return default_incar_settings_copy

    def check_icharg(incar_dict) -> None:
        """Check ICHARG tag"""
        if incar_dict.get("ICHARG") != 2:
            if incar_dict.get("ICHARG") in [
                1,
                11,
            ]:  # needs CHGCAR as input, so make sure we're giving it
                assert chgcar is not None, "CHGCAR is required for ICHARG = 1 or 11"
            if (
                incar_dict.get("ICHARG") == 0
            ):  # needs WAVECAR as input, so make sure we're giving it
                assert wavecar is not None, "WAVECAR is required for ICHARG = 0"

    def check_typos(default_incar_settings_copy) -> dict:
        """Check typos in keys"""
        incar = Incar(default_incar_settings_copy)
        incar.check_params()  # check keys
        incar_dict = incar.as_dict()
        del incar_dict["@class"]
        del incar_dict["@module"]
        return incar_dict

    def setup_incar(use_default_incar_settings, incar_settings) -> dict:
        """Setup incar parameters"""
        # Dictionary matching common functionals to their INCAR tags
        functionals = {
            "PBE": {"LHFCALC": False, "HFSCREEN": 0, "AEXX": 0},
            "HSE06": {"LHFCALC": True, "HFSCREEN": 0.2, "AEXX": 0.25},
            "PBE0": {"LHFCALC": True, "HFSCREEN": 0, "AEXX": 0.25},
        }
        if spin_orbit:
            default_incar_settings_copy = deepcopy(incar_settings_vasp_ncl)
        else:
            default_incar_settings_copy = deepcopy(incar_settings_vasp_std)
        if functional:
            default_incar_settings_copy.update(functionals[functional])
        if (
            use_default_incar_settings
        ):  # Update the default settings with the user-defined settings
            default_incar_settings_copy.update(incar_settings)
        else:  # Use the user-defined settings
            default_incar_settings_copy = deepcopy(incar_settings)
        default_incar_settings_copy["LAECHG"] = True  # For Bader analysis
        return default_incar_settings_copy

    # We set the workchain you would like to call
    workchain = WorkflowFactory("vasp.vasp")
    # Then, we set the workchain you would like to call
    # workchain = WorkflowFactory('vasp.verify')

    # And finally, we declare the options, settings and input containers
    settings = AttributeDict()
    inputs = AttributeDict()

    # Organize settings
    settings.parser_settings = {}

    # Set inputs for the following WorkChain execution

    # Set code
    inputs.code = Code.get_from_string(code_string)

    # Set structure
    inputs.structure = StructureData(pymatgen=structure)

    # Set k-points grid density
    kpoints = DataFactory("array.kpoints")()
    kpoints.set_kpoints_mesh(kmesh)
    inputs.kpoints = kpoints

    # Set INCAR parameters
    default_incar_settings_copy = setup_incar(
        use_default_incar_settings=use_default_incar_settings,
        incar_settings=incar_settings,
    )
    # Check IVDW parameters for HSE06
    default_incar_settings_copy = check_vdw_parameters(default_incar_settings_copy)
    # If LSORBIT, check if bands have been doubled:
    default_incar_settings_copy = check_number_of_bands(
        structure, default_incar_settings_copy
    )
    # Check no typos in keys
    incar_dict = check_typos(default_incar_settings_copy)
    # Check ICHARG tag
    check_icharg(incar_dict)
    inputs.parameters = DataFactory("dict")(dict={"incar": incar_dict})

    # Set potentials and their mapping. If user does not specify them, we use the VASP recommended ones
    inputs.potential_family = DataFactory("str")(potcar_family)
    if potential_mapping:
        inputs.potential_mapping = DataFactory("dict")(dict=potential_mapping)
    else:
        inputs.potential_mapping = DataFactory("dict")(
            dict=get_potcar_mapping(structure=structure)
        )

    # Set options
    inputs.options = DataFactory("dict")(dict=options)

    # Set settings
    default_settings = {
        "parser_settings": {
            "misc": [
                "total_energies",
                "maximum_force",
                "maximum_stress",
                "run_status",
                "run_stats",
                "notifications",
            ],
            "add_structure": True,  # retrieve structure
            # 'add_kpoints': True,
            # 'add_forces' : True,
            "add_energies": True,
            # 'add_bands': True,
            # 'add_trajectory': True,
        },
    }
    if settings:
        default_settings.update(settings)
    inputs.settings = DataFactory("dict")(dict=default_settings)

    # Metadata
    if not label:
        formula = structure.formula.replace(" ", "")
        label = f"scf_{formula}"
    inputs.metadata = {"label": label}

    # Set workchain related inputs, in this case, give more explicit output to report
    inputs.verbose = DataFactory("bool")(True)

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
            f"Submitted singleshot workchain with pk: {workchain.pk} and label {workchain.label}, "
            f"stored in group with label {group_label}"
        )
    else:
        print(f"Submitted singleshot workchain with pk: {workchain.pk}")

    return workchain


def submit_elf(
    pk: int,
    potential_mapping: Optional[dict] = None,
    remote_computer: Optional[str] = None,
    options: Optional[dict] = None,
):
    node = load_node(pk)
    label = node.label
    remote_folder = node.outputs.remote_folder

    incar_dict = dict(node.inputs.parameters)
    incar_dict = {key.upper(): value for key, value in incar_dict.items()}
    if not incar_dict.get("LWAVE") or not incar_dict.get("LCHARG"):
        print(
            "To parse the ELF, you need the converged wavefunction and charge density."
        )
        return
    incar_dict.update(
        {
            "LELF": True,
            "NPAR": 1,  # NCORE parallelisation not possible with LELF processing
            "KPAR": 1,
            "ALGO": None,  # Don't change orbitals, just use WAVECAR
            "LWAVE": False,
            "LCHARG": False,
        }
    )

    # We set the workchain you would like to call
    workchain = WorkflowFactory("vasp.vasp")
    # Then, we set the workchain you would like to call
    # workchain = WorkflowFactory('vasp.verify')
    # workchain = CalculationFactory('vasp.vasp')

    # And finally, we declare the options, settings and input containers
    settings = AttributeDict()
    inputs = AttributeDict()

    # Organize settings
    settings.parser_settings = {}

    # Set inputs for the following WorkChain execution
    # Set parameters
    inputs.parameters = DataFactory("dict")(dict={"incar": incar_dict})
    # Set code
    inputs.code = node.inputs.code
    # Set structure
    inputs.structure = node.inputs.structure
    # Set k-points grid density
    inputs.kpoints = node.inputs.kpoints
    # Set options
    if not options:
        inputs.options = DataFactory("dict")(dict=node.get_options())
    else:
        inputs.options = DataFactory("dict")(dict=options)
    # Set potentials and their mapping. If user does not specify them, we use the VASP recommended ones
    structure = node.inputs.structure.get_pymatgen()
    inputs.potential_family = DataFactory("str")(potcar_family)
    if potential_mapping:
        inputs.potential_mapping = DataFactory("dict")(dict=potential_mapping)
    else:
        inputs.potential_mapping = DataFactory("dict")(
            dict=get_potcar_mapping(structure=structure)
        )
    # Set settings
    inputs.settings = node.inputs.settings
    # Metadata
    inputs.metadata = {"label": f"{label}_ELF"}
    # Set workchain related inputs, in this case, give more explicit output to report
    inputs.verbose = DataFactory("bool")(True)

    # Chgcar and wavecar
    remote_computer = node.inputs.code.computer.label
    if "gpu" in remote_computer:
        remote_computer = remote_computer.split("_")[0]
    inputs.chgcar = parsing.get_charge_density_data_from_pk(
        pk, remote_computer=remote_computer
    )
    inputs.wavecar = parsing.get_wavefunction_data_from_pk(
        pk, remote_computer=remote_computer
    )
    inputs.restart_folder = remote_folder

    # Submit the requested workchain with the supplied inputs
    workchain = submit(workchain, **inputs)
    print(f"Submitted singleshot workchain with pk: {workchain.pk}")
