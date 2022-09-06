"""Useful functions to submit CP2K workchains with aiida."""

import os
from typing import Optional
from monty.serialization import dumpfn, loadfn
import warnings

# aiida
from aiida.engine import submit
from aiida.orm import RemoteData, StructureData, load_code, Dict, SinglefileData, WorkChainNode
from aiida.plugins import DataFactory, WorkflowFactory
from aiida.engine.processes.builder import ProcessBuilder
from aiida.tools.groups import GroupPath
path = GroupPath()

# pymatgen
import pymatgen
from pymatgen.core.structure import Structure

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_kind_section(
    structure: Structure,
) -> list:
    """
    Generate cp2k kind section (PBE basis) for the elements in the
    input structure.
    Args:
        structure (pymatgen.core.structure.Structure)
    Returns:
        list with the kind section for every element present in the input structure
    """
    default_kind_section = loadfn(os.path.join(MODULE_DIR, "yaml_files/cp2k/kind_pbe_file.yaml"))
    try:
        kind_section = [default_kind_section[str(element)] for element in structure.composition.elements]
    except KeyError:
        warnings.warn("The file with the default basis/potentials lacks some of the elements present in your structure!")
        return None
    return kind_section


def submit_cp2k_workchain(
    structure: Structure,
    input_parameters: dict,
    options: dict,
    kind_section: Optional[dict]=None,
    code_string: str="cp2k-9.1@daint_gpu",
    label: Optional[str]=None,
    remote_data: Optional[RemoteData]=None,
    submit_workchain: Optional[bool]=True,
    group_label: Optional[str]=None,
)-> WorkChainNode or ProcessBuilder:
    """
    Submit Cp2k BaseWorkChain.

    Returns:
        WorkChainNode or ProcessBuilder (if submit_workchain is set to False)
    """
    # Construct process builder.cp2k.
    Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")
    builder = Cp2kBaseWorkChain.get_builder()

    # Code
    code = load_code(code_string)
    builder.cp2k.code = code

    # Structure
    builder.cp2k.structure = StructureData(pymatgen=structure)

    # Parent folder
    if remote_data:
        # Update restart info in parameters
        remote_folder = remote_data.get_remote_path()
        restart_wfn_fn = f"{remote_folder}/aiida-RESTART.wfn"
        input_parameters["FORCE_EVAL"]["DFT"]["RESTART_FILE_NAME"] = restart_wfn_fn
        input_parameters["FORCE_EVAL"]["DFT"]["SCF"]["SCF_GUESS"] = "RESTART"
        input_parameters["EXT_RESTART"] = {"RESTART_FILE_NAME": f"{remote_folder}/aiida-1.restart"}
        # Restart folder
        builder.handler_overrides = Dict(dict={"restart_incomplete_calculation": True})
        builder.cp2k.parent_calc_folder = remote_data

    # Parameters
    if isinstance(input_parameters, dict):
        input_parameters = Dict(dict=input_parameters)
    if isinstance(input_parameters, Dict):
        builder.cp2k.parameters = input_parameters
    else:
        raise TypeError(
            f"Incorrect data type for `input_parameters`. "
            f"You gave me a {type(input_parameters)} but I expect a `dict` or `Dict` objects."
        )
    if not "SUBSYS" in input_parameters["FORCE_EVAL"].keys() and kind_section:
        input_parameters["FORCE_EVAL"]["SUBSYS"] = {"KIND": kind_section}
    elif not "SUBSYS" in input_parameters["FORCE_EVAL"].keys() and not kind_section:
        try:
            input_parameters["FORCE_EVAL"]["SUBSYS"] = {"KIND": generate_kind_section(structure) }
        except:
            raise KeyError("Problem when automatically generating the KIND section of the CP2K input file. "
                           "Please provide this either in the input_parameters or specify it with "
                           "the `kind_section` argument")

    # Setup pseudopotentials & basis set files
    # Basis set
    basis_file = SinglefileData(file="/home/ireaml/cp2k_files/BASIS_MOLOPT")
    # Pseudopotentials
    pseudo_file = SinglefileData(file="/home/ireaml/cp2k_files/GTH_POTENTIALS")
    builder.cp2k.file = {
        'basis': basis_file,
        'pseudo': pseudo_file,
    }

    # Options
    builder.cp2k.metadata.options = options

    # Metadata
    if not label:
        formula = structure.composition.to_pretty_string()
        label = f'cp2k_{formula}'
    builder.cp2k.metadata.label = label

    # Submit
    if submit_workchain:
        type = input_parameters["GLOBAL"]["RUN_TYPE"]
        workchain = submit(builder)
        if group_label:
            group = path[group_label].get_or_create_group()
            group = path[group_label].get_group()
            group.add_nodes(workchain)
            print(f"Submitted CP2K workchain of type {type} with pk {workchain.pk}, label {label} and group {group_label}.")
        else:
            print(f"Submitted CP2K workchain of type {type} with pk {workchain.pk}, label {label}.")
        return workchain
    else:
        return builder