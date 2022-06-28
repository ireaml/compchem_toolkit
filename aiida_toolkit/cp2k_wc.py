import os
from typing import Optional
from monty.serialization import dumpfn, loadfn
import warnings

# aiida
from aiida.engine import submit
from aiida.orm import load_code, Dict, SinglefileData, WorkChainNode
from aiida.plugins import DataFactory, WorkflowFactory

# pymatgen
import pymatgen
from pymatgen.core.structure import Structure

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
default_kind_section = loadfn(os.path.join(MODULE_DIR, "yaml_files/cp2k/kind_pbe_file.yaml"))

def generate_kind_section(
    structure: Structure,
) -> list:
    """Generate cp2k kind section (PBE basis) for the elements in the 
    input structure.
    Args:
        structure (pymatgen.core.structure.Structure)
    Returns:
        list with the kind section for every element present in the input structure   
    """
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
)-> WorkChainNode:
    """
    Submit Cp2k BaseWorkChain.

    Returns:
        WorkChainNode: 
    """
    # Construct process builder.cp2k.
    Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")
    builder = Cp2kBaseWorkChain.get_builder()
    
    # Code
    code = load_code(code_string)
    builder.cp2k.code = code
    
    # Parameters
    if isinstance(input_parameters, dict):
        input_parameters = Dict(dict=input_parameters)
    if isinstance(input_parameters, Dict):
        builder.cp2k.parameters = input_parameters
    else:
        raise TypeError(f"Incorrect data type for `parameters`. You gave me a {type(input_parameters)} but I expect a `dict` or `Dict` objects.")
    if not "SUBSYS" in input_parameters.keys() and kind_section:
        input_parameters["SUBSYS"] = {"KIND": kind_section}
    elif not "SUBSYS" in input_parameters.keys() and not kind_section:
        try:
            input_parameters["SUBSYS"] = {"KIND": generate_kind_section(structure) }
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
    workchain = submit(builder)
    print(f"Submitted relax workchain with pk: {workchain.pk} and label {workchain.label}")
    
    return workchain