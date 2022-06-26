from typing import Optional

from aiida.engine import submit
from aiida.orm import load_code, Dict, SinglefileData, WorkChainNode
from aiida.plugins import DataFactory, WorkflowFactory

import pymatgen

def submit_cp2k_workchain(
    structure: pymatgen.core.structure.Structure,
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