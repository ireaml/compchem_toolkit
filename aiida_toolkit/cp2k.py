from typing import Optional

from aiida.engine import submit
from aiida.orm import load_code, Dict, SinglefileData, WorkChainNode
from aiida.plugins import DataFactory, WorkflowFactory

import pymatgen

def submit_cp2k_workchain(
    structure: pymatgen.core.structure.Structure,
    input_parameters: dict,
    options: dict,
    code_string: str="cp2k-9.1@daint_gpu",
    label: Optional[str]=None,
)-> WorkChainNode:
    """
    Submit Cp2k BaseWorkChain.

    Returns:
        WorkChainNode: 
    """
    # Construct process builder.
    Cp2kBaseWorkChain = WorkflowFactory("cp2k.base")
    builder = Cp2kBaseWorkChain.get_builder()
    
    # Code
    code = load_code(code_string)
    builder.code = code
    
    # Parameters
    if isinstance(input_parameters, dict):
        input_parameters = Dict(dict=input_parameters)
    if isinstance(input_parameters, Dict):
        builder.parameters = input_parameters
    else:
        raise TypeError(f"Incorrect data type for `parameters`. You gave me a {type(input_parameters)} but I expect a `dict` or `Dict` objects.")
    
    # Setup pseudopotentials & basis set files
    # Basis set.
    basis_file = SinglefileData(file="/home/ireaml/cp2k_files/BASIS_MOLOPT")
    # Pseudopotentials.
    pseudo_file = SinglefileData(file="/home/ireaml/cp2k_files/GTH_POTENTIALS")
    builder.file = {
        'basis': basis_file,
        'pseudo': pseudo_file,
    }

    # Options
    builder.metadata.options = options

    # Metadata 
    if not label:
        formula = structure.composition.to_pretty_string()
        label = f'cp2k_{formula}'
    builder.metadata.label = label
    
    # Submit
    workchain = submit(builder)
    print(f"Submitted relax workchain with pk: {workchain.pk} and label {workchain.label}")
    
    return workchain