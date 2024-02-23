"""Function to analyse site magnetization"""
import numpy as np
import pandas as pd
import pymatgen


def site_magnetizations(
    outcar: pymatgen.io.vasp.outputs.Outcar,
    structure: pymatgen.core.structure.Structure,
    threshold: float = 0.1,
    verbose: bool = True,
) -> tuple:
    """
    Prints sites with magnetization above threshold.

    Args:
        outcar (pymatgen.io.vasp.outputs.Outcar): outcar object
        structure (pymatgen.core.structure.Structure): structure object
        threshold (float, optional): Magnetization threhold to print site. Defaults to 0.1 e-.

    Returns:
        tuple: _description_
    """
    # Site mags
    mag = outcar.magnetization
    significant_magnetizations = {}
    for index, element in enumerate(mag):
        mag_array = np.array(list(element.values()))
        total_mag = np.sum(mag_array[np.abs(mag_array) > 0.01])
        if np.abs(total_mag) > threshold:
            significant_magnetizations[
                f"{structure[index].species_string}({index})"
            ] = {
                "Site": f"{structure[index].species_string}({index})",
                "Coords": [round(coord, 3) for coord in structure[index].frac_coords],
                "Total mag": round(total_mag, 3),
            }
            significant_magnetizations[
                f"{structure[index].species_string}({index})"
            ].update({k: round(v, 4) for k, v in element.items()})
    df = pd.DataFrame.from_dict(significant_magnetizations, orient="index")
    if verbose:
        display(df)
    return df
