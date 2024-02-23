"""
Collection of functions to analyse VASP output in search of polarons.
Designed to analyse the output of defect calculations, where hole/electron localisation
is expected.
"""

import warnings

import pandas as pd
from pymatgen.electronic_structure.core import Spin
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun

warnings.filterwarnings("ignore", message="No POTCAR file with matching TITEL fields")


def hole_finder(vasprun: str) -> pd.DataFrame:
    """
    Identify bands where holes are localized, from VASP output.
    Useful to check output of defect calculation where localised holes are expected.
    Adapted from Scanlon Materials Theory Group wiki.
    """
    warnings.warn(
        """Python is 0-indexed, VASP in 1-indexed. All k-points and
    bands shown as an output from this script are 1-indexed, i.e. VASP-friendly."""
    )

    warnings.warn(
        """This is only designed to work for cases where you have a single hole...
    For multiple holes, this needs expanding!"""
    )

    if isinstance(vasprun, str):
        defect_data = BSVasprun(filename=vasprun)
    elif isinstance(vasprun, BSVasprun) or isinstance(vasprun, Vasprun):
        defect_data = vasprun

    print("Assigning eigenvalue spin channels...")

    eigen_up = defect_data.eigenvalues[Spin.up]
    eigen_down = defect_data.eigenvalues[Spin.down]

    for k in range(len(eigen_up)):  # loop over kpoint
        for b in range(len(eigen_up[k])):  # loop over band
            # Locate VBM
            if (
                eigen_up[k][b][0] == defect_data.eigenvalue_band_properties[2]
                or eigen_down[k][b][0] == defect_data.eigenvalue_band_properties[2]
            ):
                print(f"VBM band number: {b+1}, k-point: {k+1}")
                vbm_data = [b, k]

    print("\nLoading eigenvalue table...\n")

    eigenvals = []

    for b in range(vbm_data[0] - 6, vbm_data[0] + 6):  # check bands around VBM
        for k in range(len(eigen_up)):  # loop over kpoint
            eigenvals.append(
                [
                    b + 1,
                    k + 1,
                    eigen_up[k][b][0],
                    eigen_down[k][b][0],
                    eigen_up[k][b][1],
                    eigen_down[k][b][1],
                ]
            )
    df = pd.DataFrame(
        eigenvals,
        columns=[
            "band",
            "k-point",
            "up-eigenval",
            "down-eigenval",
            "up-occupancy",
            "down-occupancy",
        ],
    )

    # print(df)

    print(
        "======================================================================================\n"
    )

    print("Locating hole...\n")

    hole_spin_down = df[(df["up-occupancy"] == 1.0) & (df["down-occupancy"] == 0.0)]
    hole_spin_up = df[(df["up-occupancy"] == 0.0) & (df["down-occupancy"] == 1.0)]

    # Check if more than one hole
    num_kpts = len(eigen_up)

    if hole_spin_up.empty == True and hole_spin_down.empty == True:  # no holes found
        print("No holes found!")
    elif (
        hole_spin_up.empty == True and hole_spin_down.empty == False
    ):  # holes only on the spin up channel
        print(
            f"Found {hole_spin_down/num_kpts} hole in the spin down channel:\n",
            hole_spin_down,
        )
        print(
            "Recommended min and max for EINT in PARCHG calculation:",
            round(hole_spin_down["down-eigenval"].min(), 2),
            round(hole_spin_down["down-eigenval"].max(), 2),
        )
    elif (
        hole_spin_down.empty == True and hole_spin_up.empty == False
    ):  # holes only on the spin down channel
        print(
            f"Found {hole_spin_up/num_kpts} hole in the spin up channel:\n",
            hole_spin_up,
        )
        print(
            "Recommended min and max for EINT in PARCHG calculation:",
            round(hole_spin_up["up-eigenval"].min(), 2),
            round(hole_spin_up["up-eigenval"].max(), 2),
        )

    elif (
        hole_spin_up.empty == False and hole_spin_down.empty == False
    ):  # holes on both spin channels
        print(
            f"Found {hole_spin_up/num_kpts} holes in the spin up channel and {hole_spin_down/num_kpts} \
            in the spin down channel:\n",
            hole_spin_up,
            "\n",
            hole_spin_down,
        )
        print(
            "Holes in spin up: recommended min and max for EINT in PARCHG calculation:",
            round(hole_spin_up["up-eigenval"].min(), 2),
            round(hole_spin_up["up-eigenval"].max(), 2),
        )
        print(
            "Holes in spin down: Recommended min and max for EINT in PARCHG calculation:",
            round(hole_spin_down["down-eigenval"].min(), 2),
            round(hole_spin_down["down-eigenval"].max(), 2),
        )

    return df
