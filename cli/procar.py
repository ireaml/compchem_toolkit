"""
Find bands localised on particular atoms.
Useful to generate charge density plots for particular bonds.
"""
import os
import click

from pymatgen.io.vasp.outputs import BSVasprun
from pymatgen.electronic_structure.core import Spin

## CLI Commands:
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])

@click.command(
    name="analyse-procar",
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
)
@click.option(
    "--atoms",
    "-a",
    help="List of atom indexes for which to determine their energy states.",
    type=list,
)
@click.option(
    "--vasprun",
    "-v",
    help="Path to vasprun.xml file. Defaults to './vasprun.xml'",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default="./vasprun.xml",
)
@click.option(
    "--threshold",
    "-t",
    help="Threshold for the site projection of the orbital. Only orbitals with "
         "higher projections on all sites in `atoms` will be returned."
         "Defaults to 0.06.",
    type=float,
    default=0.06,
)
def analyse_procar(
    atoms: list,
    vasprun: str,
    threshold: float = 0.06,
    verbose: bool= True,
) -> None:
    """
    Determines the energy states localized in the atoms speficied with
    the `--atoms` option.

    Args:
        vasprun (str):
            path to vasprun.xml file
        atoms (list):
            list of site indexes for which to determine their energy states
        threshold (float, optional):
            Threshold for the site projection of the orbital. Only orbitals with \
            higher projections on all sites in `atoms` will be returned.
            Defaults to 0.06.
        verbose (bool, optional):
            Whether to print the energy states.
            Defaults to True.
    Returns:
        DataFrame
    """
    if os.path.exists(vasprun):
        bsvasprun = BSVasprun(vasprun, parse_projected_eigen=True,)
    else:
        raise FileNotFoundError
    # bsvasprun.projected_eigenvalues[spin][kpoint][band][orbital]
    number_of_bands = len(bsvasprun.projected_eigenvalues[Spin(1)][0])
    number_of_kpoints = len(bsvasprun.projected_eigenvalues[Spin(1)])
    projections = {}
    selected_bands = {}
    print("Analysing PROCAR. All indexing starts at 0 (pythonic), so add 1 to change to VASP indexing!")
    for band in range(0, number_of_bands):
        projections[band] = {} # store projections on selected sites
        selected_bands[band] = {} # bands with significant projection on sites, for output
        for kpoint in range(0, number_of_kpoints):
            projections[band][kpoint] = {}
            atom_projections = {}
            for atom_index in atoms: # Filter projections of selected sites
                atom_projections[atom_index] = round(
                    sum(bsvasprun.projected_eigenvalues[Spin(1)][kpoint][band][atom_index]),
                    3,
                )

            if all(
                atom_projection > threshold # projection higher than threshold for all sites
                for atom_projection in atom_projections.values()
            ):
                selected_bands[band][kpoint] = {
                    index: round(sum(value), 3)
                    for index, value in enumerate(bsvasprun.projected_eigenvalues[Spin(1)][kpoint][band])
                    if sum(value) > threshold
                } # add sites with significant contributions
                if verbose:
                    print(f"Band: {band}. Kpoint: {kpoint}")
                    print(selected_bands[band][kpoint])
                    print("-------")
