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
# @click.option(
#     "--atoms",
#     "-a",
#     help="List of atom indexes for which to determine their energy states.",
#     type=click.Tuple([int, int, int]),
# )
@click.argument(
    "atoms",
    nargs=-1,
    # help="List of atom indexes for which to determine their energy states.",
)
@click.option(
    "--vasprun",
    "-v",
    help="Path to vasprun.xml file.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default="./vasprun.xml",
    show_default=True,
)
@click.option(
    "--kpoint",
    "-k",
    help="Kpoint to analyse. If not specified, all kpoints are considered.",
    type=int,
    default=None,
    show_default=True,
)
@click.option(
    "--orbital_decomposed",
    "-o",
    help="Whether to print orbital decomposed projections.",
    type=bool,
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option(
    "--threshold",
    "-t",
    help="Threshold for the site projection of the orbital. Only orbitals with "
         "higher projections on all sites in `atoms` will be returned.",
    type=float,
    default=0.06,
    show_default=True,
)
def analyse_procar(
    atoms: list,
    vasprun: str,
    kpoint: int = None,
    orbital_decomposed: bool = False,
    threshold: float = 0.06,
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
    def get_atom_projs(bsvasprun, eigenval, band, kp, selected_bands):
        projections[band][kp] = {}
        atom_projections = {}
        for atom_index in atoms: # Filter projections of selected sites
            band_proj = bsvasprun.projected_eigenvalues[Spin(1)][kp][band]
            atom_projections[atom_index] = round(
                sum(band_proj[int(atom_index)]),
                3,
            )

        if all(
            atom_projection > threshold  # projection higher than threshold for all sites
            for atom_projection in atom_projections.values()
        ):
            selected_bands[band][kp] = {
                index: round(sum(value), 3)  # Atom_index: projection
                for index, value in enumerate(band_proj)
                if sum(value) > threshold
            }  # add sites with significant contributions
            click.secho(f"Band: {band}. Kpoint: {kp}. Energy: {str(eigenval[kp][band][0])}", fg='green', bold=True)
            if orbital_decomposed:
                # Print orbital projections for selected atoms
                print(
                    f"Orbital decomposed:",
                    {
                        f"Atom index {atom_index}": {
                            f"Orb. index {ind}": round(val, 2)
                            for ind, val in enumerate(band_proj[int(atom_index)])
                            if val > 0.05
                        } for atom_index in atoms
                    }
                )
            click.echo("-------")

    if os.path.exists(vasprun):
        bsvasprun = BSVasprun(vasprun, parse_projected_eigen=True,)
    else:
        raise FileNotFoundError

    eigenval = bsvasprun.eigenvalues[Spin.up]
    number_of_bands = len(bsvasprun.projected_eigenvalues[Spin(1)][0])
    number_of_kpoints = len(bsvasprun.projected_eigenvalues[Spin(1)])
    projections = {}
    selected_bands = {}

    print("Analysing Vasprun. All indexing starts at 0 (pythonic), so add 1 to change to VASP indexing!")
    print(
        f"VBM: {bsvasprun.eigenvalue_band_properties[2]}; "
        f"CBM: {bsvasprun.eigenvalue_band_properties[1]}; "
        f"Band gap: {round(bsvasprun.eigenvalue_band_properties[0], 2)}"
    )
    if kpoint != None:
        print(f"Will only consider kpoint {kpoint}.")
    for band in range(0, number_of_bands):
        projections[band] = {} # store projections on selected sites
        selected_bands[band] = {} # bands with significant projection on sites, for output
        if not isinstance(kpoint, int):
            for kpoint in range(0, number_of_kpoints):
                get_atom_projs(bsvasprun, eigenval, band, kpoint, selected_bands)
        else:  # Only focussing in the specified kpoint
            get_atom_projs(bsvasprun, eigenval, band, kpoint, selected_bands)

if __name__ == "__main__":
    analyse_procar()