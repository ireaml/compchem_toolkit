"""
Generate partial charge density for a given band/kpoint from VASP WAVECAR file.
"""

# Imports
from typing import Optional

import click

# pymatgen
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.outputs import Poscar, Structure, Wavecar

## CLI Commands:
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(
    name="make-parch",
    no_args_is_help=True,
    context_settings=CONTEXT_SETTINGS,
)
@click.option(
    "--wavecar",
    "-w",
    help="Path to WAVECAR file. Defaults to './WAVECAR'",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    default="./WAVECAR",
    show_default=True,
)
@click.option(
    "--contcar",
    "-c",
    help="Path to POSCAR/CONTCAR file. Defaults to './CONTCAR'",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True),
    default="./CONTCAR",
    show_default=True,
)
@click.option(
    "--band_index",
    "-b",
    help="Index of band to analyse.",
    type=int,
    show_default=True,
)
@click.option(
    "--filename",
    "-f",
    help="Filename for the PARCHG file that will be generated."
    "Default to PARCHG_band_{band_index}",
    type=str,
    show_default=True,
)
@click.option(
    "--kpoint",
    "-k",
    help="Index of the kpoint to generate the partial charge density for.",
    type=int,
    show_default=True,
)
@click.option(
    "--spin",
    "-s",
    help="Spin up (0) or down (1).",
    type=int,
    default=0,
    show_default=True,
)
@click.option(
    "--phase",
    "-p",
    help="Show phase changes in density, if present.",
    type=bool,
    default=False,
    show_default=True,
)
@click.option(
    "--vasp_type",
    "-v",
    help="Type of vasp executable ('gam', 'std' or 'ncl')." "Defaults to 'std'.",
    type=str,
    default="std",
    show_default=True,
)
def make_parchg(
    wavecar: str,
    contcar: str,
    band_index: int,
    filename: Optional[str] = None,
    kpoint: int = 0,
    spin: int = 0,
    phase: bool = False,
    vasp_type: str = "std",  # or ncl or gam
) -> None:
    """
    Generate partial charge density file for a given band and
    kpoint from VASP WAVECAR file.
    """
    if not filename:
        filename = f"./PARCHG_band_{band_index}"
    my_wavecar = Wavecar(filename=wavecar, vasp_type=vasp_type)  # or std or gam
    my_poscar = Poscar(Structure.from_file(contcar))
    print(
        "Generating PARCHG. All indexing starts at 1 (VASP), so substract 1 to change to Pythonic indexing!"
    )
    my_parchg = my_wavecar.get_parchg(
        my_poscar,
        kpoint=kpoint,  # first kpoint (Gamma in this case)
        band=band_index - 1,  # convert from python to vasp indexing
        spin=spin,  # 0 = up, 1 = down
        phase=phase,  # show phase changes in density if present
    )
    my_parchg.write_file(f"{filename}.vasp")


if __name__ == "__main__":
    make_parchg()
