"""Collection of useful functions for lobster input and output."""

from typing import Dict, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pymatgen
import seaborn as sns
from palettable.cartocolors.qualitative import Antique_10, Safe_10
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.electronic_structure.cohp import CompleteCohp, IcohpCollection
from pymatgen.electronic_structure.core import Orbital, Spin
from pymatgen.electronic_structure.plotter import CohpPlotter
from scipy import interpolate

from vasp.output import _install_custom_font
from vasp.potcar import (
    get_potcar_from_structure,
    get_valence_orbitals_from_potcar,
)

# For plotting
plt.style.use("/home/ireaml/Python_Modules/mpl_style/publication_style.mplstyle")
mpl.rc("lines", **{"linewidth": 2})
mpl.rc("text", **{"usetex": False})
mpl.rcParams["axes.unicode_minus"] = False


# Input setup


def get_number_of_bands(
    composition: pymatgen.core.composition.Composition,
    orbitals: dict,
) -> int:
    """
    Calculated the number of bands required by lobster to perform COHP analysis.

    Args:
        composition (pymatgen.core.composition.Composition): composition of your structure
        orbitals (dict): dict matching element (as str) to a list of the valence orbitals to consider (as strings too), \
            e.g. {'H': ['s'], }.
    Returns:
        (int): number of bands required for lobster post-processing
    """
    degeneracy = {"s": 1, "p": 3, "d": 5, "f": 7}
    bands = 0
    assert (
        type(composition) == pymatgen.core.composition.Composition
    ), f"Composition must be a pymatgen.core.composition.Composition.\
         You'be given me a {type(composition)}"
    for element, number_of_atoms in composition.as_dict().items():
        for orbital_type in orbitals[element]:
            bands += number_of_atoms * degeneracy[orbital_type]

    return int(bands)


def get_number_of_bands_from_struct(
    structure: Structure,
    orbitals: Dict = None,
) -> int:
    """
    Calculates the number of electronic bands required in the SCF calc for a
    latter lobster calculation.

    Args:
        structure (Structure):
            pymatgen.core.structure.Structure object
        orbitals (dict, optional):
            Dict matching element to the pseudopotential valence orbitals
            (i.e. {"Cd": {"s", "p"}})
            Defaults to None.

    Returns:
        int: _description_
    """
    if not orbitals:
        orbitals = get_valence_orbitals_from_potcar(
            potcar=get_potcar_from_structure(structure=structure)
        )
    return get_number_of_bands(
        structure.composition,
        orbitals,
    )


# Output analysis


def get_labels_by_elements(
    complete_cohp: CompleteCohp or str,
    element_1: str,
    element_2: str,
    structure_path: Optional[str] = None,
):
    """
    Selects the labels in ICOHPlist.lobster that correspond to the bonds between element_1 and element_2

    Args:
        complete_cohp (CompleteCohp or str) : pymatgenelectronic_structure.cohp.CompleteCohp object or its file path (as str)
        element_1 (str): Symbol for first element
        element_2 (str): Symbol for second element
        structure_path: path to structure file
    Returns:
        [list of ints]: List of labels for the selected bonds
    """
    if isinstance(complete_cohp, str):
        complete_cohp = CompleteCohp.from_file(
            filename=complete_cohp, structure=structure_path, fmt="lobster"
        )
    selected_labels = []
    all_labels = list(complete_cohp.bonds.keys())
    for label in all_labels:
        if all(
            element in [element_1, element_2]
            for element in [
                complete_cohp.bonds[str(label)]["sites"][0].species_string,
                complete_cohp.bonds[str(label)]["sites"][1].species_string,
            ]
        ):
            selected_labels.append(label)
    return selected_labels


def calculate_mean_icohp(
    icohpcollection,
    structure: pymatgen.core.structure.Structure,
    element_1: str,
    element_2: str,
    maxbondlength=3.5,
    are_cobis=False,
    verbose=False,
) -> dict:
    """
    Calculate mean ICOHP between two elements in your structure.

    Args:
        icohpcollection ([type]):
        structure (pymatgen.core.structure.Structure):
            pymatgen Structure object
        element_1 (str):
            symbol of element 1
        element_2 (str):
            symbol of element 2
        maxbondlength (float):
            maximum bond length to consider bonds between element_1 and element_2.
            Default to 3.5 A.
        are_cobis (bool, optional):
            Defaults to False.
        verbose (bool):
            Whether to print all ICOHP values for the bonds between the selected elements
            Defaults to False.
    Returns:
        dict: Dictionary specifying the elements and the mean ICOHP
    """
    # get sites of element 1 to then grab their ICOHP values
    indexes_element_1 = [
        index
        for index, site in enumerate(structure)
        if site.specie == Element(element_1)
    ]
    # print(indexes_element_1)

    # Grab the icohp between element_1 and element_2
    mean_icohp = []
    for index_site in indexes_element_1:
        # grab icohp by index
        icohp_element_1_element_2 = icohpcollection.get_icohp_dict_of_site(
            site=index_site,
            minbondlength=0.0,
            maxbondlength=maxbondlength,
            only_bonds_to=[element_2],
        )
        for key, icohp in icohp_element_1_element_2.items():
            # print(key+':'+ str(icohp.icohp))
            mean_icohp.append(icohp.icohp[Spin(1)])
    if are_cobis:
        print(f"ICOBI({element_1}-{element_2}): {np.mean(mean_icohp):.2f}")
    else:
        print(f"ICOHP({element_1}-{element_2}): {np.mean(mean_icohp):.2f} eV")
    if verbose:
        print(f"All ICOHP({element_1}-{element_2}): ", mean_icohp)
    return {f"{element_1}-{element_2}": round(np.mean(mean_icohp), 3)}


def plot_cohp_for_label_list(
    complete_cohp: CompleteCohp,
    selected_labels: list,
    xlim: list = [-40, 40],
    ylim: list = [-10, 5],
    fill: Optional[bool] = True,
    spin_legend: Optional[bool] = True,
    figure_path: str = None,
) -> tuple:
    """
    Convenient function to plot COHP for a list of (bond) labels.
    Adapted from original function by Seán R. Kavanagh.

    Args:
        complete_cohp (CompleteCohp):
            pymatgen.electronic_structure.cohp.CompleteCohp object
        selected_labels (list):
            List of labels of the bonds to include in the plot.
        xlim (list, optional):
            Limits for x axis (COHP/bond (eV)). Defaults to [-40, 40].
        ylim (list, optional):
            Limits for y axis (Energy - E_fermi (eV)). Defaults to [-10, 5].
        fill (bool, optional):
            whether to fill the COHP region with a color according to bonding/antibonding. Defaults to True.
        spin_legend (bool, optional):
            whether to add spin legend to the plot. Defaults to True.
        figure_path (str, optional):
            filename to save the figure as, including formar (e.g. './my_plot.svg'). Defaults to None.
    Returns:
        mpl.axes.Axes: matplotlib axes object
    """

    # Install custom font
    _install_custom_font()

    cp = CohpPlotter()

    # search for the number of the COHP you would like to plot in ICOHPLIST.lobster (the numbers in COHPCAR.lobster are different!)
    label_list = [str(i) for i in selected_labels]
    divisor = len(label_list)  # number of bonds
    cp.add_cohp(
        "Total COHP",
        complete_cohp.get_summed_cohp_by_label_list(
            label_list=label_list, divisor=divisor
        ),
    )

    # Get plot to modify it
    cp.get_plot(
        ylim=ylim, xlim=[xlim[0] / divisor, xlim[1] / divisor], integrated=False
    )
    # Get figure
    fig = plt.gcf()
    fig.set_size_inches(7, 6, forward=True)
    # Get axes
    ax = plt.gca()
    lines = ax.get_lines()
    # Remove default legend
    legend = ax.get_legend()
    legend.remove()

    # Check if spin polarised calc
    if len(lines) == 3:
        line = lines[0]
        line.set_c("tab:grey")  # non spin polarised
        line.set_linewidth(1.8)
    else:
        lines[0].set_c("tab:grey")
        lines[0].set_linestyle("-")  # 1 spin channel
        lines[1].set_c("tab:grey")
        lines[1].set_linestyle("--")  # 2 spin channel
        [line.set_linewidth(1.8) for line in lines[0:2]]
        if spin_legend:
            plt.legend(
                [lines[0], lines[1]],
                ["Spin up", "Spin down"],
                loc="upper left",
                fontsize=14,
            )

    # Annotate: 'Bonding', 'Antibonding' and 'COHP'
    label = label_list[0]
    element_1 = complete_cohp.bonds[label]["sites"][0].species_string
    element_2 = complete_cohp.bonds[label]["sites"][1].species_string
    fontsize_annot = 21
    plt.annotate(
        f"COHP({element_1}-{element_2})",
        ha="center",
        color="k",
        xy=(0.75, 0.9),
        xycoords="axes fraction",
        fontsize=fontsize_annot,
        bbox=dict(boxstyle="round", ec=(1.0, 0.5, 0.5), fc=(1.0, 0.8, 0.8)),
    )
    plt.annotate(
        r"Bonding",
        ha="right",
        color="tab:blue",
        xy=(0.95, 0.05),
        xycoords="axes fraction",
        fontsize=fontsize_annot,
        bbox=dict(boxstyle="round", fc="tab:blue", alpha=0.15),
    )
    plt.annotate(
        r"Anti-Bonding",
        ha="left",
        color="tab:orange",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
        fontsize=fontsize_annot,
        bbox=dict(boxstyle="round", fc="tab:orange", alpha=0.15),
    )

    if fill:
        for line in lines[0:-2]:  # for both spin channels
            data = line.get_data()
            try:
                tck = interpolate.splrep(data[1], data[0], s=0)
                ynew = np.arange(ylim[0], ylim[1], 0.001)
                xnew = interpolate.splev(ynew, tck, der=0)
                yrangeb = [ynew[i] for i in range(len(ynew)) if xnew[i] > 0]
                xrangeb = [xnew[i] for i in range(len(ynew)) if xnew[i] > 0]
                yrangeab = [ynew[i] for i in range(len(ynew)) if xnew[i] < 0]
                xrangeab = [xnew[i] for i in range(len(ynew)) if xnew[i] < 0]
                ax.fill_betweenx(
                    yrangeb, 0, xrangeb, color="tab:blue", alpha=0.3
                )  # alpha=0.5
                ax.fill_betweenx(
                    yrangeab, 0, xrangeab, color="tab:orange", alpha=0.3
                )  # alpha=0.5
            except TypeError:
                print(f"Problem interpolating line {line} - this will be skipped")
    # Reduce linewidth of line at Efermi and Line at x=0
    lines[-1].set_linewidth(1.2)
    lines[-2].set_linewidth(1.8)
    lines[-2].set_c("tab:grey")

    # Axis labels and tick formatting
    axis_label_size = 21
    plt.xlabel("-COHP per bond (eV)", fontsize=axis_label_size)
    plt.ylabel("$E$ - $E_f$ (eV)", fontsize=axis_label_size)
    plt.locator_params(axis="x", nbins=5)
    plt.locator_params(axis="y", nbins=7)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)

    if figure_path:
        plt.savefig(figure_path, bbox_inches="tight", dpi=600)
    return (fig, ax)


def plot_cohp_orb_res(
    complete_cohp: CompleteCohp,
    selected_label: int,
    threshold: float = 0.2,
    xlim: list = [-40, 40],
    ylim: list = [-10, 5],
    figure_path: str = None,
    palette: list = None,
) -> tuple:
    """
    Plot orbitally resolved COHP for the specified bond labels.
    Adapted from original function by Seán R. Kavanagh.

    Args:
        complete_cohp (CompleteCohp):
            pymatgen.electronic_structure.cohp.CompleteCohp object
        selected_labels (list):
            List of labels of the bonds to include in the plot.
        threshold (float):
            ICOHP threshold to consider a pair of orbitals. Only
            orbital interactions with an ICOHP higher (in absolute number)
            than the threshold are plotted.
        xlim (list, optional):
            Limits for x axis (COHP/bond (eV)). Defaults to [-40, 40].
        ylim (list, optional):
            Limits for y axis (Energy - E_fermi (eV)). Defaults to [-10, 5].
        fill (bool, optional):
            whether to fill the COHP region with a color according to bonding/antibonding.
            Defaults to True.
        spin_legend (bool, optional):
            whether to add spin legend to the plot. Defaults to True.
        figure_path (str, optional):
            filename to save the figure as, including formar (e.g. './my_plot.svg').
            Defaults to None.
        palette (list or seaborn.palettes._ColorPalette, optional):
            Matplotlib color palette.
    Returns:
        mpl.axes.Axes: matplotlib axes object
    """

    # Install custom font
    _install_custom_font()

    # Orbital dict - to generate pmg Orbital object from orbital string
    dict_orbital_to_index = {
        "s": 0,
        "py": 1,
        "pz": 2,
        "px": 3,
        "dx2": 8,
        "dxy": 4,
        "dxz": 7,
        "dyz": 5,
        "dz2": 6,
    }

    cp = CohpPlotter()

    # search for the number of the COHP you would like to plot
    # in ICOHPLIST.lobster (the numbers in COHPCAR.lobster are different!)
    label = str(selected_label)
    for orb_key in complete_cohp.orb_res_cohp[label].keys():
        cohp_of_orb = sum(complete_cohp.orb_res_cohp[label][orb_key]["COHP"][Spin(1)])
        if abs(cohp_of_orb) > threshold:
            print("Orbital", orb_key, "COHP:", round(cohp_of_orb, 2))
            orb_labels = orb_key.split("-")
            orbitals = [
                (
                    int(orb_labels[0][0]),
                    Orbital(dict_orbital_to_index[orb_labels[0][1:]]),
                ),
                (
                    int(orb_labels[1][0]),
                    Orbital(dict_orbital_to_index[orb_labels[1][1:]]),
                ),
            ]

            # Format labels
            sites = complete_cohp.bonds[label]["sites"]
            if len(orb_labels[0]) > 2 or len(orb_labels[1]) > 2:
                orb_1_fmt = r"$" + orb_labels[0][0:2] + "_{" + orb_labels[0][2:] + "}$"
                orb_2_fmt = r"$" + orb_labels[1][0:2] + "_{" + orb_labels[1][2:] + "}$"
            else:
                orb_1_fmt = orb_labels[0]
                orb_2_fmt = orb_labels[1]

            # Line label is Element(orbital)-Element(orbital) (e.g. Sn(5s)-Sn(5s))
            plot_label = (
                str(sites[0].species_string)
                + "("
                + orb_1_fmt
                + ")-"
                + str(sites[1].species_string)
                + "("
                + orb_2_fmt
                + ")"
            )
            cp.add_cohp(
                plot_label,
                complete_cohp.get_orbital_resolved_cohp(label=label, orbitals=orbitals),
            )

    # Get plot to modify it
    cp.get_plot(ylim=ylim, xlim=[xlim[0], xlim[1]], integrated=False)
    # Get figure
    fig = plt.gcf()
    fig.set_size_inches(7, 6, forward=True)
    # Get axes
    ax = plt.gca()
    lines = ax.get_lines()
    legend = ax.get_legend()

    # Modify colors
    if not palette:
        # palette = sns.color_palette("deep", len(legend.get_lines()))
        palette = Safe_10.mpl_colors
    for i in range(len(legend.get_lines())):  # orbital cohp lines
        lines[i].set_color(palette[i])
        lines[i].set_linewidth(1.8)
        legend.get_lines()[i].set_color(palette[i])
        plt.setp(legend.get_texts(), fontsize="16")
        plt.setp(legend, bbox_to_anchor=(0.5, 0.2, 0.5, 0.5))

    # Annotate: 'Bonding', 'Antibonding' and 'COHP'
    element_1 = complete_cohp.bonds[label]["sites"][0].species_string
    element_2 = complete_cohp.bonds[label]["sites"][1].species_string
    fontsize_annot = 21
    plt.annotate(
        f"COHP({element_1}-{element_2})",
        ha="center",
        color="k",
        xy=(0.75, 0.9),
        xycoords="axes fraction",
        fontsize=fontsize_annot,
        bbox=dict(boxstyle="round", fc="tab:grey", alpha=0.15),
    )
    plt.annotate(
        r"Bonding",
        ha="right",
        color="k",
        xy=(0.95, 0.05),
        xycoords="axes fraction",
        fontsize=fontsize_annot,
        bbox=dict(boxstyle="round", fc="tab:grey", alpha=0.15),
    )
    plt.annotate(
        r"Anti-Bonding",
        ha="left",
        color="k",
        xy=(0.05, 0.05),
        xycoords="axes fraction",
        fontsize=fontsize_annot,
        bbox=dict(boxstyle="round", fc="tab:grey", alpha=0.15),
    )

    # Reduce linewidth of line at Efermi and Line at x=0
    lines[-1].set_linewidth(1.2)
    lines[-2].set_linewidth(1.5)
    lines[-2].set_c("tab:grey")

    # Axis labels and tick formatting
    axis_label_size = 21
    plt.xlabel("-COHP per bond (eV)", fontsize=axis_label_size)
    plt.ylabel("$E$ - $E_f$ (eV)", fontsize=axis_label_size)
    plt.locator_params(axis="x", nbins=5)
    plt.locator_params(axis="y", nbins=7)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=16)

    if figure_path:
        plt.savefig(figure_path, bbox_inches="tight", dpi=600)
    return (fig, ax)
