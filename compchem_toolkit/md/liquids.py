import numpy as np
from pymatgen.core.structure import Structure, Molecule
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.periodic_table import Element
import matplotlib.pyplot as plt


def get_min_bond_distance(s: Structure, mode="fast", cutoff=3.2, verbose=True):
    """Calculate minimum and average distance in structure

    Args:
        s (Structure): pymatgen Structure object
        verbose (bool, optional): Print min and mean distances. Defaults to True.

    Returns: min distance, mean distance
    """
    distances = []
    pairs = [] # to avoid double counting
    coord_nums = []
    if mode == "accurate":
        nn = CrystalNN()
        for i in range(len(s)):
            nns = nn.get_nn_info(s, i)
            for n in nns:
                if (i, n) not in pairs and (n, i) not in pairs:
                    distances.append(s.get_distance(i, n["site_index"]))
                    pairs.append((i, n["site_index"]))
    elif mode == "fast":
        for i, s1 in enumerate(s):
            # Get sites within 4 A
            nn = s.get_neighbors(site=s1, r=cutoff)
            coord_nums.append(len(nn))
            # Get distances
            for n in nn:
                if (n.index, i) not in pairs: # avoid duplicates
                    distances.append(s.get_distance(i, n.index))
                    pairs.append((i, n.index))
    if verbose:
        print(f"Min distance: {min(distances):.2f} A, Mean distance: {np.mean(distances):.2f} A. Mean coord number: {np.mean(coord_nums)}")
    return min(distances), np.mean(distances), np.mean(coord_nums)


def get_density_atoms_per_A3(structure, verbose=True):
    """Calculate density in atoms/A^3"""
    Na = 6.022 * 10**23
    if isinstance(structure, Molecule):
        molecular_mass = structure.composition.weight
    else: # mass of atom
        molecular_mass = 0
        for el, n_el in structure.composition.reduced_composition.as_dict().items():
            molecular_mass += Element(el).atomic_mass * n_el
    print(f"Molecular mass: {molecular_mass}")
    density = structure.density # g/cm^3
    # Tranform from g/cm^3 to atoms/A^3
    density_atoms_A3 = density  * (Na / molecular_mass) * (1 / 10**8)**3
    if verbose:
        print(f"Density in atoms/A^3: {density_atoms_A3:.2e}")
    return density_atoms_A3


def get_coordination_number(s, verbose=True):
    nn = CrystalNN()
    coordination_numbers = []
    for i in range(len(s)):
        nns = nn.get_nn_info(s, i)
        coordination_numbers.append(len(nns))
    mean_coord_num = np.mean(coordination_numbers)
    if verbose:
        print(f"Mean coordination number: {mean_coord_num:.2f}. Max: {max(coordination_numbers)}, Min: {min(coordination_numbers)}")
    return coordination_numbers


def get_rdf_plot(
    rdf,
    label: str | None = None,
    xlim: tuple = (0.0, 8.0),
    ylim: tuple = (-0.005, 3.0),
    loc_peak: bool = False,
):
    """
    Plot the average RDF function.

    Args:
        label (str): The legend label.
        xlim (list): Set the x limits of the current axes.
        ylim (list): Set the y limits of the current axes.
        loc_peak (bool): Label peaks if True.
    """

    if label is None:
        symbol_list = [e.symbol for e in rdf.structures[0].composition]
        symbol_list = [symbol for symbol in symbol_list if symbol in rdf.species]

        label = symbol_list[0] if len(symbol_list) == 1 else "-".join(symbol_list)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rdf.interval, rdf.rdf, label=label, linewidth=2.0, zorder=1)

    if loc_peak:
        ax.scatter(
            rdf.peak_r,
            rdf.peak_rdf,
            marker="P",
            #s=24,
            c="k",
            linewidths=0.1,
            alpha=0.7,
            zorder=2,
            label="Peaks",
        )

    ax.set_xlabel("$r$ ($\\rm\\AA$)")
    ax.set_ylabel("$g(r)$")
    ax.legend(loc="upper right")
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    # ax.tight_layout()
    return fig, ax