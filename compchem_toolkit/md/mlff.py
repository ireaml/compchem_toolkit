import pymlff
from pymatgen.core.structure import Structure
import numpy as np

import itertools
from collections import OrderedDict

from dscribe.descriptors import SOAP

from maml.sampling.direct import DIRECTSampler, BirchClustering, SelectKFromClusters
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from pymlff.io import ml_ab_from_trajectory

def get_structure_from_config(config: pymlff.core.Configuration):
    # Get list of species in configuration
    species = []
    for element in config.atom_types_numbers:
        # Add the element as many times as its dict value
        species += [element] * config.atom_types_numbers[element]
    s = Structure(
        lattice=config.lattice,
        coords=config.coords,
        species=species,
        coords_are_cartesian=True, # Cartesian coords!
    )
    return s


def parse_free_energy_from_outcar(path_to_outcar: str="OUTCAR"):
    """Parse TOTEN energies from OUTCAR file"""
    str_finished_ionic_step = "aborting loop because EDIFF is reached"
    str_free = "FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)"
    str_toten = "free  energy   TOTEN  =" #str_toten = "free energy    TOTEN  ="

    # Read the OUTCAR file
    with open(path_to_outcar, "r") as f:
        lines = f.readlines()

    # Get first line containing the string str_token after a line containing str_finished_ionic_step
    totens = []
    for i, line in enumerate(lines):
        if str_finished_ionic_step in line:
            found_toten = False
            for j in range(i, len(lines)):
                if str_toten in lines[j]: # str_free in lines[j] and
                    toten = lines[j].split(str_toten)[1].split("eV")[0]
                    #print(i, j, lines[j])
                    totens.append(float(toten))
                    found_toten = True
                    break
            if not found_toten:
                print(f"TOTEN not found after line {i}!") #break
            #break
    # Save totens to file
    arr = np.array(totens)
    np.save("totens.npy", arr)


def get_mlab_from_traj(traj, n_clusters, threshold_init=0.15, plot=True, index_from_1=True):
    def get_soap_descriptors(traj, symbols_unique, r_cut = 6.0, n_max = 8, l_max = 6):
        soap = SOAP(
            species=symbols_unique,
            periodic=True,
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
        )
        soap_descriptors = []
        for a in traj:
            soap_descriptors.append(soap.create(a))
        # Separate by element
        descriptors_by_element = {s: [] for s in symbols_unique}
        for symbol, i in indices.items():
            descriptors_by_element[symbol] = [d[i[0]:i[1]] for d in soap_descriptors]
        return descriptors_by_element

    def plot_PCAfeature_coverage(all_features, selected_indexes):
        fig, ax = plt.subplots(figsize=(5, 5))
        selected_features = all_features[selected_indexes]
        plt.plot(all_features[:, 0], all_features[:, 1], "*", alpha=0.5, label=f"All {len(all_features):,} envs")
        plt.plot(
            selected_features[:, 0],
            selected_features[:, 1],
            "*",
            alpha=0.5,
            label=f"Sampled {len(selected_features):,}",
        )
        legend = plt.legend(frameon=False, fontsize=14, loc="upper left", bbox_to_anchor=(-0.02, 1.02), reverse=True)
        plt.ylabel("PC 2", size=20)
        plt.xlabel("PC 1", size=20)
        return fig, ax

    def index_to_index_struc_site(index, n_sites):
        """Go from index in a list containing all atomic environments
        of a given element to a tuple (index_struc, index_site)

        Args:
            index (int): index in the list of atomic environments
            n_sites (int): number of sites in the structure of the element
        """
        index_struc = index // n_sites
        index_site = index % n_sites
        return (index_struc, index_site)

    def perform_direct_sampling(
        descriptors_by_element, n_clusters, threshold_init=0.15, plot=True
    ):
        # Get the symbols of the atoms in the structure
        DIRECT_sampler = DIRECTSampler(
            structure_encoder=False,
            clustering=BirchClustering(n=n_clusters, threshold_init=threshold_init),
            select_k_from_clusters=SelectKFromClusters(k=1),
        )
        DIRECT_selection = {}
        for s, descriptors in descriptors_by_element.items():
            print(f"Sampling {s}")
            # Concatenate descriptors along 1st axis
            descriptors = np.concatenate(descriptors, axis=0)
            DIRECT_selection[s] = DIRECT_sampler.fit_transform(descriptors)
            print(
                f"DIRECT selected {len(DIRECT_selection[s]['selected_indexes'])} from {len(DIRECT_selection[s]['PCAfeatures'])} environments "
            )
            all_features = DIRECT_selection[s]["PCAfeatures"]
            selected_indexes = DIRECT_selection[s]["selected_indexes"]
            if plot:
                plot_PCAfeature_coverage(all_features, selected_indexes)
        return DIRECT_selection

    symbols = [list(OrderedDict.fromkeys(a.get_chemical_symbols())) for a in traj]
    symbols_unique = list(OrderedDict.fromkeys(itertools.chain.from_iterable(symbols)))
    symbols_a0 = traj[0].get_chemical_symbols()
    counter = {symbol: symbols_a0.count(symbol) for symbol in symbols_unique}
    # Get indices of each species in the symbols list
    indices = {}
    i = 0
    for s, c in counter.items():
        indices[s] = (i, i + c)
        # print(s, symbols_a0[i:i + c])
        i += c
    # Generate SOAP descriptors for the environments of each species
    descriptors_by_element = get_soap_descriptors(traj, symbols_unique)
    DIRECT_selection = perform_direct_sampling(
        descriptors_by_element, n_clusters, threshold_init=threshold_init, plot=plot
    )

    basis_set = {}
    for symbol, n_sites in counter.items():
        index_start = indices[symbol][0]
        basis_set[symbol] = [index_to_index_struc_site(index, n_sites=n_sites) for index in DIRECT_selection[symbol]['selected_indexes']]
        basis_set[symbol] = [(b[0], b[1] + index_start) for b in basis_set[symbol]]
    # Sanity check
    for symbol, ind in indices.items():
        min_env = min([b[1] for b in basis_set[symbol]])
        max_env = max([b[1] for b in basis_set[symbol]])
        # Assert min and max are within the range of indices
        assert min_env >= indices[symbol][0]
        assert max_env < indices[symbol][1]
    if index_from_1: # Add 1 because VASP starts from 1
        basis_set = {symbol: [(b[0] + 1, b[1] + 1) for b in basis_set[symbol]] for symbol in basis_set.keys()}

    mlab = ml_ab_from_trajectory(traj, basis_set=basis_set)
    return mlab