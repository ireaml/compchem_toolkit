from ase.io import read, write
import numpy as np
from mace.calculators import MACECalculator
from tqdm.notebook import tqdm
from maml.sampling.direct import DIRECTSampler, BirchClustering, SelectKFromClusters, M3GNetStructure
# import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from maml.describers._site import SmoothOverlapAtomicPosition
import os

def generate_splits(path_mace: str, path_train_db: str, min_n_strucs=50, n_splits=10):

    def plot_PCAfeature_coverage(all_features, selected_indexes, method="DIRECT"):
        fig, ax = plt.subplots(figsize=(5, 5))
        selected_features = all_features[selected_indexes]
        plt.plot(all_features[:, 0], all_features[:, 1], "*", alpha=0.5, label=f"All {len(all_features):,} structures")
        plt.plot(
            selected_features[:, 0],
            selected_features[:, 1],
            "*",
            alpha=0.5,
            label=f"{method} sampled {len(selected_features):,}",
        )
        legend = plt.legend(frameon=True, fontsize=14, loc="upper left", bbox_to_anchor=(-0.02, 1.02), reverse=True)
        for lh in legend.legendHandles:
            lh.set_alpha(1)
        plt.ylabel("PC 2", size=20)
        plt.xlabel("PC 1", size=20)

    mace_calc = MACECalculator(model_paths=path_mace, device="cuda")
    db = read(path_train_db, index=":")
    # Separate isolated atoms from the training set
    isolated_atoms = []
    db_train = []
    for a in db:
        if a.info.get("config_type") == "IsolatedAtom":
            isolated_atoms.append(a)
        else:
            db_train.append(a)
    # Descriptors
    descriptors = []
    for a in tqdm(db_train):
        descriptors.append(mace_calc.get_descriptors(a))
    descriptors_mace_avg = [np.mean(desc, axis=0) for desc in descriptors]
    # Splits
    n_strucs = len(descriptors_mace_avg)
    n_strucs_per_split = np.linspace(min_n_strucs, n_strucs, n_splits).astype(int)
    # Sample
    %%time
    DIRECT_sampler = DIRECTSampler(
        clustering=BirchClustering(n=800, threshold_init=0.05),
        select_k_from_clusters=SelectKFromClusters(k=1),
        structure_encoder=None,
    )
    DIRECT_selection = DIRECT_sampler.fit_transform(X=descriptors_mace_avg)
    print(
        f"DIRECT selected {len(DIRECT_selection['selected_indexes'])} structures from {len(DIRECT_selection['PCAfeatures'])} structures."
    )
    samplers = []
    selections = []
    for num_strucs in n_strucs_per_split[:-1]:
        DIRECT_sampler = DIRECTSampler(
            clustering=BirchClustering(n=num_strucs, threshold_init=0.05),
            select_k_from_clusters=SelectKFromClusters(k=1),
            structure_encoder=None,
        )
        DIRECT_selection = DIRECT_sampler.fit_transform(X=descriptors_mace_avg)
        print(
            f"DIRECT selected {len(DIRECT_selection['selected_indexes'])} structures from {len(DIRECT_selection['PCAfeatures'])} structures."
        )
        selections.append(DIRECT_selection)
        samplers.append(DIRECT_sampler)

    list_selected_indexes = []
    for i, num_strucs in enumerate(n_strucs_per_split[:-1]):
        DIRECT_sampler = samplers[i]
        DIRECT_selection = selections[i]
        explained_variance = DIRECT_sampler.pca.pca.explained_variance_ratio_
        m = DIRECT_selection["PCAfeatures"].shape[1]
        DIRECT_selection["PCAfeatures_unweighted"] = DIRECT_selection["PCAfeatures"] / explained_variance[:m]
        all_features = DIRECT_selection["PCAfeatures_unweighted"]
        selected_indexes = DIRECT_selection["selected_indexes"]
        list_selected_indexes.append(selected_indexes)

    # Plot sampled structures for each Split
    fig, axs = plt.subplots(2, (n_splits)//2, figsize=(10, 4), sharex=True, sharey=True)
    axs = axs.flatten()
    zorder = 10
    markers = ["s", "D", "v", "^", "<", ">", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_", ".", ","]
    i = 0
    list_selected_indexes_with_all = list_selected_indexes + [np.arange(len(all_features))]
    for num_strucs, selected_indexes in zip(n_strucs_per_split[:], list_selected_indexes_with_all):
        selected_features = all_features[selected_indexes]
        zorder -= 1
        # Plot first the first 4 splits in the first row
        if i < 5:
            ax = axs[i]
            ax.scatter(all_features[:, 0], all_features[:, 1], marker="o", alpha=0.5, label=f"All ({len(all_features):,})", zorder=0, edgecolor="none",)
            ax.scatter(
                selected_features[:, 0],
                selected_features[:, 1],
                marker="x", #markers[i],
                alpha=0.5,
                label=len(selected_features),
                zorder=zorder,
                edgecolor="none",
            )
            ax.set_title(f"{len(selected_features):,}", size=14, y=0.91)
        else:
            ax = axs[i]
            ax.scatter(all_features[:, 0], all_features[:, 1], marker="o", alpha=0.5, label=f"All ({len(all_features):,})", zorder=0, edgecolor="none")
            ax.scatter(
                selected_features[:, 0],
                selected_features[:, 1],
                marker="x", #markers[i],
                alpha=0.5,
                label=len(selected_features),
                zorder=zorder,
                edgecolor="none",
            )
            ax.set_title(f"{len(selected_features):,}", size=14, y=0.91)
        i += 1
    # Remove y ticks
    for ax in axs:
        ax.set_yticks([])
        ax.set_xticks([])
    # Save figure
    fig.savefig("learn_direct_samplings.png", bbox_inches="tight", dpi=300, transparent=True)

    # Create folder for each split
    os.makedirs("Splits", exist_ok=True)
    for i, selected_indexes in enumerate(list_selected_indexes_with_all):
        # Precede folder name by 01_, 02_, etc. to ensure correct sorting
        folder_name = f"{i+1:02d}_{n_strucs_per_split[i]}"
        print(folder_name)
        folder_path = os.path.join("Splits", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        # Select structures from trajectory and save into folder
        strucs = [db_train[j] for j in selected_indexes]
        # Insert isolated atoms at the beginning
        strucs = isolated_atoms + strucs
        write(os.path.join(folder_path, "db.xyz"), strucs, write_results=True, write_info=True)
    # Save to npy list_selected_indexes_with_all
    np.save("list_selected_indexes_with_all.npy", list_selected_indexes_with_all)
    return list_selected_indexes_with_all