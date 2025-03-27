from maml.sampling.direct import DIRECTSampler, BirchClustering, SelectKFromClusters
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mace.calculators import MACECalculator
from tqdm.notebook import tqdm
import os
from dscribe.descriptors import SOAP

def perform_direct_sampling(
        descriptors,
        n_clusters,
        threshold_init=0.15,
        plot=True,
        score=True
    ):
        def plot_PCAfeature_coverage(all_features, selected_indexes, jointplot=True):
            selected_features = all_features[selected_indexes]
            if jointplot:
                # Use seaborn jointplot
                df = pd.DataFrame(all_features[:, 0:2], columns=["PC 1", "PC 2"])
                df["Type"] = ["All"] * len(all_features)
                # Add selected features
                df_selected = pd.DataFrame(selected_features[:, 0:2], columns=["PC 1", "PC 2"])
                df_selected["Type"] = ["Selected"] * len(selected_features)
                df = pd.concat([df, df_selected], axis=0)
                g = sns.jointplot(
                    data=df,
                    x="PC 1",
                    y="PC 2",
                    hue="Type",
                    kind="scatter",
                    alpha=0.5,
                    height=5,
                    legend=True,
                    edgecolor="none"
                )
                g.ax_joint.collections[0].set_visible(False)
                g.ax_joint.scatter(
                    df["PC 1"], df["PC 2"],
                    marker="o", color="#5FABA2", alpha=0.3, label="All",
                )
                g.ax_joint.scatter(
                    df_selected["PC 1"], df_selected["PC 2"],
                    marker="*", label=f"Sampled {len(selected_features):,}",
                    color="#D4447E", alpha=0.3,
                )
                # Remove ticks
                g.ax_joint.set_xticks([])
                g.ax_joint.set_yticks([])
                g.ax_joint.set_xlim(min(df["PC 1"])-2, max(df["PC 1"])+2)
                g.ax_joint.set_ylim(min(df["PC 2"])-2, max(df["PC 2"])+2)
                return g

            else:
                fig, ax = plt.subplots(figsize=(5, 5))
                ax.plot(all_features[:, 0], all_features[:, 1], "*", alpha=0.5, label=f"All {len(all_features):,} envs")
                ax.plot(
                    selected_features[:, 0],
                    selected_features[:, 1],
                    "*",
                    alpha=0.5,
                    label=f"Sampled {len(selected_features):,}",
                )
                ax.legend(frameon=False, fontsize=14, loc="upper left", bbox_to_anchor=(-0.02, 1.02), reverse=True)
                ax.set_ylabel("PC 2", size=20)
                ax.set_xlabel("PC 1", size=20)
                return fig

        def calculate_all_FCS(all_features, selected_indexes, b_bins=100):
            def calculate_feature_coverage_score(all_features, selected_indexes, n_bins=100):
                selected_features = all_features[selected_indexes]
                n_all = np.count_nonzero(
                    np.histogram(all_features, bins=np.linspace(min(all_features), max(all_features), n_bins))[0]
                )
                n_select = np.count_nonzero(
                    np.histogram(selected_features, bins=np.linspace(min(all_features), max(all_features), n_bins))[0]
                )
                return n_select / n_all
            select_scores = [
                calculate_feature_coverage_score(all_features[:, i], selected_indexes, n_bins=b_bins)
                for i in range(all_features.shape[1])
            ]
            return select_scores

        def plot_scores(scores_MPF_DIRECT):
            x = np.arange(len(scores_MPF_DIRECT))
            x_ticks = [f"PC {n+1}" for n in range(len(x))]
            fig, ax = plt.subplots(figsize=(15, 4))
            ax.bar(
                x + 0.6,
                scores_MPF_DIRECT,
                width=0.3,
                label=f"Coverage score = {np.mean(scores_MPF_DIRECT):.3f}",
            )
            # ax.set_yticks(np.linspace(0, 1.0, 6))
            ax.set_ylabel("Coverage score")
            ax.set_xlabel("Principal component")
            # Remove xticks
            # ax.set_xticks(x + 0.45, x_ticks)
            ax.legend(shadow=True, loc="lower right", fontsize=16)
            return fig

        # Get the symbols of the atoms in the structure
        DIRECT_sampler = DIRECTSampler(
            structure_encoder=False,
            clustering=BirchClustering(n=n_clusters, threshold_init=threshold_init),
            select_k_from_clusters=SelectKFromClusters(k=1),
        )
        print(f"Sampling...")
        # Concatenate descriptors along 1st axis
        # descriptors = np.concatenate(descriptors, axis=0)
        DIRECT_selection = DIRECT_sampler.fit_transform(descriptors)
        print(
            f"DIRECT selected {len(DIRECT_selection['selected_indexes'])} from {len(DIRECT_selection['PCAfeatures'])} structures."
        )
        #print("Shape DIRECT_selection[s]['PCAfeatures']:", DIRECT_selection[s]['PCAfeatures'].shape)
        all_features = DIRECT_selection["PCAfeatures"]
        selected_indexes = DIRECT_selection["selected_indexes"]
        if plot:
            fig = plot_PCAfeature_coverage(all_features, selected_indexes)
            if score:
                n_pcas = all_features.shape[1]
                scores_DIRECT = calculate_all_FCS(
                    all_features, selected_indexes, b_bins=n_pcas
                )
                ax = plot_scores(scores_DIRECT)
            return DIRECT_selection, fig, ax
        return DIRECT_selection, None, None



def sample(
    db: list,
    n_clusters: int,
    path_MACE_model: str="",
    device="cuda",
    threshold_init=0.15,
    plot=True,
    score=True
) -> list:
    """
    Sample structures from a database using the DIRECT algorithm.

    Args:
        db (list): List of ASE atoms objects.
        n_clusters (int): Number of clusters to use for sampling.
        path_MACE_model (str): Path to the MACE model. If not provided, SOAP descriptors will be used.
        device (str): Device to use for MACE calculation of descriptors ("cuda" or "cpu").
            Default is "cuda".
        threshold_init (float): Initial threshold for clustering.
        plot (bool): Whether to plot the results showing a 2D descriptor map with
            the sampled structures and the full dataset.
        score (bool): Whether to calculate the coverage score (how well
            the sampled structures cover the full configurational space).

    Returns:
        list: List of selected structures.
    """
    if os.path.exists(path_MACE_model):
        mace_calc = MACECalculator(model_paths=path_MACE_model, device=device)
        mace_descriptors = []
        for atoms in tqdm(db):
            mace_descriptors.append(mace_calc.get_descriptors(atoms))
        # Save descriptors
        np.save("mace_descriptors.npy", mace_descriptors)
        mace_descriptors_avg = np.mean(mace_descriptors, axis=1)
    else:
        print(f"Path to MACE model {path_MACE_model} does not exist. Using SOAP descriptors.")
        # Use dscribe SOAP descriptors
        species = list(set([atom.symbol for atoms in db for atom in atoms]))
        soap = SOAP(
            species=species,
            r_cut=5.0,
            n_max=8,
            l_max=6,
            sigma=0.5,
            periodic=True,
            average="inner",
        )
        mace_descriptors_avg = []
        for atoms in tqdm(db):
            mace_descriptors_avg.append(soap.create(atoms))

    DIRECT_selection, fig, ax = perform_direct_sampling(
        mace_descriptors_avg,
        n_clusters,
        threshold_init=threshold_init,
        plot=plot,
        score=score
    )
    idx = DIRECT_selection['selected_indexes']
    traj_selected = [db[i] for i in idx]
    return traj_selected, fig