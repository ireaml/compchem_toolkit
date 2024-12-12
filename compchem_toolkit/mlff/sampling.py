from maml.sampling.direct import DIRECTSampler, BirchClustering, SelectKFromClusters
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
            # plt.xticks(x + 0.45, x_ticks, size=16)
            ax.set_yticks(np.linspace(0, 1.0, 6), size=16)
            ax.set_ylabel("Coverage score", size=20)
            ax.set_xlabel("Principal component", size=20)
            # Remove xticks
            ax.set_xticks([])
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
        return DIRECT_selection, fig