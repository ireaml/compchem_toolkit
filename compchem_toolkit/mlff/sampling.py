from maml.sampling.direct import DIRECTSampler, BirchClustering, SelectKFromClusters
import numpy as np
import matplotlib.pyplot as plt

def perform_direct_sampling(
        descriptors,
        n_clusters,
        threshold_init=0.15,
        plot=True,
        score=True
    ):
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
            # fig, ax = plt.subplots(figsize=(15, 4))
            x = np.arange(len(scores_MPF_DIRECT))
            x_ticks = [f"PC {n+1}" for n in range(len(x))]

            plt.figure(figsize=(15, 4))
            plt.bar(
                x + 0.6,
                scores_MPF_DIRECT,
                width=0.3,
                label=f"Coverage score = {np.mean(scores_MPF_DIRECT):.3f}",
            )
            # plt.xticks(x + 0.45, x_ticks, size=16)
            plt.yticks(np.linspace(0, 1.0, 6), size=16)
            plt.ylabel("Coverage score", size=20)
            plt.xlabel("Principal component", size=20)
            # Remove xticks
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.legend(shadow=True, loc="lower right", fontsize=16)
            return plt

        # Get the symbols of the atoms in the structure
        DIRECT_sampler = DIRECTSampler(
            structure_encoder=False,
            clustering=BirchClustering(n=n_clusters, threshold_init=threshold_init),
            select_k_from_clusters=SelectKFromClusters(k=1),
        )
        print(f"Sampling {s}")
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
            plot_PCAfeature_coverage(all_features, selected_indexes)
        if score:
            n_pcas = all_features.shape[1]
            scores_DIRECT = calculate_all_FCS(all_features, selected_indexes, b_bins=n_pcas)
            ax = plot_scores(scores_DIRECT)
        return DIRECT_selection