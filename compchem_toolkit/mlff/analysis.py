import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("../style/whitney.mplstyle")

def plot_max_force(trajs):
    max_force_per_config = []
    for traj in trajs:
        forces = [a.get_forces() for a in traj]
        # Mean abs force for each atom:
        mean_force = [np.mean(np.abs(f), axis=1) for f in forces]
        max_force_per_config.append([np.max(f, axis=0) for f in mean_force])
    # Plot distribution of forces in violin and strip plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_ylabel("Max $<F_{xyz}>$ (eV/Å)")
    ax.set_xlabel("System")
    sns.violinplot(data=max_force_per_config, ax=ax, cut=0, inner="quartile", linewidth=0.5, density_norm="count")
    # Add stripplot
    sns.stripplot(data=max_force_per_config, ax=ax, color="grey", alpha=0.5, size=2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    ax.set_xticks(np.arange(len(max_force_per_config)))
    ax.set_xticklabels([f"System {i+1}" for i in range(len(max_force_per_config))])
    return fig, ax

def plot_max_stress(trajs):
    mean_stress_per_config = []
    for traj in trajs:
        stresses = [a.get_stresses() for a in traj]
        # Mean abs stress for each atom:
        mean_stress_per_config = [np.mean(np.abs(s), axis=1) for s in stresses]
    # Plot distribution of stresses in violin and strip plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_ylabel("Max $<\sigma_{xyz}>$ (GPa)")
    ax.set_xlabel("System")
    sns.violinplot(data=mean_stress_per_config, ax=ax, cut=0, inner="quartile", linewidth=0.5, density_norm="count")
    # Add stripplot
    sns.stripplot(data=mean_stress_per_config, ax=ax, color="grey", alpha=0.5, size=2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    ax.set_xticks(np.arange(len(mean_stress_per_config)))
    ax.set_xticklabels([f"System {i+1}" for i in range(len(mean_stress_per_config))])
    return fig, ax

def plot_energies(trajs):
    energies = []
    for traj in trajs:
        energies.append([a.get_potential_energy() for a in traj])
    # Plot distribution of energies in violin and strip plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_ylabel("Energy (eV)")
    ax.set_xlabel("System")
    sns.violinplot(data=energies, ax=ax, cut=0, inner="quartile", linewidth=0.5, density_norm="count")
    # Add stripplot
    sns.stripplot(data=energies, ax=ax, color="grey", alpha=0.5, size=2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    ax.set_xticks(np.arange(len(energies)))
    ax.set_xticklabels([f"System {i+1}" for i in range(len(energies))])
    return fig, ax