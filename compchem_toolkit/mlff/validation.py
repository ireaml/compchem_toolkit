from ase.io import read, write
from tqdm.auto import tqdm
import numpy as np
from mace.calculators import MACECalculator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ase.atoms import Atoms

def mae(residuals):
    return np.mean(np.abs(residuals))


def rmse(residuals):
    return np.sqrt(np.mean(residuals**2))


def plot_energy_errors(traj_val, energies_dft, energies_mace, name_plots):
    # Energies
    errors = 1000 * ( np.array(energies_mace) - np.array(energies_dft))
    # Create dataframe
    df = pd.DataFrame({"DFT energy (eV)": energies_dft, "Error (meV)": errors})
    # Set thermostat as integer
    df["T (K)"] = np.ones(len(traj_val))
    # Plot using jointplot
    g = sns.jointplot(data=df, x="DFT energy (eV)", y="Error (meV)", hue="T (K)", kind="scatter", alpha=0.7,
                    edgecolor="none", palette=['#5FABA2', '#D4447E', '#E9A66C'])
    # Grey dashed line at 0
    g.ax_joint.axhline(0, color="grey", linestyle="--", alpha=1, linewidth=1.2)
    # Add MAE & RMSE as box to plot
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    # Per atom
    mae_per_atom = mae / len(traj_val[0])
    rmse_per_atom = rmse / len(traj_val[0])
    txt = f"MAE: {mae_per_atom:.2f} meV/atom\nRMSE: {rmse_per_atom:.2f} meV/atom"
    g.ax_joint.text(0.60, 0.95, txt, transform=g.ax_joint.transAxes, fontsize=12, verticalalignment="top",
                    bbox=dict(facecolor="#507BAA", edgecolor="none", alpha=0.1))
    # Move legend to the top left
    # g.ax_joint.legend(loc="upper left", title="T (K)")
    # Remove legend
    g.ax_joint.get_legend().remove()
    # Add title
    g.fig.suptitle("Energy", fontsize=15, y=1.02)
    # Save figure
    g.savefig(f"./test_errors_energies_{name_plots}.pdf", dpi=300, bbox_inches="tight")
    return g


def plot_force_errors(traj_val, forces_dft, forces_mace, name_plots):
    forces_mace_arr = np.vstack(forces_mace)
    forces_dft_arr = np.vstack(forces_dft)
    force_residuals = forces_mace_arr - forces_dft_arr
    # there are 17280 atomic environments in the test set
    # and therefore we have 17280 force residuals, each of which
    # is a 3-vector
    print(f"Shape of residuals:", force_residuals.shape)
    # we can take the (L2) norm of each force residual to get
    # the euclidean magnitude of the force residual
    # we can interpret this as the distance between the DFT and GAP forces
    force_residuals_magnitudes = np.linalg.norm(force_residuals, axis=1)
    print(f"Shape of magnitude of residuals:", force_residuals_magnitudes.shape)
    # now that we have scalar values, we can use the normal MAE and RMSE metrics
    print("Magnitude-wise errors")
    mag_mae = 1000*mae(force_residuals_magnitudes)
    mag_rmse = 1000*rmse(force_residuals_magnitudes)
    print("MAE: ", round(mag_mae, 1), "meV/Å")
    print("RMSE:", round(mag_rmse, 1), "meV/Å")
    # Component wise
    print("Component-wise errors")
    force_residuals_components = force_residuals.reshape(-1)
    comp_mae = 1000*mae(force_residuals_components)
    comp_rmse = 1000*rmse(force_residuals_components)
    print("MAE: ", round(comp_mae, 1), "meV/Å")
    print("RMSE:", round(comp_rmse, 1), "meV/Å")
    # Create dataframe
    df = pd.DataFrame({"F$_{\\alpha,DFT}$ (eV/Å)": forces_dft_arr.reshape(-1),
                       "Error (meV/Å)": 1000*force_residuals_components})
    temps_arr = np.ones(len(traj_val))
    df["Thermostat (K)"] = temps_arr.repeat(3*len(traj_val[0])).astype(int)
    # Plot using jointplot
    g_vec = sns.jointplot(data=df, x="F$_{\\alpha,DFT}$ (eV/Å)", y="Error (meV/Å)",
                    hue="Thermostat (K)", kind="scatter", alpha=0.1, edgecolor="none",
                    palette=['#5FABA2', '#D4447E', '#E9A66C'])
    # Grey dashed line at 0
    g_vec.ax_joint.axhline(0, color="grey", linestyle="--", alpha=1, linewidth=1.2)
    # Add MAE & RMSE as box to plot
    txt = f"MAE: {comp_mae:.0f} meV/Å\nRMSE: {comp_rmse:.0f} meV/Å"
    g_vec.ax_joint.text(0.08, 0.95, txt, transform=g_vec.ax_joint.transAxes, fontsize=12, verticalalignment="top",
                    bbox=dict(facecolor="#507BAA", edgecolor="none", alpha=0.1))
    g_vec.fig.suptitle("Force components", fontsize=15, y=1.02)
    # Move legend to top right
    # g_vec.ax_joint.legend(loc="upper right", title="Thermostat (K)")
    # Remove legend
    g_vec.ax_joint.get_legend().remove()
    g_vec.savefig(f"./test_errors_forces_vector_comps_{name_plots}.png", dpi=300, bbox_inches="tight", transparent=True)

    # Magnitude
    df = pd.DataFrame({
        "|F$_{DFT}$| (eV/Å)": np.linalg.norm(forces_dft_arr, axis=1),
        "Error (meV/Å)": 1000*force_residuals_magnitudes})
    # Plot with jointplot
    temps_arr = np.ones(len(traj_val))
    df["Thermostat (K)"] = temps_arr.repeat(len(traj_val[0])).astype(int)
    g_mag = sns.jointplot(data=df, x="|F$_{DFT}$| (eV/Å)", y="Error (meV/Å)",
                    hue="Thermostat (K)", kind="scatter", alpha=0.1, edgecolor="none",
                    palette=['#5FABA2', '#D4447E', '#E9A66C'])
    # Grey dashed line at 0
    g_mag.ax_joint.axhline(0, color="grey", linestyle="--", alpha=1, linewidth=1.2)
    # Add MAE & RMSE as box to plot
    txt = f"MAE: {mag_mae:.0f} meV/Å\nRMSE: {mag_rmse:.0f} meV/Å"
    g_mag.ax_joint.text(0.08, 0.95, txt, transform=g_mag.ax_joint.transAxes, fontsize=12, verticalalignment="top",
                    bbox=dict(facecolor="#507BAA", edgecolor="none", alpha=0.1,))
    # Increase alpha for legend markers
    # legend = g.ax_joint.get_legend()
    # Increse alpha for markers in legend
    # legend.legend_handles[0].set_alpha(0.6)
    # legend.legend_handles[1].set_alpha(0.6)
    # legend.legend_handles[2].set_alpha(0.6)
    g_mag.ax_joint.get_legend().remove()
    g_mag.fig.suptitle("Force magnitude", fontsize=15, y=1.02)
    g_mag.savefig(f"./test_errors_forces_magnitude_{name_plots}.png", dpi=300, bbox_inches="tight")
    return g_vec, g_mag


def plot_stress_errors(traj_val, stresses_dft, stresses_mace, name_plots):
    stress_mace_arr = np.vstack(stresses_mace)
    stress_dft_arr = np.vstack(stresses_dft)
    stress_residuals = stress_mace_arr - stress_dft_arr
    print(f"Shape of residuals:", stress_residuals.shape)
    # Magnitudes
    stress_residuals_magnitudes = np.linalg.norm(stress_residuals, axis=1)
    print(f"Shape of magnitude of residuals:", stress_residuals_magnitudes.shape)
    # now that we have scalar values, we can use the normal MAE and RMSE metrics
    mag_mae = 1000*mae(stress_residuals_magnitudes)
    mag_rmse = 1000*rmse(stress_residuals_magnitudes)
    print("MAE: ", round(mag_mae, 1), "meV/Å3")
    print("RMSE:", round(mag_rmse, 1), "meV/Å3")
    # Components
    stress_residuals_components = stress_residuals.reshape(-1)
    comp_mae = 1000*mae(stress_residuals_components)
    comp_rmse = 1000*rmse(stress_residuals_components)
    print("MAE: ", round(comp_mae, 1), "meV/Å")
    print("RMSE:", round(comp_rmse, 1), "meV/Å")
    ####
    temperatures = np.ones(len(traj_val))
    temperatures_extended = np.repeat(temperatures, 6)
    df = pd.DataFrame({
        "S$_{\\alpha,DFT}$ (meV/Å$^3$)": 1000*stress_dft_arr.reshape(-1),
        "Error (meV/Å$^3$)": 1000*stress_residuals_components,
        "Thermostat (K)": temperatures_extended.astype(int)
    })
    # Plot using jointplot
    a = 0.2
    g_vec = sns.jointplot(data=df, x="S$_{\\alpha,DFT}$ (meV/Å$^3$)", y="Error (meV/Å$^3$)", kind="scatter", alpha=a,
                          edgecolor="none", hue=df["Thermostat (K)"], palette=['#5FABA2', '#D4447E', '#E9A66C'])
    # Grey dashed line at 0
    g_vec.ax_joint.axhline(0, color="grey", linestyle="--", alpha=1, linewidth=1.2)
    # Move legent to top right
    # g.ax_joint.legend(loc="upper right", title="Thermostat (K)")
    g_vec.ax_joint.get_legend().remove()
    # Add MAE & RMSE as box to plot
    txt = f"MAE: {comp_mae:.1f} meV/Å$^3$\nRMSE: {comp_rmse:.1f} meV/Å$^3$"
    g_vec.ax_joint.text(0.08, 0.95, txt, transform=g_vec.ax_joint.transAxes, fontsize=12, verticalalignment="top",
                    bbox=dict(facecolor="#507BAA", edgecolor="none", alpha=0.1))
    # Title
    g_vec.fig.suptitle("Stress components", fontsize=15, y=1.02)
    # Save figure
    g_vec.savefig(f"test_errors_stress_vector_comps_{name_plots}.pdf", dpi=300, bbox_inches="tight")
    # Magnitude
    df = pd.DataFrame({
        "|S$_{DFT}$| (meV/Å$^3$)": 1000*np.linalg.norm(stress_dft_arr, axis=1),
        "Error (meV/Å$^3$)": 1000*stress_residuals_magnitudes,
        "Thermostat (K)": temperatures.astype(int)
    })
    g_mag = sns.jointplot(data=df, x="|S$_{DFT}$| (meV/Å$^3$)", y="Error (meV/Å$^3$)", kind="scatter",
                          alpha=0.7, edgecolor="none", hue="Thermostat (K)", palette=['#5FABA2', '#D4447E', '#E9A66C'])
    g_mag.ax_joint.axhline(0, color="grey", linestyle="--", alpha=1, linewidth=1.2)
    # Add MAE & RMSE as box to plot
    txt = f"MAE: {mag_mae:.1f} meV/Å$^3$\nRMSE: {mag_rmse:.1f} meV/Å$^3$"
    g_mag.ax_joint.text(0.08, 0.95, txt, transform=g_mag.ax_joint.transAxes, fontsize=12, verticalalignment="top",
                    bbox=dict(facecolor="#507BAA", edgecolor="none", alpha=0.1))
    g_mag.ax_joint.get_legend().remove()
    # Title
    g_mag.fig.suptitle("Stress magnitudes", fontsize=15, y=1.02)
    # Save figure
    g_mag.savefig(f"test_errors_stress_magnitude_{name_plots}.pdf", dpi=300, bbox_inches="tight")
    return g_vec, g_mag


def plot_validation(
    path_trajectory: str,
    path_mace_model: str,
    name_plots: str=None,
    path_mpl_style="/home/irea/00_Packages/style_sheets/whitney.mplstyle"
) -> dict:
    """Plot validation results for a MACE model.

    Args:
        path_trajectory (str): Path to the trajectory file. Must be compatible with ASE reader.
        path_mace_model (str): Path to the MACE model.
        name_plots (str, optional): Label to add to the filename of the saved plots. Defaults to None.
        path_mpl_style (str, optional): Path to the matplotlib style file to customise plots.
            Defaults to "/home/irea/00_Packages/style_sheets/whitney.mplstyle".

    Returns:
        dict: Dictionary with the matplotlib figures for energy, forces and stresses.
    """
    plt.style.use(path_mpl_style)

    mace_calc = MACECalculator(model_paths=path_mace_model, device="cuda")
    if isinstance(path_trajectory, str):
        traj_val = read(path_trajectory, ":")
    elif isinstance(path_trajectory, list) and isinstance(path_trajectory[0], Atoms):
        traj_val = path_trajectory

    energies_dft, forces_dft, stresses_dft = [], [], []
    energies_mace, forces_mace, stresses_mace = [], [], []
    for atoms in tqdm(traj_val):
        # DFT results
        energies_dft.append(atoms.get_potential_energy())
        forces_dft.append(atoms.get_forces())
        stresses_dft.append(atoms.get_stress())
        # MLFF prediction
        atoms.calc = mace_calc
        energies_mace.append(atoms.get_potential_energy())
        forces_mace.append(atoms.get_forces())
        stresses_mace.append(atoms.get_stress())
    if not name_plots:
        composition = traj_val[0].get_chemical_formula()
        num_atoms = traj_val[0].num_atoms
        name_plots = f"{composition}_{num_atoms}atoms"
    # Energies
    print("Plotting energy errors...")
    g_energy = plot_energy_errors(traj_val, energies_dft, energies_mace, name_plots)
    # Forces
    print("Plotting force errors...")
    g_vec, g_mag = plot_force_errors(traj_val, forces_dft, forces_mace, name_plots)
    # Stresses
    print("Plotting stress errors...")
    g_vec_stress, g_mag_stress = plot_stress_errors(traj_val, stresses_dft, stresses_mace, name_plots)
    dic_plots = {
        "energy": g_energy,
        "forces": (g_vec, g_mag),
        "stresses": (g_vec_stress, g_mag_stress)
    }
    return dic_plots