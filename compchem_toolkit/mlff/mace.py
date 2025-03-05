import os
from ase.io import read, write
import matplotlib.pyplot as plt

def parse_training_errors(filename, multihead=False):
    with open (filename, "r") as f:
        lines = f.readlines()
    if multihead:
        epoch_lines = [line for line in lines if "Epoch" in line][1:]
        # Split epoch_lines: odd numbers are 1st head, even numbers are 2nd head
        epoch_lines_1 = epoch_lines[::2]
        epoch_lines_2 = epoch_lines[1::2]
        epoch_number = [int(line.split("Epoch ")[1].split(":")[0]) for line in epoch_lines_1]
        rmse_E_1 = [float(line.split("RMSE_E_per_atom=")[1].split("meV")[0]) for line in epoch_lines_1]
        rmse_F_1 = [float(line.split("RMSE_F=")[1].split("meV / A")[0]) for line in epoch_lines_1]
        rmse_S_1 = [float(line.split("RMSE_stress=")[1].split("meV / A^3")[0]) for line in epoch_lines_1]
        rmse_E_2 = [float(line.split("RMSE_E_per_atom=")[1].split("meV")[0]) for line in epoch_lines_2]
        rmse_F_2 = [float(line.split("RMSE_F=")[1].split("meV / A")[0]) for line in epoch_lines_2]
        rmse_S_2 = [float(line.split("RMSE_stress=")[1].split("meV / A^3")[0]) for line in epoch_lines_2]
        return epoch_number, rmse_E_1, rmse_F_1, rmse_S_1, rmse_E_2, rmse_F_2, rmse_S_2
    else:
        epoch_lines = [line for line in lines if "Epoch" in line][1:]
        epoch_number = [int(line.split("Epoch ")[1].split(":")[0]) for line in epoch_lines]
        rmse_E = [float(line.split("RMSE_E_per_atom=")[1].split("meV")[0]) for line in epoch_lines]
        rmse_F = [float(line.split("RMSE_F=")[1].split("meV / A")[0]) for line in epoch_lines]
        rmse_S = [float(line.split("RMSE_stress=")[1].split("meV / A^3")[0]) for line in epoch_lines]
        return epoch_number, rmse_E, rmse_F, rmse_S


def plot_training_curve(filename, multihead=False, path_mpl_style=""):
    if os.path.exists(path_mpl_style):
        plt.style.use(path_mpl_style)
    else:
        print(f"Could not find a style sheet with path {path_mpl_style}. Using default style.")
    if multihead:
        epoch_number, rmse_E_1, rmse_F_1, rmse_S_1, rmse_E_2, rmse_F_2, rmse_S_2 = parse_training_errors(
            filename, multihead=multihead
        )
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        ax[0].plot(epoch_number, rmse_E_1, label="H1")
        ax[0].plot(epoch_number, rmse_E_2, label="H2")
        ax[0].set_ylabel("RMSE${_E} (meV)")
        ax[0].legend()
        ax[1].plot(epoch_number, rmse_F_1, label="H1")
        ax[1].plot(epoch_number, rmse_F_2, label="H2")
        ax[1].set_ylabel("RMSE${_F} (meV/A)")
        ax[2].plot(epoch_number, rmse_S_1, label="H1")
        ax[2].plot(epoch_number, rmse_S_2, label="H2")
        ax[2].set_ylabel("RMSE${_S} (meV/A^3)")
        ax[2].set_xlabel("Epoch")
    else:
        epoch_number, rmse_E, rmse_F, rmse_S = parse_training_errors(filename)
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        ax[0].plot(epoch_number, rmse_E)
        ax[0].set_ylabel("RMSE${_E} (meV)")
        ax[1].plot(epoch_number, rmse_F)
        ax[1].set_ylabel("RMSE${_F} (meV/A)")
        ax[2].plot(epoch_number, rmse_S)
        ax[2].set_ylabel("RMSE${_S} (meV/A^3)")
        ax[2].set_xlabel("Epoch")
    return fig, ax


def get_mace_db_from_mlab(
    mlab,
    key_energy="REF_energy",
    key_forces="REF_forces",
    key_stress="REF_stress"
):
    mlab.write_extxyz(filename="MLAB.extxyz", stress_unit="eV/A^3")
    traj = read("MLAB.extxyz", index=":")
    os.remove("MLAB.extxyz")
    # Set info for each atom
    for i in range(len(traj)):
        traj[i].arrays[key_forces] = traj[i].get_forces() # Needed for mace parsing!
        traj[i].info[key_energy] = traj[i].calc.results["energy"] # Parsed from VASP ML_AB with pymlff #.calc.results["free_energy"]
        traj[i].info[key_stress] = traj[i].get_stress() # voigt_6_to_full_3x3_stress(traj[i].get_stress())
        traj[i].info["original_energy"] = traj[i].calc.results["energy"]
    # Test it can be read OK by MACE
    write("db_wout_ref.xyz", traj, write_info=True, write_results=True)
    try:
        from mace.data.utils import load_from_xyz
        atomic_energies_dict, configs = load_from_xyz(
            file_path="db_wout_ref.xyz",
            config_type_weights={},
            energy_key=key_energy,
            stress_key=key_stress,
            forces_key=key_forces,
        )
    except ModuleNotFoundError:
        print("MACE is not installed. Cannot verify if the xyz file can be read by MACE.")
    return traj