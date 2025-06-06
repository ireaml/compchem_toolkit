import os
from ase.calculators.vasp import Vasp
from ase.io import write
from tqdm import tqdm
import tailer
from multiprocessing import Pool, cpu_count

# Number of parallel processes (adjust as needed; here set to 64)
N_PROCESSES = cpu_count()

def process_folder(folder):
    """
    Check if the VASP run in `folder` finished successfully.
    If so, load its output with ASE/Vasp(restart=True) and return the Atoms object.
    Otherwise, return None.
    """
    outcar_path = os.path.join(folder, "OUTCAR")
    if not os.path.exists(outcar_path):
        print(f"Calc {folder} hasn't an OUTCAR.")
        return None

    # Look at the last few lines of OUTCAR to see if the run completed
    last_lines = tailer.tail(open(outcar_path, "r"), 4)
    vol_in_lines = ["Voluntary context switches" in l for l in last_lines]
    if not any(vol_in_lines):
        print(f"Calc {folder} not finished")
        return None

    # Load the calculator from the VASP output directory
    try:
        calc_load = Vasp(restart=True, directory=folder)
        energy = calc_load.get_potential_energy()
        forces = calc_load.get_forces()
        atoms = calc_load.get_atoms()
        atoms.set_calculator(calc_load)
        atoms.info["id"] = f"val_{folder}"
        return atoms
    except Exception as e:
        print(f"Error loading {folder}: {e}")
        return None


if __name__ == "__main__":
    # Ensure the VASP pseudopotential path is set in each worker
    print("VASP_PP_PATH =", os.environ.get("VASP_PP_PATH", "<not set>"))

    # Collect all "scf_*" folders in the current directory
    folders = [d for d in os.listdir() if d.lower().startswith("scf")]
    folders.sort(key=lambda x: int(x.split("_")[1]))
    print("Found folders:", folders)

    # Process folders in parallel
    atoms_list = []
    with Pool(processes=N_PROCESSES) as pool:
        # imap_unordered lets us wrap with tqdm for progress
        for result in tqdm(pool.imap_unordered(process_folder, folders), total=len(folders)):
            if result is not None:
                atoms_list.append(result)

    # Write all successfully loaded Atoms objects to a single trajectory
    write("scf.traj", atoms_list)
    print(f"Wrote {len(atoms_list)} structures to scf.traj")