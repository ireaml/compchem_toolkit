import os
from ase.calculators.vasp import Vasp
from ase.io import write
from tqdm import tqdm
import tailer

#os.environ["VASP_PP_PATH"] = "/home/mmm1193/Scratch/VASP_PSEUDOS"
print("VASP_PP_PATH", os.environ["VASP_PP_PATH"])

atoms_list = []
folders = [d for d in os.listdir() if d.lower().startswith("scf")]
folders.sort(key=lambda x: int(x.split("_")[1]))
print(folders)
#folders = [d for d in folders if int(d.split("_")[1]) < 100]
for folder in tqdm(folders):
    if os.path.exists(f"{folder}/OUTCAR"):
        last_lines = tailer.tail(open(f"{folder}/OUTCAR"), 4)       
        vol_in_lines = ["Voluntary context switches" in l for l in last_lines]
        if not any(vol_in_lines):
            print(f"Calc {folder} not finished")
            continue
        # Load the calculator from the VASP output files
        calc_load = Vasp(restart=True, directory=folder)
        # Get energy and forces
        energy = calc_load.get_potential_energy()
        forces = calc_load.get_forces()
        # Add to atoms object
        atoms = calc_load.get_atoms()
        atoms.set_calculator(calc_load)
        atoms.info["id"] = f"c422_{folder}"
        atoms_list.append(atoms)
    else:
        print(f"Calc {folder} hasn't an OUTCAR.")
# Write to trajectory
write("scf_c422_V_Cl_0.traj", atoms_list)
