from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data
import os

# list folder sin current directory
folders = [f for f in os.listdir() if os.path.isdir(f)]
for folder in folders: # each Temperature folder
    print(folder)
    traj = read(f"{folder}/dump.lammpstrj", "::5")
    print(f"Len: {len(traj)}")
    for i, atoms in enumerate(traj):
        # Process atoms
        atoms.symbols[atoms.symbols == "H"] = "Cd"
        atoms.symbols[atoms.symbols == "He"] = "Te"
        write_lammps_data(
            f"{folder}/{i}.data", atoms,
            specorder=["Cd", "Te"], units="metal", masses=True
        )
    write(f"{folder}/selected_inherent_struc_pre_relaxation.traj", traj) # lammps_md.traj