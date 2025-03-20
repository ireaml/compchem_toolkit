from ase.io import read, write
import os

# list folders in current directory
folders = [f for f in os.listdir() if os.path.isdir(f)]
for folder in folders: # for each temperature folderåå
    print(folder)
    traj = read(f"{folder}/dump.lammpstrj", "::3")
    print(f"Len: {len(traj)}")
    write(f"{folder}/selected_inherent_struc_pre_relaxation.traj", traj)
