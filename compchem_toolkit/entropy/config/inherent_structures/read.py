from ase.io import read, write
import os

# list folder sin current directory
folders = [f for f in os.listdir() if os.path.isdir(f)]
for folder in folders:
    print(folder)
    traj = read(f"{folder}/dump.lammpstrj", "::3")
    print(f"Len: {len(traj)}")
    write(f"{folder}/lammps_md.traj", traj)
