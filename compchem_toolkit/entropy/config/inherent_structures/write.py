from ase.io import read, write
from ase.io.lammpsdata import write_lammps_data

traj = read("./selected_inherent_struc_pre_relaxation.traj", ":")
for i, atoms in enumerate(traj):
    atoms.symbols[atoms.symbols == "H"] = "Cd"
    atoms.symbols[atoms.symbols == "He"] = "Te"
    write_lammps_data(f"{i}.data", atoms, specorder=["Cd", "Te"], units="metal", masses=True)
    #write(f"{i}.xyz", images=[atoms,]) # specorder=["Cd", "Te"],)
    #write(f"{i}.dump", images=[atoms,], specorder=["Cd", "Te"], format="lammps-dump-text")
