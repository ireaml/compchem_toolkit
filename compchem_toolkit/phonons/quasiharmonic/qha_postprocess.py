#!/usr/bin/env python

"""
Parse energy-volume data from 00_Relax directories
"""

# Generic imports
from pathlib import Path
import os
import shutil

from pymatgen.io.vasp.outputs import Vasprun

#output_files = sorted(Path().glob("./*_qha_*/Energy/vasprun.xml"))
dirs = sorted([d for d in os.listdir(".") if "qah" in d])
output_files = []
for d in dirs:
    if os.path.exists(f"{d}/00_Relax/vasprun.xml"):
        output_files.append(f"{d}/00_Relax/vasprun.xml")
        shutil.copy(f"{d}/00_Relax/CONTCAR", f"{d}/POSCAR")
print("Output files to parse:", output_files)

vols = []
e_tots = []
for f in output_files:
    print(f"Parsing {f}")
    v = Vasprun(
        filename=f,
        parse_dos=False,
        parse_eigen=False,
        parse_potcar_file=False,
    )
    if v.converged_electronic and v.converged_ionic:
        vols.append(v.structures[-1].volume)
        e_tots.append(v.final_energy)
    else:
        print(f"Vasprun {f} not converged!")

# write volume, energy to file
with open("e-v.dat", "w") as f:
    for vol, e_tot in zip(vols, e_tots):
        f.write(f"{vol:20.10f} {e_tot:20.10e}\n")

