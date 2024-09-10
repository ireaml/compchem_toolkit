#!/usr/bin/env python

# Generic imports
import os
import shutil
from pathlib import Path
import numpy as np

# Reference for conveniently representing a path in python:
#   docs.python.org/3/library/pathlib.html

# Pymatgen
from pymatgen.core.structure import Structure
from pymatgen.io.vasp.inputs import Incar

# Read input geometry, save the lattice constant and convert to phonopy atoms object
initial_atoms= Structure.from_file("POSCAR")

scale_factors = np.linspace(0.95, 1.05, 11)

print(f"Initial volume: {initial_atoms.volume:.3f} \u212B^3")

for ii, factor in enumerate(scale_factors):
    i_fmt = "{:02d}".format(ii)
    unitcell = initial_atoms.copy()
    unitcell.scale_lattice(volume=initial_atoms.volume * factor)
    volume = unitcell.volume
    # create working directory and store input files
    workdir = Path(f"{i_fmt}_qah_{volume:.3f}")
    workdir.mkdir(exist_ok=True)
    unitcell.to(fmt="poscar", filename=str(workdir / "POSCAR"))

    # Make relaxation directory
    relax_dir = Path(f"{i_fmt}_qah_{volume:.3f}/00_Relax")
    relax_dir.mkdir(exist_ok=True)
    unitcell.to(fmt="poscar", filename=str(relax_dir / "POSCAR"))

    # Copy input files
    for input_file in ["INCAR", "POTCAR", "KPOINTS"]:
        if os.path.exists(input_file):
            shutil.copy(input_file, workdir); shutil.copy(input_file, relax_dir)

    # Update INCAR in relaxation directory
    incar = Incar.from_file(f"{relax_dir}/INCAR")
    incar.update({
        "IBRION": 1,
	"NSW": 2000,
	"ISIF": 2, # constant shape and volume
	"EDIFF": 10**(-8),
	"EDIFFG": -10**(-4),
	#"ISYM": 2, # symmetry on for relaxation
    })
    incar.write_file(f"{relax_dir}/INCAR")

    # prompt what we've done:
    print(f"Volume of sample {ii:4d}: {volume:.3f} \u212B^3")
    print(f"Input files written to: {workdir}\n")
    #print(f"Created relaxation directory, setting ISYM = 2 for this. Change if this is a defect!")
