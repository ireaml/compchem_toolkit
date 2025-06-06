import os
from tqdm import tqdm
from pymatgen.io.vasp.outputs import BSVasprun
import numpy as np

list_levels = []
bandgaps = [] # Accounting for kpoints

list_folders = [
    f for f in os.listdir(".") if os.path.isdir(f"./{f}") and f.startswith("scf")]
print(len(list_folders))
list_folders.sort()
results = {}
for folder in tqdm(list_folders[:]):
    try:
        # print(folder)
        vr = BSVasprun(
            f'./{folder}/vasprun.xml',
            parse_potcar_file=False,
            #parse_dos=False,
            #parse_eigen=True
            parse_projected_eigen=False,
            #separate_spins=True,
        )
        band_props = vr.eigenvalue_band_properties
        results[folder] = [vr.eigenvalues, band_props]
    except:
        print(f"Couldnt parse {folder}")

# Save results
np.save("parsed_eigenvalues.npy", results, allow_pickle=True)