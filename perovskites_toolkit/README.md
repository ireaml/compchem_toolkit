# perovskites-oct-distortions
Collection of python functions useful to manipulate the structures of hybrid organic/inorganic perovskites.
Main directory contains:
- `rotations.py` : functions to apply random rotations to the organic molecules occupiying the A site in your perovskite. Useful to break symmetries, avoid the polar parallel arrangement (if your cation has a dipole moment, e.g. formamdinium)
- folder `octahedral_distortions`: hosts the file `oct_distortions.py`. This contains the function `get_tilting_angles`, which calculates the average of the Pb-I-Pb angles in your structure. Values close to 180º indicate small tilting while angles deviating from 180º point towards significant tilting of the octahedra.
