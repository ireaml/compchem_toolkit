"""Function to assign the Wannier orbitals to the closest atom"""

import os
from typing import Optional, Union

from pymatgen.core.sites import Site
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ase_toolkit.pdb import read_pdb


class Wannier_centers:
    """
    Class to parse Wannier centers and find closest atoms.

    Args:
        filename (str):
            Path to the Wannier centers file generated by CP2K, in pdb format.
        verbose (bool):
            Whether to print classification.
            Defaults to True.

    Returns:
        Dict: Dictionary matching the Wannier function index to a list of tuples
            with the atoms closest to the function center (e.g.
            { wannier_index: [(species_sting, site_index, distance)] }
            )
    """

    def __init__(
        self,
        filename: str,
        verbose: Optional[bool] = True,
    ) -> None:
        # Set class variables
        self.verbose = verbose

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} does not exist.")
        atoms = read_pdb(filename)
        # Transform to pmg Structure
        aaa = AseAtomsAdaptor()
        struct = aaa.get_structure(atoms)
        atom_sites = [site for site in struct if site.species_string != "X0+"]
        struct_copy = Structure.from_sites(
            atom_sites
        )  # Real structure without wannier centers
        # Parse wannier centers
        wan_centers = [site for site in struct if site.species_string == "X0+"]
        print("Total number of wannier orbitals:", len(wan_centers))
        # Find atom closest to each center
        self.classification = (
            {}
        )  # Dict like {wannier_index: [(species_string, site_index, distance), ]}
        for i, center in enumerate(wan_centers):
            self.classification[i + 1] = sorted(  # convert to non-pythonic indexes
                [
                    (
                        neigh.species_string,
                        neigh.index,
                        round(neigh.distance(center), 1),
                    )
                    for neigh in struct_copy.get_neighbors(center, r=0.9)
                ],
                key=lambda x: x[1],  # Sort by distance between Wannier center and atom
            )
            if verbose:
                print(
                    f"Wannier center {i+1}:",  # convert to non-pythonic indexes
                    self.classification[i + 1],
                )

    def get_classification(self) -> dict:
        """
        Return the classidication dictionary with matches
        each Wannier function to the closest atoms.
        """
        return self.classification

    def get_wannier_of_element(
        self,
        element: str,
    ):
        """
        Parse the Wannier functons associated to the specified element.

        Args:
            element (str):
                Element symbol of the element which Wannier functions are desired.
        """
        return {
            k: v for k, v in self.classification.items() if v and v[0][0] == element
        }

    def get_wannier_of_site(
        self,
        site_index: int,
    ) -> dict:
        """
        Parse the Wannier functons associated to the specified site.

        Args:
            site_index (int):
                Pymatgen index of the site which Wannier functions are desired.
        """
        if isinstance(site_index, int):
            selected_wanniers = {
                k: v
                for k, v in self.classification.items()
                if v and v[0][1] == site_index
            }
            if self.verbose:
                print(selected_wanniers)
            return selected_wanniers
        else:
            print(f"The argument site_index must be an integer!")
