# `compchem_toolkit`
Collection of `python` functions to work with `aiida`, `pymatgen`, `VASP`, `LAMMPS` and `LOBSTER`.
Designed to speed up input setup and output analysis.
The functions are organized in different subpackages: `aiida_tools`, `pymatgen`, `vasp`, `lobster`, `lammps`, `ase`, `cp2k` and `cli`.

1. `aiida_tools`
    * relax
      * submit_vasp_relax
    * singleshot
      * submit_vasp_singleshot
    * parsing
      * get_vasprun_from_pk
      * get_outcar_from_pk
      * get_dos_from_pk
      * transfer_chgcar
      * parse_chgcar
      * transfer_vasp_files
    * cp2k_wc
      * submit_cp2k_workchain
    * aiida_utils
      * get_options_dict
      * get_struct
2.  `vasp`
    * input
      * get_number_of_bands
      * check_paralellization
    * output
      * analyse_procar
      * plot_dos
      * make_parchg
    * potcar
      * get_potcar_valence_electrons
      * get_potcar_from_structure
      * get_potcars_from_mapping
      * get_number_of_electrons
      * get_valence_orbitals_from_potcar
    * bader
      * parse_acf
    * dynamics
      * get_selective_dynamics_from_index
    * hole_finder
      * hole_finder
    * magnetisation
      * site_magnetizations
3. `lammps`
   * parse_log
4. `pymatgen`
    * molecule
      * parse_molecule_from_sdf
    * slab
      * get_layer_sites
      * get_structure_from_slab
    * structure
      * get_atomic_disp
5. `lobster`
    * lobster_utils
      * get_labels_by_elements
      * calculate_mean_icohp
      * plot_cohp_for_label_list
6. `cli`
    * make_parchg
    * procar
      * analyse_procar


**NOTE**: Some of the functions have been adapted from external sources. In these cases,
the source is mentioned in the function docstring.
