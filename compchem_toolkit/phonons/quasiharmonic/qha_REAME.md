Workflow to quasiharmonic phonons:

1. In QHA directory, generate input files for unit cell (POSCAR,INCAR,POTCAR,KPOINTS)
2. Run qha_preprocess.sh to generate distorted/expanded structures
3. Run relaxation fixing shape and volume of the cell
4. Run qha_postprocess.sh to parse energy-volume files and copy relaxed structures to their parent (phonon) directories (where phonon calcs will be run)
4. For each contracted/expanded structure, run phonopy to generate displacements and submit their force calculations
5. Finally parse forces from vaspruns using phonopy, generate thermal_properties.yaml data with phonopy (e.g. "phonopy -f disp-\*/vasprun.xml ; phonopy mesh.conf -t") and then run qha command:
   > 'for i in *qah*; do cd $i ; phonopy -f disp-*/vasprun.xml ; phonopy  ../mesh.conf  -t  ; cd .. ; done'
   > phonopy-qha ../e-v.dat ../*qah_*/thermal_properties.yaml -p -s --pressure 0.000101325 
