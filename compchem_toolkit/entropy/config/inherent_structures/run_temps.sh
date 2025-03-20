p="/home/irea/01_Compile_LAMMPS/03_With_replica_misc_serial/lammps/build-kokkos-cuda"

conda activate mace

for i in */ ; # for each temperature
    do cd $i ;
    echo $i ;
    yes | cp ../{lammps_relax.in,struc.restart,MACE_model_swa.model-lammps.pt,read.py} . ;
    python ./read.py ; # Create data files for all inherent structures
    mpiexec -np 1 ${p}/lmp -k on g 1 -sf kk -pk kokkos -i lammps_relax.in  > lammps.out ;
    cd .. ;
done
