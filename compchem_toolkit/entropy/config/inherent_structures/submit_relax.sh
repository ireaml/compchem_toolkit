for i in */ ;
do cd $i ;
echo $i ;
yes | cp ../{lammps_relax.in,job_relax} . ;
# python ./read.py ; # Create data files
# mpiexec -np 1 ${p}/lmp -k on g 1 -sf kk -pk kokkos -i lammps_relax.in  > lammps.out ;
qsub -N re_$i job_relax ;
cd .. ;
done