# For one temperature folder, check the progress of the inherent structure relaxations
grep -B 1 "Reading data file ..." log.lammps | tail -n2