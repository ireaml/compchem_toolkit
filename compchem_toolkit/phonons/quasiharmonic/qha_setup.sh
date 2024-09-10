# Generate displacements for each expanded/contracted structure
# Note that the expanded/contracted structure should be previously relaxed!

dim='2 2 2'
# --pa="AUTO"

for i in *qah*;
do cd $i ;
# cp 00_Relax/CONTCAR POSCAR;
phonopy -d --dim=$dim POSCAR >> phonopy_setup.out;
~/work/scripts/phono_setup.sh ;
cd ..;
done
