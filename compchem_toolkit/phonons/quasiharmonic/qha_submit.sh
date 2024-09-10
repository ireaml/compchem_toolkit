# Submit force calcs for each qah directory
for i in *qah*/ ; do 
	echo $i ; 
	cd $i ; 
	for d in disp-* ; do 
		cd $d ;
	        if [[ -s ../../job_phonon ]] ; then	
		    cp ../../job_phonon . ; 
		    sbatch --job-name ${i}_${d} job_phonon ;
	        else echo "Missing ../../job_phonon script!"
		fi;	
		cd .. ; 
	done ; 
	cd .. ; 
done
