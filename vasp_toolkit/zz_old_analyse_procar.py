### #! /usr/bin/python
#! ~/envs/aiida-vasp-dev/bin/python3.8 
import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='A crappy scipt to read PROCAR.')

parser.add_argument("-b", "--band", help="Get most important orbitals for band.", type=int, default=259)
parser.add_argument("-k", "--kpoint", help="Get most important orbitals for kpoint.", type=int, default=1)
parser.add_argument("-t", "--threshold", help="Threshold for a total atom contribution to band.", type=float, default=0.035)
parser.add_argument("-o", "--orbital", help="Threshold for orbital contribution.", type=float, default=0.005)
parser.add_argument("-r", "--reverse", help="Reverse list to get last occurence of band in case PROCAR has been rewritten", type=bool, default=False)
args = parser.parse_args()
band_number = args.band
kpoint_number = args.kpoint
reverse_list=args.reverse
#print("Selected band:", band_number)

cwd=os.getcwd()
#cwd="/home/uccaosq/Sn2SbS2I3/vacancies/vac_1_Sn_-2/parchg_gam"
path_PROCAR=str(cwd)+"/"+"PROCAR"
path_POSCAR=str(cwd)+"/"+"POSCAR"

#band_number=259
#Thresholds for orbital contribution. Atoms with a total contribution higher than threshold and whose orbitals with contributions higher than orb_threshold will be displayed
tot_threshold=args.threshold
orb_threshold=args.orbital

def read_file(myfile):
    if os.path.exists(myfile):
        with open(myfile) as f:
            ff=f.read()
        return ff
    else: 
        print("File does not exist!")

def get_struc_formula(myfile):
    formula=read_file(path_POSCAR).splitlines()[5:7]
    atoms_numbers={}
    for index,atom in enumerate(formula[0].split()):
        current_number=int(formula[1].split()[index])
        if index > 0 :
            last_number=sum([int(x) for x in formula[1].split()[:index]])
            atoms_numbers[atom]=str(1+last_number)+"--"+ str( current_number + last_number )
        else:
            atoms_numbers[atom]= '1'+"--"+ str(current_number)
    return atoms_numbers

def get_kpoint(kpoint_number):
    ff=read_file(path_PROCAR)
    procar=ff.split("k-point ")
    number_of_kpoints = len(procar) - 1 
    if kpoint_number > number_of_kpoints :
        print(f"Only {number_of_kpoints} in your grid pal")
    #print(procar[kpoint_number])
    procar_kpoint = procar[kpoint_number]
    print(f"Analysing kpoint {procar_kpoint[0:5]}") 
    return procar_kpoint # gets the bands for the kpoint 'kpoint_numer'

def get_band_for_kpoint(band_num, procar_kpoint):
    procar=procar_kpoint.split("band ")
    if reverse_list == True:
        procar=procar[int(len(procar)/2):] #Get second half of file in case it has overwritten
    selected_band = procar[int(band_num)].splitlines()
    dict_band={}
    dict_band["band_number"]=int(selected_band[0].split("#")[0])
    dict_band["energy"]=float(selected_band[0].split("#")[1].split("energy")[1])
    dict_band["occupancy"]=float(selected_band[0].split("#")[2].split("occ.")[1])
    orbital_names=selected_band[2].split()[1:]
    #print(orbital_names)
    orbital_contributions={}
    for line in selected_band[3:-1]:
        ion=line.split()
        orbitals={}
        ion_number = ion[0]
        for index,orbital in enumerate(ion[1:]):
            orbitals[orbital_names[index]] = orbital
        orbital_contributions[ion_number]=orbitals
    dict_band["orbitals"]=orbital_contributions
    return dict_band

def get_band(band_num):
    ff=read_file(path_PROCAR)
    procar=ff.split("band ")
    if reverse_list == True:
        procar=procar[int(len(procar)/2):] #Get second half of file in case it has overwritten
    selected_band = procar[int(band_num)].splitlines()
    dict_band={}
    dict_band["band_number"]=int(selected_band[0].split("#")[0])
    dict_band["energy"]=float(selected_band[0].split("#")[1].split("energy")[1])
    dict_band["occupancy"]=float(selected_band[0].split("#")[2].split("occ.")[1])
    orbital_names=selected_band[2].split()[1:]
    #print(orbital_names)  
    orbital_contributions={}
    for line in selected_band[3:-1]: 
        ion=line.split()
        orbitals={}
        ion_number = ion[0]
        for index,orbital in enumerate(ion[1:]):
            orbitals[orbital_names[index]] = orbital
        orbital_contributions[ion_number]=orbitals
    dict_band["orbitals"]=orbital_contributions
    return dict_band

def analyse_orbitals(
    dict_band, 
    tot_threshold = tot_threshold,
    orb_threshold = orb_threshold
    ):
    orbital = dict_band["orbitals"]
    best_orbitals = {}
    #print(orbital)
    for ion_number, ion_orbitals in orbital.items():
        if ion_number != 'tot' and float(ion_orbitals['tot']) > tot_threshold:
            important_orbitals={}
            for key in ion_orbitals.keys():
                if float(ion_orbitals[key]) > float(orb_threshold):
                    important_orbitals[key] = float(ion_orbitals[key])
            best_orbitals[ion_number]= important_orbitals
    return best_orbitals, dict_band

def display(
    best_orbitals, 
    dict_band
    ):
    print('Band', dict_band['band_number'])
    print('Energy', dict_band["energy"])
    df=pd.DataFrame(best_orbitals)
    print(df.transpose())
    #for ion_number, value in best_orbitals.items():
    #    print("Atom", ion_number)
    #    print("Significant orbital contributions \n", value)
    #    print("\n")

def analyse_procar(kpoint_number, band_number):
    if kpoint_number == 0 : # aanalyse all kpoints
        ff=read_file(path_PROCAR)
        procar=ff.split("k-point ")
        number_of_kpoints = len(procar) - 1
        print(f"Number of kpoints {number_of_kpoints}")
        for kpoint in range(1,number_of_kpoints,1):
            print(f"KPOINT {kpoint}")
            best_orbitals, dict_band = analyse_orbitals( get_band_for_kpoint(band_number,
                                                    procar[kpoint]) )
            display(best_orbitals, dict_band)
            print()
    else:
        # for several k-points:
        best_orbitals, dict_band = analyse_orbitals( get_band_for_kpoint(band_number,
                                                    get_kpoint(kpoint_number) ) )
        display(best_orbitals, dict_band)

def check_ions(nions,
               band_number1, 
               band_number2,
               tot_threshold = tot_threshold,
               orb_threshold = orb_threshold):
    ## check if certain ions are contributing to the band
    print(f"Checking contributions from ions {nions}...")
    if kpoint_number == 0 : # aanalyse all kpoints
        ff=read_file(path_PROCAR)
        procar=ff.split("k-point ")
        number_of_kpoints = len(procar) - 1
        for kpoint in range(1, number_of_kpoints, 1):
            for band_number in range(band_number1, band_number2+1, 1):
                #print(f"BAND {band_number}")
                best_orbitals, dict_band = analyse_orbitals( 
                                                    get_band_for_kpoint(band_number,
                                                                        procar[kpoint]),
                                                    tot_threshold,
                                                    orb_threshold )
                if all([str(nion) in best_orbitals.keys() for nion in nions]) == True :
                    print(f"Found contributions for BAND {band_number} and KPOINT {kpoint}")
                    #print(f"KPOINT {kpoint}")
                    #print(f"BAND {band_number}")
                    if [ best_orbitals[str(nion)]['tot'] > 0.05 for nion in nions] :
                        #print(f"BAND {band_number}")
                        display(best_orbitals, dict_band)                    
                        print()

print(get_struc_formula(path_POSCAR))
check_ions([13,14, 194, 195], 100, 500)

