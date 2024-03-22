#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: LiangTing
Email: lianting.zj@gmail.com
2020/03/04 21:42:48
Modified By: Irea Mosquera Lois
"""

from __future__ import division, print_function
import numpy as np
import os
import yaml

__all__ = ["phonon"]

class Phonon(object):

    def __init__(self, deal_file):

        self._deal_file = deal_file

        # phonon information
        self.qpoints_number = None  # get number of qpoints used  to  calculate
        self.poscar_atoms_number = None  # get number of atoms in POSCAR

        # Properties are for use within the class only
        self._phonon = None  # get phonon dictionary
        self._group_velocity = []
        self._frequencies = []
        self._eigenvectors = []
        self._qpoints = []
        self._distances = []
        # self._labels = []  # Labels specified are depicted in band structure plot at the points of band segments.

        # get participation
        self._participation = []

        if not os.path.isfile(self._deal_file):
            # Check that the .yaml file exists
            raise ValueError("\nSorry, the file " + self._deal_file + " does not exist or not in this folder.")
        else:
            print('\n' + self._deal_file + ' exists, Processing file is starting, please wait !!!\n')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def deal_yaml_file(self):
        with open(self._deal_file, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)  # Load the yaml file and convert it to a dictionary
            self.qpoints_number = data['nqpoint']  # get number of qpoints used to calculate
            self.poscar_atoms_number = data['natom']  # get number of atoms in poscar
            self.phonon = data.get('phonon')  # get phonon dictionary

    @property
    def get_frequencies(self):
        # Get phonon frequencies
        for phonon in self.phonon: # loop over qpoints
            self._qpoints.append(phonon['q-position'])
            distance = phonon.get('distance') # can be used to plot band structure
            if distance is not None:
                distance = phonon.get('distance_from_gamma')
            self._distances.append(distance)

            # frequencies
            for f in phonon['band']:  # loop over bands at this qpoint
                self._frequencies.append(f['frequency'])

        return np.array(self._frequencies)

    @property
    def get_eigenvectors(self):
        # Get phonon eigenvectors
        for phonon in self.phonon:  # loop over qpoints
            for ev in phonon['band']:  # loop over bands at this qpoint
                self._eigenvectors.append(ev['eigenvector'])

        return np.array(self._eigenvectors)

    @property
    def get_participation(self):

        # Get phonon participation
        participation_dict = {} # Mapping qpoint to band to participation. Just for saving to yaml file
        atomic_participations = {}
        for phonon in self.phonon:  # loop over qpoints
            participation_dict[str(phonon['q-position'])] = []
            atomic_participations[str(phonon['q-position'])] = {} # store atomic contribs to each band
            # although for us, we'd only use Gamma point
            for index, ev in enumerate(phonon['band']):  # loop over bands at this qpoint
                e_add = 0
                count = 0
                atomic_participations[str(phonon['q-position'])][index] = []
                for atom in ev['eigenvector']: # loop over atoms
                    x = complex(atom[0][0], atom[0][1])  # Build  plural
                    y = complex(atom[1][0], atom[1][1])
                    z = complex(atom[2][0], atom[2][1])
                    x2 = x * x.conjugate()  # Take the conjugate
                    y2 = y * y.conjugate()
                    z2 = z * z.conjugate()
                    e = (x2 + y2 + z2) ** 2  # The in-plane and out-of-plane phonon modes (e = x, y and z) are considered here
                    e_add += e.real          # Only the real part is taken because the imaginary part is 0 (Corresponds to the number of atoms in poscar)
                    count += 1               # Count the number of atoms in the system (equal to the number of atoms in POSCAR)
                    # atomic_participations[str(phonon['q-position'])][index].append(e.real)

                # Calculate atomic participation (how much each atom contributes to the phonon mode)
                # e.g. norm(realeigenvector[i,:])^2 / sumEsquarred
                for atom in ev['eigenvector']: # loop over atoms
                    x = complex(atom[0][0], atom[0][1])  # Build  plural
                    y = complex(atom[1][0], atom[1][1])
                    z = complex(atom[2][0], atom[2][1])
                    x2 = x * x.conjugate()  # Take the conjugate
                    y2 = y * y.conjugate()
                    z2 = z * z.conjugate()
                    e = (x2 + y2 + z2) ** 2
                    atomic_participations[str(phonon['q-position'])][index].append(
                        e.real / e_add
                    )

                if count != self.poscar_atoms_number:
                    raise ValueError('\nDeal_phonon_dispersion should not reach here!!!')
                else:
                    p = int(self.poscar_atoms_number) * e_add
                    self._participation.append(1 / p)
                    # Save to dict
                    participation_dict[str(phonon['q-position'])].append(
                        {
                            "Band_index": index,
                            "Frequency": ev['frequency'],
                            "Participation": 1 / p
                        }
                    )

        # Save to yaml file
        with open('participation_ratio.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(participation_dict, f, default_flow_style=False)
        # Same for atomic participations
        with open('atomic_participations.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(atomic_participations, f, default_flow_style=False)
        print('\n Participation ratio is generated OK ^_^')

        return np.array(self._participation)

    @property
    def get_group_velocity(self):
        # Get group_velocity
        for phonon in self.phonon:
            if 'group_velocity' in phonon['band'][0]:
                for v in phonon['band']:
                    Gv = v['group_velocity']
                    group_velocity = np.sqrt((((Gv[0]) * 100) ** 2 + ((Gv[1]) * 100) ** 2 + ((Gv[2]) * 100) ** 2))
                    self._group_velocity.append(group_velocity)
            else:
                self._group_velocity = None

        # Is exists the group_velocity?
        if self._group_velocity is None:

            raise ValueError('\nGroup_velocity does not exist')

        else:

            return np.array(self._group_velocity)

    @staticmethod
    def plot_phonon_figures(frequency, participation_ratio):
        import matplotlib.pyplot as plt
        # figure for participation_ratio
        fig, ax = plt.subplots(   # Set output image size
            nrows=1, ncols=1, figsize=(9.0, 6.0), dpi=300, tight_layout=True,
        )
        ax.plot(
            frequency, participation_ratio,
            'o', # "or"
            alpha=0.3,
        )
        # ax.set_ylims(0, 1)
        # ax.set_xlims(0, 16)
        # ax.set_xticks(fontsize=15)
        # ax.set_yticks(fontsize=15)
        ax.set_xlabel('Frequency (THz)', fontsize=15)
        ax.set_ylabel('Participation ratio', fontsize=15)
        ax.legend(fontsize=15, loc='best')
        #fig.show()  # show the plot on the screen
        fig.savefig('participation_ratio.png', dpi=300)  # save the plot as a png file
        # Exit the program
        plt.close(fig)

    def plot_dispersion(self):
        pass

    def writeToFile(self, file_name='phonon_inf.csv'):
        #import csv
        pass

if __name__ == "__main__":
    deal_file = 'mesh.yaml' # mesh.yaml or band.yaml
    ph = Phonon(deal_file)
    ph.deal_yaml_file()
    ph.plot_phonon_figures(ph.get_frequencies, ph.get_participation)
    print('PPR Calculate All Done\n')