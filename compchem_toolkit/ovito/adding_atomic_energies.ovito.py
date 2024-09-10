from ovito.data import *
import numpy as np
import matplotlib.pyplot as plt

cmap_Cd = plt.cm.Blues
cmap_Te = plt.cm.Oranges

energies = np.load(
    'C:/Users/Irea/atomic_energies_4frames.npy'
)

def modify(frame: int, data: DataCollection):

    # This user-defined modifier function gets automatically called by OVITO whenever the data pipeline is newly computed.
    # It receives two arguments from the pipeline system:
    #
    #    frame - The current animation frame number at which the pipeline is being evaluated.
    #    data  - The DataCollection passed in from the pipeline system.
    #            The function may modify the data stored in this DataCollection as needed.
    #
    # What follows is an example code snippet doing nothing aside from printing the current
    # list of particle properties to the log window. Use it as a starting point for developing
    # your own data modification or analysis functions.

    if data.particles != None:
        print("There are %i particles with the following properties:" % data.particles.count)
        for property_name in data.particles.keys():
            print("  '%s'" % property_name)
        type_property = data.particles['Particle Type']
        data.particles_.create_property('Kinetic Energy', data=energies[frame])
        for property_name in data.particles.keys():
            print("  '%s'" % property_name)
        # Colors
        a = energies[frame][:32]
        norm_Cd = plt.Normalize(a.min(), a.max())
        colors_Cd = cmap_Cd(norm_Cd(a))[:, :3] # remove alpha
        a = energies[frame][32:]
        norm_Te = plt.Normalize(a.min(), a.max())
        colors_Te = cmap_Te(norm_Te(a))[:, :3] # remove alpha
        colors = np.vstack((colors_Cd, colors_Te))
        data.particles_.create_property('Color', data=colors)
        
