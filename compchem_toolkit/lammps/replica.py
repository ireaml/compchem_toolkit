"""
Convenient (raw) code to reorder thermo output from LAMMPS simulations
(e.g. LAMMPS output is ordered by replicas rather than temperature, but we
are often interested in the output of each temperature).
"""
# Check if lammps_logfile is installed
try:
    import lammps_logfile
except ImportError:
    raise ImportError("Please install lammps_logfile using `pip install lammps-logfile`")
import matplotlib.pyplot as plt
import numpy as np

# Parse main log file, which list information about which temperature
# is assigned to each replica at each thermodynamic output timestep
# (see https://guriang.unpad.ac.id/hpc/lammpsdoc/temper.html)
with open("./log.lammps", "r") as f:
    lines = f.readlines()
# Get index of line with "Step" to separate Equilibration from Production
step_line = [i for i, line in enumerate(lines) if "Step" in line]
prod_data = lines[step_line[1]+1:] # Production data
prod_data = [line.split() for line in prod_data] # format data
for i, line in enumerate(prod_data):
    prod_data[i] = [float(x) for x in line]
steps = [line[0] for line in prod_data]
replicas = [line[1:] for line in prod_data]
# Can plot replica vs step with
# plt.plot(steps, replicas)

# Parse relevant quantities - in my case enthalpy
field = "Enthalpy"
log_replica = []
log_replica_steps = []
log_temp = []
for r in range(0, 9+1):
    # STore enthalpies from each replica in a list
    log = lammps_logfile.File(f"./log.lammps.{r}")
    log_replica.append(log.get(field))
    log_replica_steps.append(log.get("Step"))
    log_temp.append(log.get("Temp"))


# Reorder the data so we dont have mixed results from different temperatures

N_every = 10 # We output thermo data every 10 steps
N_eq = 1000 # We first equilibrated for 1000 steps - so our production data starts from 1000

logs_sorted = {k: [] for k in range(0, 9+1)}
temps_sorted = {k: [] for k in range(0, 9+1)} # To sanity check ordering of temperatures
for i in range(len(prod_data)-1):
    print(prod_data[i]) # this is step, and temperature ids
    step_initial = int(prod_data[i][0]) - N_eq
    step_next = int(prod_data[i+1][0]) - N_eq
    for replica_id, temp_id in enumerate(prod_data[i][1:]): # the position/index is the replica id, the value is the temperature id
        # print(f"Replica: {replica_id}, Temp: {temp_id}, Step_i: {step_initial}, Step_f: {step_next}") # for sanity checking
        logs_sorted[int(temp_id)].extend(log_replica[replica_id][step_initial//N_every:step_next//N_every])
        # print(log_replica_steps[replica_id][step_initial//N_every:step_next//N_every])
        # break
        temps_sorted[int(temp_id)].extend(log_temp[replica_id][step_initial//N_every:step_next//N_every])

# Plot
# For each temperature/key, plot the temperatures
for k, v in temps_sorted.items():
    x = N_eq + N_every * np.arange(len(v))
    plt.plot(x, v, label=f"T{k}", marker="o", markersize=2)
plt.legend(title="Replica")
# Label axis
plt.xlabel("Step")
plt.ylabel("Temperature")