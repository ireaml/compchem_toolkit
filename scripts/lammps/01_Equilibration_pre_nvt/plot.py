import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use("~/00_Packages/style_sheets/whitney.mplstyle")
import numpy as np
cols = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Function to read data from a file (assuming two columns: timestep, value)
def read_data(filename):
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]  # Extract timestep and value

# Read data from files
time_vol, vol = read_data("vol_ave_npt.dat")
time_lx, lx = read_data("lx_ave_npt.dat")
time_ly, ly = read_data("ly_ave_npt.dat")
time_lz, lz = read_data("lz_ave_npt.dat")

# Create figure and subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter plot for volume
axes[0].scatter(time_vol, vol, color=cols[0], s=5, alpha=0.6, label='Volume')
axes[0].plot(time_vol, vol, color=cols[0],  alpha=0.6)
axes[0].set_xlabel("Timestep")
axes[0].set_ylabel("Volume (Å³)")
axes[0].set_title("Volume Evolution")
axes[0].legend()
#axes[0].grid(True, linestyle='--', alpha=0.5)

# Scatter plot for lattice parameters
axes[1].scatter(time_lx, lx, s=5, color=cols[0], alpha=0.6, label='Lx')
axes[1].scatter(time_ly, ly, s=5, color=cols[1], alpha=0.6, label='Ly')
axes[1].scatter(time_lz, lz, s=5, color=cols[2], alpha=0.6, label='Lz')
axes[1].plot(time_lx, lx, color=cols[0], alpha=0.6)
axes[1].plot(time_ly, ly, color=cols[1], alpha=0.6)
axes[1].plot(time_lz, lz, color=cols[2], alpha=0.6)
axes[1].set_xlabel("Timestep")
axes[1].set_ylabel("Lattice Parameter (Å)")
axes[1].set_title("Lattice Parameters Evolution")
axes[1].legend()
#axes[1].grid(True, linestyle='--', alpha=0.5)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
# Save figure
fig.savefig("equilibration.pdf", bbox_inches="tight")
