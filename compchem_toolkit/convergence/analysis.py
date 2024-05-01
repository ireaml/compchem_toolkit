import matplotlib.pyplot as plt


def get_convergence_values(kpoints_file, threshold=0.001):
    with open(kpoints_file, "r") as f:
        kpoints = f.read()
    # Get values from 3rd column into list
    kpoints_energies_per_atom = [float(line.split()[2]) for line in kpoints.split("\n")[1:-1]]
    # Get encut values from 1st column into list
    data = [line.split() for line in kpoints.split("\n")[1:-1]]
    kpoints_values = [line[0].split("k")[1].split("_")[-1].split()[0] for line in data]
    conv_value = kpoints_energies_per_atom[-1]
    if conv_value == 0:
        print("   Energy of denser k-grid is 0 - likely the calculation didn't finish ok. Check it!")
        # Set conv_value to last value different from 0
        for e in kpoints_energies_per_atom[::-1]:
            if e != 0:
                conv_value = e
                break
    # Find first kgrid value that is within 1 meV/atom of the converged value
    for k, e in zip(kpoints_values, kpoints_energies_per_atom):
        if abs(e - conv_value) < threshold:
            k = [int(i) for i in k.split(",")]
            return k, abs(e - conv_value)


def parse_encut(encut):
    # Get values from 3rd column into list
    encut_energies_per_atom = [float(line.split()[2]) for line in encut.split("\n")[1:]]
    # Get encut values from 1st column into list
    encut_values = [int(line.split()[0][1:]) for line in encut.split("\n")[1:]]
    # return encut_values, encut_energies_per_atom

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(encut_values, encut_energies_per_atom, marker="o", color="#D4447E")
    # Draw lines +- 5 meV/atom from the last point (our most accurate value)
    for threshold, color in zip([0.005, 0.001], ("#E9A66C", "#5FABA2")):
        ax.hlines(
            y=encut_energies_per_atom[-1] + threshold,
            xmin=encut_values[0], xmax=encut_values[-1],
            color=color, linestyles="dashed",
            label=f"{1000*threshold} meV/atom"
        )
        ax.hlines(
            y=encut_energies_per_atom[-1] - threshold,
            xmin=encut_values[0], xmax=encut_values[-1],
            color=color, linestyles="dashed",
        )
        # Fill the area between the lines
        ax.fill_between(
            encut_values,
            encut_energies_per_atom[-1] - threshold,
            encut_energies_per_atom[-1] + threshold,
            color=color, alpha=0.08,
        )
    # Add laels
    ax.set_xlabel("ENCUT (eV)")
    ax.set_ylabel("Energy per atom (eV)")
    ax.legend(frameon=True)
    return fig

def parse_kpoints(kpoints_file, title=None):
    """Function to parse kpoints convergence results from the string produced by vaspup2.0
    and plot them."""
    with open(kpoints_file, "r") as f:
        kpoints = f.read()
    # Get values from 3rd column into list
    kpoints_energies_per_atom = [float(line.split()[2]) for line in kpoints.split("\n")[1:-1]]
    # Get encut values from 1st column into list
    data = [line.split() for line in kpoints.split("\n")[1:-1]]
    kpoints_values = [line[0].split("k")[1].split("_")[-1].split()[0] for line in data]
    # print(kpoints_values)
    #return kpoints_values, kpoints_energies_per_atom
    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(kpoints_values, kpoints_energies_per_atom, marker="o", color="#D4447E")
    # Draw lines +- 5 meV/atom from the last point (our most accurate value)
    for threshold, color in zip([0.005, 0.001], ("#E9A66C", "#5FABA2")):
        ax.hlines(
            y=kpoints_energies_per_atom[-1] + threshold,
            xmin=kpoints_values[0],
            xmax=kpoints_values[-1],
            color=color,
            linestyles="dashed",
            label=f"{1000*threshold} meV/atom"
        )
        ax.hlines(
            y=kpoints_energies_per_atom[-1] - threshold,
            xmin=kpoints_values[0],
            xmax=kpoints_values[-1],
            color=color,
            linestyles="dashed",
        )
        # Fill the area between the lines
        ax.fill_between(
            kpoints_values,
            kpoints_energies_per_atom[-1] - threshold,
            kpoints_energies_per_atom[-1] + threshold,
            color=color,
            alpha=0.08,
        )
    # Add axis labels
    ax.set_xlabel("KPOINTS")
    ax.set_ylabel("Energy per atom (eV)")
    # Rotate xticks
    ax.set_xticklabels(kpoints_values, rotation=90)
    ax.legend(frameon=True)
    if title:
        ax.set_title(title)
    return fig