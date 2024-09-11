def parse_training_errors(filename):
    with open (filename, "r") as f:
        lines = f.readlines()
    epoch_lines = [line for line in lines if "Epoch" in line][1:]
    epoch_number = [int(line.split("Epoch ")[1].split(":")[0]) for line in epoch_lines]
    rmse_E = [float(line.split("RMSE_E_per_atom=")[1].split("meV")[0]) for line in epoch_lines]
    rmse_F = [float(line.split("RMSE_F=")[1].split("meV / A")[0]) for line in epoch_lines]
    rmse_S = [float(line.split("RMSE_stress_per_atom=")[1].split("meV / A^3")[0]) for line in epoch_lines]
    return epoch_number, rmse_E, rmse_F, rmse_S