
import numpy as np
from ase import Atoms

def calculate_mean_structure_ase(traj) -> Atoms:
    """
    Calculates the mean structure from an ASE trajectory, correctly
    handling periodic boundary conditions by first unwrapping the coordinates.

    Args:
        trajectory_file: Path to the trajectory file (e.g., .traj, .xtc).

    Returns:
        An ASE Atoms object representing the mean structure, with coordinates
        wrapped back into the central simulation cell.
    """

    # --- Step A: Unwrap the Trajectory ---
    # Get the periodic cell and initialize unwrapped positions with the first frame
    cell = traj[0].get_cell()
    unwrapped_positions = [traj[0].get_positions()]

    # Loop through the trajectory to calculate continuous coordinates
    for i in range(1, len(traj)):
        displacement = traj[i].get_positions() - traj[i-1].get_positions()

        # Apply minimum image convention to the displacement
        scaled_disp = np.linalg.solve(cell.T, displacement.T).T
        mic_displacement = np.dot(scaled_disp - np.rint(scaled_disp), cell)

        new_unwrapped_pos = unwrapped_positions[-1] + mic_displacement
        unwrapped_positions.append(new_unwrapped_pos)

    unwrapped_positions_array = np.array(unwrapped_positions)

    # --- Step B: Calculate the Mean Positions ---
    mean_coords = np.mean(unwrapped_positions_array, axis=0)

    # --- Step C: Create and Return the Mean Structure ---
    # Create a new Atoms object with the mean positions
    mean_structure = Atoms(
        symbols=traj[0].get_chemical_symbols(),
        positions=mean_coords,
        cell=cell,
        pbc=traj[0].pbc
    )

    # Wrap the final average coordinates back into the primary cell for a clean output
    mean_structure.wrap()

    return mean_structure