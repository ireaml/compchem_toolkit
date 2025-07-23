"""Generate animations of phonon harmonic modes calculated with Phonopy."""

from phonopy.phonon.animation import write_animation
from phonopy.harmonic.dynamical_matrix import DynamicalMatrix
from phonopy.api_phonopy import Phonopy

def generate_animation(
    phonon: Phonopy,
    band_indices: list[int],
    q_point=[0, 0, 0],
):
    """
    Generate animations for specified phonon modes.

    Parameters:
    phonon: Phonopy object containing phonon calculations.
    band_indices: List of indices for the phonon modes to animate.
    """
    # Ensure band_indices is a list
    if not isinstance(band_indices, list):
        raise ValueError("band_indices must be a list of integers.")
    # Assert phonon.dynamical_matrix is available and phonopy.harmonic.dynamical_matrix.DynamicalMatrix
    dyn_matrix = phonon.dynamical_matrix
    if not isinstance(dyn_matrix, DynamicalMatrix):
        raise TypeError("phonon.dynamical_matrix must be an instance of DynamicalMatrix.")
    print("Shape of dynamical matrix:", dyn_matrix.shape) #(e.g. num_modes x num_modes)
    for band_index in band_indices: # modes with imaginary frequency and mode showing largest diff between neutral and +1
        q_point_str = str(q_point).replace(" ", "").replace("[", "").replace("]", "").replace(",", "")
        write_animation(
            dynamical_matrix=dyn_matrix,
            q_point=q_point,
            anime_type="xyz",
            filename=f"animation_q_{q_point_str}_b_{band_index}.xyz",
            band_index=band_index,
            amplitude=20,
            num_div=10,
            shift=0,
            factor=1,
        )
        print(f"Animation for band {band_index} written to animation_q_{q_point_str}_b_{band_index}.xyz")