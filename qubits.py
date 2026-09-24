import numpy as np
import random

def get_bloch_vector(state_vector):
    """
    Converts a 2D complex quantum state vector into a 3D Cartesian vector on the Bloch sphere.
    Expects state_vector as a list or array: [alpha, beta]
    """
    alpha = np.complex128(state_vector[0])
    beta = np.complex128(state_vector[1])

    # Calculate Theta
    # We use arctan2(|beta|, |alpha|) instead of arccos for numerical stability.
    # Because |alpha| = cos(theta/2) and |beta| = sin(theta/2), 
    # arctan(|beta|/|alpha|) gives theta/2 perfectly.
    theta = 2 * np.arctan2(np.abs(beta), np.abs(alpha))

    # Calculate Phi
    # Phi is the relative phase. Subtracting the angle of alpha from beta 
    # automatically factors out any irrelevant global phase in the vector.
    phi = np.angle(beta) - np.angle(alpha)

    # Convert spherical angles to 3D Cartesian coordinates
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Round to avoid floating point dust (e.g., returning 1e-16 instead of 0.0)
    return np.round([x, y, z], decimals=5).tolist()

def measure_qubit(bloch_vector):
    """
    Calculates the Z-basis probabilities from a Bloch vector and 
    simulates a single measurement outcome.
    
    Args:
        bloch_vector (list): A 3-element array [vx, vy, vz]
        
    Returns:
        tuple: (outcome, probabilities)
               outcome is either 0 or 1.
               probabilities is the [P(0), P(1)] list.
    """
    if len(bloch_vector) != 3:
        raise ValueError("The Bloch vector must have exactly 3 components (vx, vy, vz).")
        
    vz = bloch_vector[2]
    
    # Clip vz to [-1, 1] to handle minor floating-point inaccuracies 
    vz = max(-1.0, min(1.0, vz))
    
    p0 = (1 + vz) / 2.0
    p1 = (1 - vz) / 2.0
    
    probabilities = [p0, p1]
    
    # random.choices takes a list of options and a list of weights.
    # It returns a list, so we grab the first element [0].
    outcome = random.choices([[0, 0 , 1], [0, 0, -1]], weights=probabilities, k=1)[0]
    
    print("\n--- MEASUREMENT ---")
    print(f"Probabilities: P(0) = {p0:.4f}, P(1) = {p1:.4f}")
    print(f"Collapsed to: {0 if outcome[2] == 1 else 1}")
    return outcome


def bloch_0():
    """Returns the Bloch vector for the |0> state."""
    return get_bloch_vector([1, 0])

def bloch_1():
    """Returns the Bloch vector for the |1> state."""
    return get_bloch_vector([0, 1])

def bloch_plus():
    """Returns the Bloch vector for the |+> state."""
    return get_bloch_vector([1/np.sqrt(2), 1/np.sqrt(2)])

def bloch_minus():
    """Returns the Bloch vector for the |-> state."""
    return get_bloch_vector([1/np.sqrt(2), -1/np.sqrt(2)])
