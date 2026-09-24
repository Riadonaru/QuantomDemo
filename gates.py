import numpy as np

def unitary_to_bloch_rotation(U: np.ndarray) -> np.ndarray:
    """
    Transforms a 2x2 quantum gate (Unitary matrix) into a 3x3 rotation 
    matrix on the Bloch sphere.
    """
    # Define the Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    paulis = [sigma_x, sigma_y, sigma_z]
    
    # Initialize the 3x3 rotation matrix
    R = np.zeros((3, 3))
    
    # Precompute U adjoint (conjugate transpose)
    U_dagger = U.conj().T
    
    # Compute each element of the 3x3 rotation matrix
    for j in range(3):
        for k in range(3):
            # R_{jk} = 1/2 * Tr(sigma_j * U * sigma_k * U^dagger)
            matrix_product = paulis[j] @ U @ paulis[k] @ U_dagger
            
            # The result is strictly real, so we take the real part to 
            # discard floating point zero-imaginary artifacts
            R[j, k] = 0.5 * np.real(np.trace(matrix_product))
            
    return R


def hadamard_gate() -> np.ndarray:
    # Example 1: The Hadamard Gate
    # Swaps the X and Z axes, and flips the Y axis.
    H = (1 / np.sqrt(2)) * np.array([
        [1,  1],
        [1, -1]
    ], dtype=complex)
    
    return unitary_to_bloch_rotation(H)

def phase_gate() -> np.ndarray:
    # Example 2: Phase Gate (S Gate)
    # 90-degree rotation around the Z-axis.
    S = np.array([
        [1, 0],
        [0, 1j]
    ], dtype=complex)
    
    return unitary_to_bloch_rotation(S)

def pauli_z_gate() -> np.ndarray:
    # Example 3: Pauli-Z Gate
    # 180-degree rotation around the Z-axis.
    Z = np.array([
        [1, 0],
        [0, -1]
    ], dtype=complex)
    
    return unitary_to_bloch_rotation(Z)