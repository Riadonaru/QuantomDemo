import gates
import qubits
import animation

gate1 = gates.hadamard_gate()
gate2 = gates.pauli_z_gate()
qubit = qubits.bloch_0()
animation.animate_transformation(qubit, [gate1, gate2])