import gates
import qubits
import animation

gate1 = gates.hadamard_gate()
gate2 = gates.pauli_z_gate()
gate3 = gates.phase_gate()
qubit1 = qubits.bloch_0()
animation.animate_transformation([qubit1], [gate1, gate2, gate1])