from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector, Parameter


def cyclic_permutation(layer, n_qubits):
    """Cyclic permutation shifted by (layer + 1)"""
    return [(i + (layer%(n_qubits-1)) + 1) % n_qubits for i in range(n_qubits)]


def serial(degree, trainable_blocks, scaling):
    feature = Parameter("x")
    params_per_block = 3  # Rot

    total_blocks = (trainable_blocks * (degree + 1))
    # Parameter dictionary
    weights = {
        "W": [ParameterVector(f"W_{r}", params_per_block) for r in range(total_blocks)],
    }

    qc = QuantumCircuit(2)  # main qubit + control qubit

    block_idx = 0
    # Alternating W(theta) and S(x)
    for i in range(degree):
        for _ in range(trainable_blocks):
            qc.rz(weights["W"][block_idx][0], 0)
            qc.ry(weights["W"][block_idx][1], 0)
            qc.rz(weights["W"][block_idx][2], 0)
            qc.cx(0, 1)
            block_idx += 1
        # S(x)
        qc.rx(scaling * feature, 0)

    # final W
    for _ in range(trainable_blocks):
        qc.rz(weights["W"][block_idx][0], 0)
        qc.ry(weights["W"][block_idx][1], 0)
        qc.rz(weights["W"][block_idx][2], 0)
        qc.cx(0, 1)
        block_idx += 1

    return qc, weights, feature, "serial"


def strongly_parallel(n_qubits, trainable_layers, scaling):
    feature = Parameter("x")
    n_params_per_layer = 3 * n_qubits  # Rot + CNOT

    # Parameter dictionary
    weights = {
        "W": [[ParameterVector(f"W_{h}_{l}", n_params_per_layer) for l in range(trainable_layers)] for h in range(2)],
    }

    qc = QuantumCircuit(n_qubits)

    def W_layer(theta, permutation):
        # Rot on all qubits
        for i in range(0, 3*n_qubits, 3):
            # qc.rv(theta[i], theta[i+1], theta[i+2], i//3)
            qubit = i // 3
            qc.rz(theta[i], qubit)
            qc.ry(theta[i+1], qubit)
            qc.rz(theta[i+2], qubit)
        # Strongly entangling CNOT pattern: i -- perm[i]
        for i in range(n_qubits):
            j = permutation[i]
            qc.cx(i, j)

    # W(theta)
    for l in range(trainable_layers):
        permutation = cyclic_permutation(l, n_qubits)
        W_layer(weights["W"][0][l], permutation)

    # S(x)
    for i in range(n_qubits):
        qc.rx(scaling * feature, i)

    # W(theta)
    for l in range(trainable_layers):
        permutation = cyclic_permutation(l, n_qubits)
        W_layer(weights["W"][1][l], permutation)

    return qc, weights, feature, "strongly parallel"


def strongly_crz(n_qubits, trainable_layers, scaling):
    feature = Parameter("x")
    n_params_per_layer = 2 * n_qubits  # RY + CRZ (1 per qubit)

    # Parameter dictionary
    weights = {
        "W": [[ParameterVector(f"W_{h}_{l}", n_params_per_layer) for l in range(trainable_layers)] for h in range(2)],
        "final": [ParameterVector(f"final_{h}", n_qubits) for h in range(2)]
    }

    qc = QuantumCircuit(n_qubits)

    def W_layer(theta, permutation):
        # RY on all qubits
        for i in range(n_qubits):
            qc.ry(theta[i], i)
        # Strongly entangling CRZ pattern: i -- perm[i]
        for i in range(n_qubits):
            j = permutation[i]
            qc.crz(theta[n_qubits + i], i, j)

    # W(theta)
    for l in range(trainable_layers):
        permutation = cyclic_permutation(l, n_qubits)
        W_layer(weights['W'][0][l], permutation)
    for i in range(n_qubits):
        qc.ry(weights['final'][0][i], i)

    # S(x)
    for i in range(n_qubits):
        qc.rx(scaling * feature, i)

    # W(theta)
    for l in range(trainable_layers):
        permutation = cyclic_permutation(l, n_qubits)
        W_layer(weights['W'][1][l], permutation)
    for i in range(n_qubits):
        qc.ry(weights['final'][1][i], i)

    return qc, weights, feature, "strongly crz"


def strongly_rzz(n_qubits, trainable_layers, scaling):
    feature = Parameter("x")
    n_params_per_layer = 2 * n_qubits  # RY + RZZ (1 per qubit)

    # Parameter dictionary
    weights = {
        "W": [[ParameterVector(f"W_{h}_{l}", n_params_per_layer) for l in range(trainable_layers)] for h in range(2)],
        "final": [ParameterVector(f"final_{h}", n_qubits) for h in range(2)]
    }

    qc = QuantumCircuit(n_qubits)

    def W_layer(theta, permutation):
        # RY on all qubits
        for i in range(n_qubits):
            qc.ry(theta[i], i)
        # Strongly entangling RZZ pattern: i -- perm[i]
        for i in range(n_qubits):
            j = permutation[i]
            qc.rzz(theta[n_qubits + i], i, j)

    # W(theta)
    for l in range(trainable_layers):
        permutation = cyclic_permutation(l, n_qubits)
        W_layer(weights['W'][0][l], permutation)
    for i in range(n_qubits):
        qc.ry(weights['final'][0][i], i)

    # S(x)
    for i in range(n_qubits):
        qc.rx(scaling * feature, i)

    # W(theta)
    for l in range(trainable_layers):
        permutation = cyclic_permutation(l, n_qubits)
        W_layer(weights['W'][1][l], permutation)
    for i in range(n_qubits):
        qc.ry(weights['final'][1][i], i)

    return qc, weights, feature, "strongly rzz"


def all_to_all_crz(n_qubits, trainable_layers, scaling):

    feature = Parameter('x')  # single scalar input
    weights = {
        "W": [[ParameterVector(f"W_{h}_{l}", n_qubits + n_qubits*(n_qubits - 1)) for l in range(trainable_layers)] for h in range(2)],
        "final": [ParameterVector(f"final_{h}", n_qubits) for h in range(2)]
    }
    qc = QuantumCircuit(n_qubits)

    def W_layer(theta, qc):
        for i in range(n_qubits):
            qc.ry(theta[i], i)
        idx = n_qubits
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    qc.crz(theta[idx], i, j)
                    idx += 1

    for l in range(trainable_layers):
        W_layer(weights['W'][0][l], qc)
    for i in range(n_qubits):
        qc.ry(weights['final'][0][i], i)

    for i in range(n_qubits):
        qc.rx(scaling * feature, i)

    for l in range(trainable_layers):
        W_layer(weights['W'][1][l], qc)
    for i in range(n_qubits):
        qc.ry(weights['final'][1][i], i)

    return qc, weights, feature, "all-to-all crz"


def all_to_all_rzz(n_qubits, trainable_layers, scaling):
    """
    Qiskit version of the VQC with RY + RZZ entangling layers and RX encoding
    """
    feature = Parameter('x')  # single input scalar x
    weights = {
        "W": [[ParameterVector(f"W_{h}_{l}", n_qubits + (n_qubits * (n_qubits - 1))//2) for l in range(trainable_layers)] for h in range(2)],
        "final": [ParameterVector(f"final_{h}", n_qubits) for h in range(2)]
    }

    qc = QuantumCircuit(n_qubits)

    def W_layer(theta, qc):
        """Trainable block: RY + all-to-all RZZ entanglement."""
        # Apply RY to all qubits
        for i in range(n_qubits):
            qc.ry(theta[i], i)

        # All-to-all RZZ gates (i ≠ j)
        idx = n_qubits
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qc.rzz(theta[idx], i, j)
                idx += 1

    # W(theta)
    for l in range(trainable_layers):
        W_layer(weights['W'][0][l], qc)
    for i in range(n_qubits):
        qc.ry(weights['final'][0][i], i)

    # S(x)
    for i in range(n_qubits):
        qc.rx(scaling * feature, i)

    # W(theta)
    for l in range(trainable_layers):
        W_layer(weights['W'][1][l], qc)
    for i in range(n_qubits):
        qc.ry(weights['final'][1][i], i)

    return qc, weights, feature, "all-to-all rzz"


