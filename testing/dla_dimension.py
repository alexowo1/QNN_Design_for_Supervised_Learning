import numpy as np
import pennylane as qml
from pennylane.pauli import pauli_sentence


# --- low-level: matrices & effective generators --------------------------------
def _op_matrix(op, wire_order):
    return qml.matrix(op, wire_order=wire_order)

def _hermitian_generator_matrix(op, wire_order):
    """Return H such that op(theta) = exp(-i * theta * H)."""
    try:
        GenObs, coeff = qml.generator(op)  # PL convention: op = exp(i * coeff * θ * GenObs)
    except Exception:
        return None
    if GenObs is None or coeff is None:
        return None
    G = qml.matrix(GenObs, wire_order=wire_order)
    # exp(i * coeff * θ * G) == exp(-i * θ * H)  ->  H = -coeff * G
    return (-coeff) * G

def _effective_generators_from_ops(ops, wire_order):
    """
    For each parametrized gate in 'ops', compute H_eff = U_suffix H U_suffix^\dagger
    (suffix = product of all gates after it, evaluated at the supplied sample params).
    Returns list of Hermitian H_eff.
    """
    L = len(ops)
    d = 2 ** len(wire_order)
    suffix = [None] * (L + 1)
    suffix[L] = np.eye(d, dtype=complex)
    U = np.eye(d, dtype=complex)
    for i in range(L - 1, -1, -1):
        suffix[i] = U
        U = _op_matrix(ops[i], wire_order) @ U

    H_effs = []
    for i, op in enumerate(ops):
        H = _hermitian_generator_matrix(op, wire_order)
        if H is None:
            continue
        U_suf = suffix[i]
        H_eff = U_suf @ H @ U_suf.conj().T
        # enforce Hermiticity numerically
        H_eff = 0.5 * (H_eff + H_eff.conj().T)
        H_effs.append(H_eff)
    return H_effs

# main: build a tape from qnode.func, then lie_closure
def dim_dla_from_qnode(qnode, *, sample_args=(), sample_kwargs=None, matrix_backend=True, tol=None, max_iterations=10000):
    """
    Compute dim(g) for a PennyLane QNode *without* using qnode.tape.
    You MUST provide sample_args/kwargs that match the QNode signature (zeros are fine).
    """
    if sample_kwargs is None:
        sample_kwargs = {}

    # Rebuild a tape by calling the underlying Python function under a recording context
    with qml.tape.QuantumTape() as tape:
        qnode.func(*sample_args, **sample_kwargs)
    ops = list(tape.operations)
    wire_order = list(tape.wires)

    # Build effective generators at these sample parameters (Hermitian matrices)
    H_effs = _effective_generators_from_ops(ops, wire_order)

    # Decompose each H_eff -> PauliSentence on the same wire order
    pauli_gens = [qml.pauli_decompose(H, pauli=True, wire_order=wire_order) for H in H_effs]

    # Lie closure in Pauli space (fast). Don't set matrix=True here.
    dla_basis = qml.liealg.lie_closure(pauli_gens, pauli=True, tol=tol, max_iterations=max_iterations)

    return len(dla_basis), dla_basis

