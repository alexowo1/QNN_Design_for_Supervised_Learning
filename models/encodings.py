import pennylane as qml


def univariate_parallel_encoding(x, n_qubits, scaling):
    for w in range(n_qubits):
        qml.RX((scaling**w) * x, wires=w)


def across_qubits_multivariate_parallel_encoding(xy, n_qubits, scaling):
    """Encoding 1st feature on first half of available qubits and 2nd feature on second half."""
    x, y = xy
    half = n_qubits // 2
    for w in range(half):
        qml.RX((scaling**w) * x, wires=w)
    for w in range(half, n_qubits):
        qml.RX((scaling**(w-half)) * y, wires=w)


def across_qubits_multivariate_parallel_encoding_3x2y(xy, n_qubits, scaling):
    """Encoding 1st feature on first half of available qubits and 2nd feature on second half."""
    x, y = xy
    for w in range(3):
        qml.RX((scaling**w) * x, wires=w)
    for w in range(3, n_qubits):
        qml.RX((scaling**(w-3)) * y, wires=w)


def across_qubits_multivariate_parallel_encoding_2x3y(xy, n_qubits, scaling):
    """Encoding 1st feature on first half of available qubits and 2nd feature on second half."""
    x, y = xy
    for w in range(2):
        qml.RX((scaling**w) * x, wires=w)
    for w in range(2, n_qubits):
        qml.RX((scaling**(w-2)) * y, wires=w)


def super_parallel_encoding_xy(xy, n_qubits, scaling):
    """Encoding features in alternating order across qubits."""
    x, y = xy
    for q in range(n_qubits):
        if q % 2 == 0:
            qml.RX((scaling**q) * x, wires=q)
        else:
            qml.RX((scaling**q) * y, wires=q)


def super_parallel_encoding_yx(xy, n_qubits, scaling):
    """Encoding features in alternating order across qubits."""
    x, y = xy
    for q in range(n_qubits):
        if q % 2 == 0:
            qml.RX((scaling**q) * y, wires=q)
        else:
            qml.RX((scaling**q) * x, wires=q)


def across_qubits_multivariate_parallel_xy_encoding(xy, n_qubits, scaling):
    """Encoding 1st feature on first half of available qubits and 2nd feature on second half."""
    x, y = xy
    half = n_qubits // 2
    for w in range(half):
        qml.RX((scaling**w) * x, wires=w)
    for w in range(half, n_qubits):
        qml.RY((scaling**(w-half)) * y, wires=w)


def per_qubit_multivariate_parallel_encoding(xy, n_qubits, scaling):
    """encoding with RX-RY on each qubit"""
    x, y = xy
    for w in range(n_qubits):
        qml.RX((scaling**w) * x, wires=w)
        qml.RY((scaling**w) * y, wires=w)


# for serial models:

def univariate_serial_encoding(x, layer, scaling):
    qml.RX((scaling**layer) * x, wires=0)


def across_qubits_multivariate_serial_encoding(xy, layer, scaling):
    """Encoding 1st feature on first half of available qubits and 2nd feature on second half."""
    x, y = xy
    # half = n_qubits // 2
    # for w in range(half):
    #     qml.RX((scaling**w) * x, wires=w)
    # for w in range(half, 2):
    #     qml.RX((scaling**(w-half)) * y, wires=w)
    qml.RX((scaling**layer) * x, wires=0)
    qml.RX((scaling**layer) * y, wires=1)


def per_layer_multivariate_serial_encoding(xy, layer, scaling):
    x, y = xy
    qml.RX((scaling**layer) * x, wires=0)
    qml.RY((scaling**layer) * y, wires=0)