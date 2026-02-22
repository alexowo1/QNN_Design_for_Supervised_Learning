import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
import jax
import jax.numpy as jnp
from pennylane.wires import Wires
from encodings import *

dev_name = "default.qubit"
diff_method = "best"


def cyclic_permutation(layer, n_qubits):
    # cyclic permutation shifted by layer
    return [(i + (layer % (n_qubits-1)) + 1) % n_qubits for i in range(n_qubits)]


def sort_cyclic_permutation(idx_list):
    n = len(idx_list)
    for i in range(n):
        swapped = False

        for j in range(i, n):
            if idx_list[i][1] == idx_list[j][0]:
                idx_list[j], idx_list[i+1] = idx_list[i+1], idx_list[j]
                swapped = True
        if not swapped:
            break


def toString_operations(operations):
    """transform operations to string"""
    operations_string = []

    for operation in operations:
        match operation:
            case qml.RX:
                operations_string.append("rx")
            case qml.RY:
                operations_string.append("ry")
            case qml.RZ:
                operations_string.append("rz")
            case qml.CNOT:
                operations_string.append("cnot")
            case qml.CY:
                operations_string.append("cy")
            case qml.CZ:
                operations_string.append("cz")
            case qml.CRX:
                operations_string.append("crx")
            case qml.CRY:
                operations_string.append("cry")
            case qml.CRZ:
                operations_string.append("crz")
            case qml.IsingXX:
                operations_string.append("rxx")
            case qml.IsingYY:
                operations_string.append("ryy")
            case qml.IsingZZ:
                operations_string.append("rzz")
            case qml.PauliZ:
                operations_string.append("z")
            case qml.PauliY:
                operations_string.append("y")
            case qml.PauliX:
                operations_string.append("x")

    return operations_string


def build_layer(operations, n_qubits, theta, layer=None, entangling_strength=None, measure_qubit=0):
    idx = 0
    for operation in operations:
        if operation in (qml.RX, qml.RY, qml.RZ):
            for i in range(measure_qubit, n_qubits):
                operation(theta[idx], wires=i)
                idx += 1
        elif operation in (qml.CNOT, qml.CY, qml.CZ):
            match entangling_strength:
                case "basic":
                    for i in range(n_qubits):
                        j = (i + 1) % n_qubits
                        operation(wires=[i, j])
                case "strongly":
                    for i, j in zip(range(n_qubits), cyclic_permutation(layer, n_qubits)):
                        operation(wires=[i, j])
                case "linked_permutated":
                    idx_list = list(zip(range(n_qubits), cyclic_permutation(layer, n_qubits)))
                    sort_cyclic_permutation(idx_list)
                    for i, j in idx_list:
                        operation(wires=[i, j])
                case "all_to_all":
                    for i in range(n_qubits):
                        for j in range(n_qubits):
                            if i != j:
                                operation(wires=[i, j])
                case "all_to_all_down":
                    for i in range(n_qubits):
                        for j in range(i+1, n_qubits):
                            operation(wires=[i, j])
        else:
            match entangling_strength:
                case "basic":
                    for i in range(n_qubits):
                        j = (i + 1) % n_qubits
                        operation(theta[idx], wires=[i, j])
                        idx += 1
                case "strongly":
                    for i, j in zip(range(n_qubits), cyclic_permutation(layer, n_qubits)):
                        operation(theta[idx], wires=[i, j])
                        idx += 1
                case "linked_permutated":
                    idx_list = list(zip(range(n_qubits), cyclic_permutation(layer, n_qubits)))
                    sort_cyclic_permutation(idx_list)
                    for i, j in idx_list:
                        operation(theta[idx], wires=[i, j])
                        idx += 1
                case "all_to_all":
                    for i in range(n_qubits):
                        for j in range(n_qubits):
                            if i != j:
                                operation(theta[idx], wires=[i, j])
                                idx += 1
                case "all_to_all_down":
                    for i in range(n_qubits):
                        for j in range(i+1, n_qubits):
                            operation(theta[idx], wires=[i, j])
                            idx += 1


def weight_initialisation_parameterized_entangling(n_qubits, trainable_layers, entangling_strength, measure_bool, operations_bool, random_key):
    match entangling_strength:
        case "basic" | "strongly" | "linked_permutated":
            layer_size = 2 * n_qubits + operations_bool * n_qubits
        case "all_to_all":
            layer_size = n_qubits**2 + operations_bool * n_qubits
        case "all_to_all_down":
            layer_size = (n_qubits**2 + n_qubits)//2 + operations_bool * n_qubits
        case _:
            raise NotImplementedError(f"{entangling_strength} is not implemented.")
    w_theta_size = trainable_layers * layer_size
    total_size = 2 * w_theta_size + n_qubits + 1 + measure_bool * (n_qubits-1) - operations_bool * 2 * n_qubits
    weights = jax.random.normal(random_key, shape=(total_size,)) * 0.01
    return weights, total_size, w_theta_size, layer_size


def weight_initialisation_nonparameterized_entangling(n_qubits, trainable_layers, measure_bool, operations_bool, random_key):
    layer_size = 2 * n_qubits + operations_bool * n_qubits
    w_theta_size = trainable_layers * layer_size
    total_size = 2 * w_theta_size + (n_qubits + 1 + measure_bool * (n_qubits-1)) * (2 + operations_bool)
    weights = jax.random.normal(random_key, shape=(total_size,)) * 0.01
    return weights, total_size, w_theta_size, layer_size


def observable(paired_measurement, measure_wires, measure_axis):
    # singles
    single_obs = [measure_axis(i) for i in measure_wires]
    n = len(single_obs)
    alpha = 1.0
    coeffs = [alpha / n] * n
    Obs = qml.Hamiltonian(coeffs, single_obs)
    if paired_measurement:
        # pairs
        pair_obs = [measure_axis(i) @ measure_axis((i+1) % len(measure_wires)) for i in measure_wires]
        m = len(pair_obs)
        beta = 1.0
        coeffs = ([alpha / n] * n) + ([beta / m] * m)
        Obs = qml.Hamiltonian(coeffs, single_obs + pair_obs)

    return Obs


def build_parameterized_entangling_model(entangling_strength, operations, n_qubits, trainable_layers, measure_wire, measure_axis, encoding, scaling, random_key, paired_measurement=False):
    """
    entangling_strength: "basic" | "strongly" | "linked_permutated" | "all_to_all | "all_to_all_down"
    
    operations: single-qubit rotation and two-qubit controlled rotation gates, maximum 3 operations, e.g. (qml.RY, qml.qml.CRZ) will result in RY-CRZ-RY gates per Qubit per trainable layer
    
    n_qubits: number of qubits, also determines frequencies
    
    trainable_layers: number of trainable layers per parameterized trainblock (2 trainblocks, W(𝜃)S(x)W(𝜃))
    
    measure_wire: on which Qubit(s) to apply the measurement at the end of the circuit

    measure_axis: on which axis to measure, choose between qml.PauliZ, qml.PauliY or qml.PauliX

    encoding: encoding of classical data with 1 or 2 input features, depending on circuit structure and input dimensionality
                --> for serial circuits serial encoding, for parallel circuits parallel encoding
                
                --> for 1 input feature univariate encoding, for 2 input features multivariate encoding

                choose between: univariate_serial_encoding, per_layer_multivariate_serial_encoding,
                    univariate_parallel_encoding, per_qubit_multivariate_parallel_encoding, across_qubits_multivariate_parallel_encoding
    
    scaling: 1 for standard angle encoding, 3 for exponential angle encoding
    
    random_key: pseudo-random number generator (PRNG) key given an integer seed

    paired_measurement: True or False, if multiple Qubits are measured, choose between average of single expvals or single + paired with nearest neighbor
        (e.g. PauliZ(wire) or PauliZ(wire) @ PauliZ(wire+1) for all quits)
    """
    dev = qml.device(dev_name, wires=n_qubits)
    # encode number of operations and measure-wires to a bool-like type (0 or 1)
    measure_bool = 0
    if isinstance(measure_wire, (range, list)):
        measure_bool = 1
    operations_bool = len(operations) - 2
    # make sure measure_wire is iterable for
    m_w = Wires(measure_wire)
    # for naming purposes
    operations_string = toString_operations(operations)
    measure_string = toString_operations([measure_axis])
    weights, total_size, w_theta_size, layer_size = weight_initialisation_parameterized_entangling(n_qubits, trainable_layers, entangling_strength, measure_bool, operations_bool, random_key)

    def W(theta, layer):
        """train block"""
        build_layer(operations, n_qubits, theta, layer, entangling_strength)

    if not operations_bool:
        name = f"{entangling_strength}_{operations_string[0]}_{operations_string[1]}_{operations_string[0]}_measured_at_{measure_wire if not measure_bool else "all"}_in_{measure_string[0] if not paired_measurement else measure_string[0] + "+" + 2 * measure_string[0]}"
        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def model(data, w):
            w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
            w2 = w[w_theta_size:w_theta_size+n_qubits]
            w3 = w[w_theta_size+n_qubits:-1 - measure_bool * (n_qubits-1)].reshape(trainable_layers, layer_size)
            # W(𝜃)
            for layer in range(trainable_layers):
                W(w1[layer], layer)
            # final rotation to each qubit
            build_layer([operations[0]], n_qubits, w2)

            # S(x)
            encoding(data, n_qubits, scaling)

            # W(𝜃)
            for layer in range(trainable_layers):
                W(w3[layer], layer)
            # final rotation(s) to measure-qubit(s)
            build_layer([operations[0]], m_w[-1] + 1, w[-1 - measure_bool * (n_qubits-1):], measure_qubit=m_w[0])

            # obs = [qml.PauliZ(wire) for wire in m_w]
            # coeffs = [1.0 / len(m_w)] * len(m_w)
            # O = qml.Hamiltonian(coeffs, obs)
            O = observable(paired_measurement, m_w, measure_axis)
            return qml.expval(O)
    else:
        name = f"{entangling_strength}_{operations_string[0]}_{operations_string[1]}_{operations_string[2]}_measured_at_{measure_wire if not measure_bool else "all"}_in_{measure_string[0] if not paired_measurement else measure_string[0] + "+" + 2 * measure_string[0]}"
        @qml.qnode(dev, interface="jax", diff_method=diff_method)
        def model(data, w):
            w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
            w2 = w[w_theta_size:-layer_size - (measure_bool-1) * (n_qubits-1)].reshape(trainable_layers-1, layer_size)
            w3 = w[-layer_size + (measure_bool-1) * (n_qubits-1):-1 - measure_bool * (n_qubits-1)]
            # W(𝜃)
            for layer in range(trainable_layers):
                W(w1[layer], layer)

            # S(x)
            encoding(data, n_qubits, scaling)

            # W(𝜃)
            for layer in range(trainable_layers-1):
                W(w2[layer], layer)
            # making sure only the measure qubit gets the last rotation
            build_layer(operations[:-1], n_qubits, w3, trainable_layers - 1, entangling_strength)
            # final rotation to measure-qubit
            build_layer([operations[-1]], m_w[-1] + 1, w[-1 - measure_bool * (n_qubits-1):], measure_qubit=m_w[0])

            # obs = [qml.PauliZ(wire) for wire in m_w]
            # coeffs = [1.0 / len(m_w)] * len(m_w)
            # O = qml.Hamiltonian(coeffs, obs)
            O = observable(paired_measurement, m_w, measure_axis)
            return qml.expval(O)

    return model, weights, total_size, name


def build_nonparameterized_entangling_model(entangling_strength, operations, n_qubits, trainable_layers, measure_wire, measure_axis, encoding, scaling, random_key, paired_measurement=False):
    dev = qml.device(dev_name, wires=n_qubits)
    # encode number of operations and measure-wires to a bool-like type (0 or 1)
    measure_bool = 0
    if isinstance(measure_wire, (range, list)):
        measure_bool = 1
    operations_bool = len(operations) - 3
    n_rot_params = 0
    for operation in operations:
        if operation in (qml.RX, qml.RY, qml.RZ):
            n_rot_params += 1
    # make sure measure_wire is iterable for
    m_w = Wires(measure_wire)
    # for naming purposes
    operations_string = toString_operations(operations)
    measure_string = toString_operations([measure_axis])
    name = f"{entangling_strength}_{operations_string[0]}_{operations_string[1]}_{operations_string[2]}{"_" + operations_string[3] if operations_bool else ""}_measured_at_{measure_wire if not measure_bool else "all"}_in_{measure_string[0] if not paired_measurement else measure_string[0] + "+" + 2*measure_string[0]}"
    weights, total_size, w_theta_size, layer_size = weight_initialisation_nonparameterized_entangling(n_qubits, trainable_layers, measure_bool, operations_bool, random_key)

    def W(theta, layer):
        """train block"""
        build_layer(operations, n_qubits, theta, layer, entangling_strength)

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:w_theta_size + n_qubits*n_rot_params]
        w3 = w[w_theta_size + n_qubits*n_rot_params:-n_rot_params - measure_bool * n_rot_params * (n_qubits-1)].reshape(trainable_layers, layer_size)
        # W(𝜃)
        for layer in range(trainable_layers):
            W(w1[layer], layer)
            # final rotation to each qubit
        build_layer(operations[:-1], n_qubits, w2)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in range(trainable_layers):
            W(w3[layer], layer)
        # last rotations to measure-qubit
        build_layer(operations[:-1], m_w[-1] + 1, w[-n_rot_params + measure_bool * n_rot_params * (n_qubits-1):], measure_qubit=m_w[0])

        # obs = [qml.PauliZ(wire) for wire in m_w]
        # coeffs = [1.0 / len(m_w)] * len(m_w)
        # O = qml.Hamiltonian(coeffs, obs)
        O = observable(paired_measurement, m_w, measure_axis)
        return qml.expval(O)

    return model, weights, total_size, name

