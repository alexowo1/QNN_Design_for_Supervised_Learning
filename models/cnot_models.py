import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
import jax
import jax.numpy as jnp
from encodings import *

dev_name = "default.qubit"
diff_method = "best"


def cyclic_permutation(layer, n_qubits):
    # cyclic permutation shifted by layer
    return [(i + (layer % (n_qubits-1)) + 1) % n_qubits for i in range(n_qubits)]


def toString_operations(rotations):
    rotations_string = []

    for rotation in rotations:
        match rotation:
            case qml.RX:
                rotations_string.append("rx")
            case qml.RY:
                rotations_string.append("ry")
            case qml.RZ:
                rotations_string.append("rz")

    return rotations_string


def basic_cnot_entangling(encoding, n_qubits, trainable_layers, scaling, random_key, rotations):
    dev = qml.device(dev_name, wires=n_qubits)
    # for returning name of model for later evaluation
    rotations_string = toString_operations(rotations)
    # calculating length of weight array
    n_rot_params = 3
    layer_size = n_qubits * n_rot_params
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + layer_size
    total_size = trainblock_size + w_theta_size + n_rot_params
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        for wire, (alpha, beta, gamma) in enumerate(theta):
            rotations[0](alpha, wire)
            rotations[1](beta, wire)
            rotations[2](gamma, wire)

        # Ring of CNOT
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.CNOT(wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, n_qubits, n_rot_params)
        w2 = w[w_theta_size:trainblock_size].reshape(n_qubits, n_rot_params)
        w3 = w[trainblock_size:-n_rot_params].reshape(trainable_layers, n_qubits, n_rot_params)
        # W(𝜃)
        for layer in w1:
            W(layer)
        for wire, (alpha, beta, gamma) in enumerate(w2):
            rotations[0](alpha, wire)
            rotations[1](beta, wire)
            rotations[2](gamma, wire)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in w3:
            W(layer)
        # last rotations to measure-qubit
        rotations[0](w[-3], 0)
        rotations[1](w[-2], 0)
        rotations[2](w[-1], 0)
        
        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, f"basic_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[2]}"


def strongly_cnot_entangling(encoding, n_qubits, trainable_layers, scaling, random_key, rotations):
    dev = qml.device(dev_name, wires=n_qubits)
    # for returning name of model for later evaluation
    rotations_string = toString_operations(rotations)
    # calculating length of weight array
    n_rot_params = 3
    layer_size = n_qubits * n_rot_params
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + layer_size
    total_size = trainblock_size + w_theta_size + n_rot_params
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta, permutation):
        for wire, (alpha, beta, gamma) in enumerate(theta):
            rotations[0](alpha, wire)
            rotations[1](beta, wire)
            rotations[2](gamma, wire)

        # permutating ring of CNOT
        for i, j in zip(range(n_qubits), permutation):
            qml.CNOT(wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, n_qubits, n_rot_params)
        w2 = w[w_theta_size:trainblock_size].reshape(n_qubits, n_rot_params)
        w3 = w[trainblock_size:-n_rot_params].reshape(trainable_layers, n_qubits, n_rot_params)
        # W(𝜃)
        for layer in range(n_qubits):
            W(w1[layer], cyclic_permutation(layer, n_qubits))
        for wire, (alpha, beta, gamma) in enumerate(w2):
            rotations[0](alpha, wire)
            rotations[1](beta, wire)
            rotations[2](gamma, wire)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in range(n_qubits):
            W(w3[layer], cyclic_permutation(layer, n_qubits))
        # last rotations to measure-qubit
        rotations[0](w[-3], 0)
        rotations[1](w[-2], 0)
        rotations[2](w[-1], 0)
        
        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, f"strongly_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[2]}"


def all_to_all_cnot_entangling(encoding, n_qubits, trainable_layers, scaling, random_key, rotations):
    dev = qml.device(dev_name, wires=n_qubits)
    # for returning name of model for later evaluation
    rotations_string = toString_operations(rotations)
    # calculating length of weight array
    n_rot_params = 3
    layer_size = n_qubits * n_rot_params
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + layer_size
    total_size = trainblock_size + w_theta_size + n_rot_params
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        for wire, (alpha, beta, gamma) in enumerate(theta):
            rotations[0](alpha, wire)
            rotations[1](beta, wire)
            rotations[2](gamma, wire)

        # all-to-all CNOT
        for j in range(n_qubits):
            for k in range(n_qubits):
                if j != k:
                    qml.CNOT(wires=[j, k])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, n_qubits, n_rot_params)
        w2 = w[w_theta_size:trainblock_size].reshape(n_qubits, n_rot_params)
        w3 = w[trainblock_size:-n_rot_params].reshape(trainable_layers, n_qubits, n_rot_params)
        # W(𝜃)
        for layer in w1:
            W(layer)
        for wire, (alpha, beta, gamma) in enumerate(w2):
            rotations[0](alpha, wire)
            rotations[1](beta, wire)
            rotations[2](gamma, wire)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in w3:
            W(layer)
        # last rotations to measure-qubit
        rotations[0](w[-3], 0)
        rotations[1](w[-2], 0)
        rotations[2](w[-1], 0)
        
        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, f"all_to_all_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[2]}"















"""old redundant models below"""


def basic_rot(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    n_rot_params = 3
    # W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, n_qubits, n_rot_params), minval=0, maxval=2 * jnp.pi)
    # final_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits, n_rot_params), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_weights}

    layer_size = n_qubits * n_rot_params
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + layer_size
    total_size = trainblock_size + w_theta_size + n_rot_params
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """train block"""
        # idx = 0
        # Rot to each qubit
        for wire, (alpha, beta, gamma) in enumerate(theta):
            qml.Rot(alpha, beta, gamma, wires=wire)
            # idx += 1

        # Ring of CNOT
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.CNOT(wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, n_qubits, n_rot_params)
        w2 = w[w_theta_size:trainblock_size].reshape(n_qubits, n_rot_params)
        w3 = w[trainblock_size:-n_rot_params].reshape(trainable_layers, n_qubits, n_rot_params)
        # W(𝜃)
        for layer in w1:
            W(layer)
        for wire, (alpha, beta, gamma) in enumerate(w2):
            qml.Rot(alpha, beta, gamma, wires=wire)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in w3:
            W(layer)
        # final rot to measure qubit
        qml.Rot(w[-3], w[-2], w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "basic_parallel"


def basic_ry_rz_ry(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    n_rot_params = 3
    layer_size = n_qubits * n_rot_params
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + layer_size
    total_size = trainblock_size + w_theta_size + n_rot_params
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """train block"""
        # RY-RZ-RY to each qubit
        for wire, (alpha, beta, gamma) in enumerate(theta):
            qml.RY(alpha, wires=wire)
            qml.RZ(beta, wires=wire)
            qml.RY(gamma, wires=wire)

        # Ring of CNOT
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.CNOT(wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, n_qubits, n_rot_params)
        w2 = w[w_theta_size:trainblock_size].reshape(n_qubits, n_rot_params)
        w3 = w[trainblock_size:-n_rot_params].reshape(trainable_layers, n_qubits, n_rot_params)
        # W(𝜃)
        for layer in w1:
            W(layer)
        for wire, (alpha, beta, gamma) in enumerate(w2):
            qml.RY(alpha, wires=wire)
            qml.RZ(beta, wires=wire)
            qml.RY(gamma, wires=wire)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in w3:
            W(layer)
        # final RY-RZ-RY to measure qubit
        qml.RY(w[-3], wires=0)
        qml.RZ(w[-2], wires=0)
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "basic_ry_rz_ry"


def basic_ry_rx_ry(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    n_rot_params = 3
    layer_size = n_qubits * n_rot_params
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + layer_size
    total_size = trainblock_size + w_theta_size + n_rot_params
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """train block"""
        # RY-RX-RY to each qubit
        for wire, (alpha, beta, gamma) in enumerate(theta):
            qml.RY(alpha, wires=wire)
            qml.RX(beta, wires=wire)
            qml.RY(gamma, wires=wire)

        # Ring of CNOT
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.CNOT(wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, n_qubits, n_rot_params)
        w2 = w[w_theta_size:trainblock_size].reshape(n_qubits, n_rot_params)
        w3 = w[trainblock_size:-n_rot_params].reshape(trainable_layers, n_qubits, n_rot_params)
        # W(𝜃)
        for layer in w1:
            W(layer)
        for wire, (alpha, beta, gamma) in enumerate(w2):
            qml.RY(alpha, wires=wire)
            qml.RX(beta, wires=wire)
            qml.RY(gamma, wires=wire)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in w3:
            W(layer)
        # final RY-RX-RY to measure qubit
        qml.RY(w[-3], wires=0)
        qml.RX(w[-2], wires=0)
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "basic_ry_rx_ry"


def basic_rz_rx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    n_rot_params = 3
    layer_size = n_qubits * n_rot_params
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + layer_size
    total_size = trainblock_size + w_theta_size + n_rot_params
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """train block"""
        # RZ-RX-RZ to each qubit
        for wire, (alpha, beta, gamma) in enumerate(theta):
            qml.RZ(alpha, wires=wire)
            qml.RX(beta, wires=wire)
            qml.RZ(gamma, wires=wire)

        # Ring of CNOT
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.CNOT(wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, n_qubits, n_rot_params)
        w2 = w[w_theta_size:trainblock_size].reshape(n_qubits, n_rot_params)
        w3 = w[trainblock_size:-n_rot_params].reshape(trainable_layers, n_qubits, n_rot_params)
        # W(𝜃)
        for layer in w1:
            W(layer)
        for wire, (alpha, beta, gamma) in enumerate(w2):
            qml.RZ(alpha, wires=wire)
            qml.RX(beta, wires=wire)
            qml.RZ(gamma, wires=wire)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in w3:
            W(layer)
        # final RZ-RX-RZ to measure qubit
        qml.RZ(w[-3], wires=0)
        qml.RX(w[-2], wires=0)
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "basic_rz_rx_rz"


def basic_ry_rx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    n_rot_params = 3
    layer_size = n_qubits * n_rot_params
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + layer_size
    total_size = trainblock_size + w_theta_size + n_rot_params
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """train block"""
        # RY-RX-RZ to each qubit
        for wire, (alpha, beta, gamma) in enumerate(theta):
            qml.RY(alpha, wires=wire)
            qml.RX(beta, wires=wire)
            qml.RZ(gamma, wires=wire)

        # Ring of CNOT
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.CNOT(wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, n_qubits, n_rot_params)
        w2 = w[w_theta_size:trainblock_size].reshape(n_qubits, n_rot_params)
        w3 = w[trainblock_size:-n_rot_params].reshape(trainable_layers, n_qubits, n_rot_params)
        # W(𝜃)
        for layer in w1:
            W(layer)
        for wire, (alpha, beta, gamma) in enumerate(w2):
            qml.RY(alpha, wires=wire)
            qml.RX(beta, wires=wire)
            qml.RZ(gamma, wires=wire)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in w3:
            W(layer)
        # final RY-RX-RZ to measure qubit
        qml.RY(w[-3], wires=0)
        qml.RX(w[-2], wires=0)
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "basic_ry_rx_rz"


def strongly_rot(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    n_rot_params = 3
    layer_size = n_qubits * n_rot_params
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + layer_size
    total_size = trainblock_size + w_theta_size + n_rot_params
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta, permutation):
        """train block"""
        # idx = 0
        # Rot to each qubit
        for wire, (alpha, beta, gamma) in enumerate(theta):
            qml.Rot(alpha, beta, gamma, wires=wire)
            # idx += 1

            # Ring of CNOT (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.CNOT(wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, n_qubits, n_rot_params)
        w2 = w[w_theta_size:trainblock_size].reshape(n_qubits, n_rot_params)
        w3 = w[trainblock_size:-n_rot_params].reshape(trainable_layers, n_qubits, n_rot_params)
        # W(𝜃)
        for layer in range(trainable_layers):
            W(w1[layer], cyclic_permutation(layer, n_qubits))
        for wire, (alpha, beta, gamma) in enumerate(w2):
            qml.Rot(alpha, beta, gamma, wires=wire)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in range(trainable_layers):
            W(w3[layer], cyclic_permutation(layer, n_qubits))
        # final rot to measure qubit
        qml.Rot(w[-3], w[-2], w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "strongly_parallel"


def strongly_ry_rz_ry(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    n_rot_params = 3
    layer_size = n_qubits * n_rot_params
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + layer_size
    total_size = trainblock_size + w_theta_size + n_rot_params
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta, permutation):
        """train block"""
        # idx = 0
        # RY-RZ-RY to each qubit
        for wire, (alpha, beta, gamma) in enumerate(theta):
            qml.RY(alpha, wires=wire)
            qml.RZ(beta, wires=wire)
            qml.RY(gamma, wires=wire)
            # idx += 1

            # Ring of CNOT (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.CNOT(wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, n_qubits, n_rot_params)
        w2 = w[w_theta_size:trainblock_size].reshape(n_qubits, n_rot_params)
        w3 = w[trainblock_size:-n_rot_params].reshape(trainable_layers, n_qubits, n_rot_params)
        # W(𝜃)
        for layer in range(trainable_layers):
            W(w1[layer], cyclic_permutation(layer, n_qubits))
        for wire, (alpha, beta, gamma) in enumerate(w2):
            qml.RY(alpha, wires=wire)
            qml.RZ(beta, wires=wire)
            qml.RY(gamma, wires=wire)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in range(trainable_layers):
            W(w3[layer], cyclic_permutation(layer, n_qubits))
        # final RY-RZ-RY to measure qubit
        qml.RY(w[-3], wires=0)
        qml.RZ(w[-2], wires=0)
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "strongly_ry_rz_ry"


def strongly_ry_rx_ry(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    n_rot_params = 3
    layer_size = n_qubits * n_rot_params
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + layer_size
    total_size = trainblock_size + w_theta_size + n_rot_params
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta, permutation):
        """train block"""
        # RY-RX-RY to each qubit
        for wire, (alpha, beta, gamma) in enumerate(theta):
            qml.RY(alpha, wires=wire)
            qml.RX(beta, wires=wire)
            qml.RY(gamma, wires=wire)

            # Ring of CNOT (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.CNOT(wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, n_qubits, n_rot_params)
        w2 = w[w_theta_size:trainblock_size].reshape(n_qubits, n_rot_params)
        w3 = w[trainblock_size:-n_rot_params].reshape(trainable_layers, n_qubits, n_rot_params)
        # W(𝜃)
        for layer in range(trainable_layers):
            W(w1[layer], cyclic_permutation(layer, n_qubits))
        for wire, (alpha, beta, gamma) in enumerate(w2):
            qml.RY(alpha, wires=wire)
            qml.RX(beta, wires=wire)
            qml.RY(gamma, wires=wire)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in range(trainable_layers):
            W(w3[layer], cyclic_permutation(layer, n_qubits))
        # final RY-RX-RY to measure qubit
        qml.RY(w[-3], wires=0)
        qml.RX(w[-2], wires=0)
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "strongly_ry_rx_ry"


def strongly_rz_rx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    n_rot_params = 3
    layer_size = n_qubits * n_rot_params
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + layer_size
    total_size = trainblock_size + w_theta_size + n_rot_params
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta, permutation):
        """train block"""
        # RZ-RX-RZ to each qubit
        for wire, (alpha, beta, gamma) in enumerate(theta):
            qml.RZ(alpha, wires=wire)
            qml.RX(beta, wires=wire)
            qml.RZ(gamma, wires=wire)

            # Ring of CNOT (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.CNOT(wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, n_qubits, n_rot_params)
        w2 = w[w_theta_size:trainblock_size].reshape(n_qubits, n_rot_params)
        w3 = w[trainblock_size:-n_rot_params].reshape(trainable_layers, n_qubits, n_rot_params)
        # W(𝜃)
        for layer in range(trainable_layers):
            W(w1[layer], cyclic_permutation(layer, n_qubits))
        for wire, (alpha, beta, gamma) in enumerate(w2):
            qml.RZ(alpha, wires=wire)
            qml.RX(beta, wires=wire)
            qml.RZ(gamma, wires=wire)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in range(trainable_layers):
            W(w3[layer], cyclic_permutation(layer, n_qubits))
        # final RZ-RX-RZ to measure qubit
        qml.RZ(w[-3], wires=0)
        qml.RX(w[-2], wires=0)
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "strongly_rz_rx_rz"


def strongly_ry_rx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    n_rot_params = 3
    layer_size = n_qubits * n_rot_params
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + layer_size
    total_size = trainblock_size + w_theta_size + n_rot_params
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta, permutation):
        """train block"""
        # RY-RX-RZ to each qubit
        for wire, (alpha, beta, gamma) in enumerate(theta):
            qml.RY(alpha, wires=wire)
            qml.RX(beta, wires=wire)
            qml.RZ(gamma, wires=wire)

            # Ring of CNOT (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.CNOT(wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, n_qubits, n_rot_params)
        w2 = w[w_theta_size:trainblock_size].reshape(n_qubits, n_rot_params)
        w3 = w[trainblock_size:-n_rot_params].reshape(trainable_layers, n_qubits, n_rot_params)
        # W(𝜃)
        for layer in range(trainable_layers):
            W(w1[layer], cyclic_permutation(layer, n_qubits))
        for wire, (alpha, beta, gamma) in enumerate(w2):
            qml.RY(alpha, wires=wire)
            qml.RX(beta, wires=wire)
            qml.RZ(gamma, wires=wire)

        # S(x)
        encoding(data, n_qubits, scaling)

        # W(𝜃)
        for layer in range(trainable_layers):
            W(w3[layer], cyclic_permutation(layer, n_qubits))
        # final RY-RX-RZ to measure qubit
        qml.RY(w[-3], wires=0)
        qml.RX(w[-2], wires=0)
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "strongly_ry_rx_rz"


def all_to_all_parallel(x, n_qubits, trainable_layers, scaling, random_key1, random_key2):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    n_rot_params = 3
    W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, n_qubits, n_rot_params), minval=0, maxval=2 * jnp.pi)
    final_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits, n_rot_params), minval=0, maxval=2 * jnp.pi)
    weights = {"W": W_weights, "final": final_weights}

    def S(x):
        """encoding block"""
        for w in range(n_qubits):
            qml.RX((scaling ** w) * x, wires=w)

    def W(theta):
        """train block"""
        for i in range(n_qubits):
            qml.Rot(theta[i][0], theta[i][1], theta[i][2], wires=i)

        for j in range(n_qubits):
            for k in range(n_qubits):
                if j != k:
                    qml.CNOT(wires=[j, k])

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):
        # W(𝜃)
        for layer in weights["W"][0]:
            W(layer)
        for i in range(n_qubits):
            qml.Rot(weights["final"][0][i][0], weights["final"][0][i][1], weights["final"][0][i][2], wires=i)

        S(x)
        # W(𝜃)
        for layer in weights["W"][1]:
            W(layer)
        for i in range(n_qubits):
            qml.Rot(weights["final"][1][i][0], weights["final"][1][i][1], weights["final"][1][i][2], wires=i)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, "all_to_all_parallel"


def all_to_all_ry_rz_ry(x, n_qubits, trainable_layers, scaling, random_key1, random_key2):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    n_rot_params = 3
    W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, n_qubits, n_rot_params), minval=0, maxval=2 * jnp.pi)
    final_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits, n_rot_params), minval=0, maxval=2 * jnp.pi)
    weights = {"W": W_weights, "final": final_weights}

    def S(x):
        """encoding block"""
        for w in range(n_qubits):
            qml.RX((scaling ** w) * x, wires=w)

    def W(theta):
        """train block"""
        for i in range(n_qubits):
            qml.RY(theta[i][0], wires=i)
            qml.RZ(theta[i][1], wires=i)
            qml.RY(theta[i][2], wires=i)

        for j in range(n_qubits):
            for k in range(n_qubits):
                if j != k:
                    qml.CNOT(wires=[j, k])

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):
        # W(𝜃)
        for layer in weights["W"][0]:
            W(layer)
        for i in range(n_qubits):
            qml.RY(weights["final"][0][i][0], wires=i)
            qml.RZ(weights["final"][0][i][1], wires=i)
            qml.RY(weights["final"][0][i][2], wires=i)

        S(x)
        # W(𝜃)
        for layer in weights["W"][1]:
            W(layer)
        for i in range(n_qubits):
            qml.RY(weights["final"][1][i][0], wires=i)
            qml.RZ(weights["final"][1][i][1], wires=i)
            qml.RY(weights["final"][1][i][2], wires=i)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, "all_to_all_ry_rz_ry"


def all_to_all_ry_rx_ry(x, n_qubits, trainable_layers, scaling, random_key1, random_key2):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    n_rot_params = 3
    W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, n_qubits, n_rot_params), minval=0, maxval=2 * jnp.pi)
    final_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits, n_rot_params), minval=0, maxval=2 * jnp.pi)
    weights = {"W": W_weights, "final": final_weights}

    def S(x):
        """encoding block"""
        for w in range(n_qubits):
            qml.RX((scaling ** w) * x, wires=w)

    def W(theta):
        """train block"""
        for i in range(n_qubits):
            qml.RY(theta[i][0], wires=i)
            qml.RX(theta[i][1], wires=i)
            qml.RY(theta[i][2], wires=i)

        for j in range(n_qubits):
            for k in range(n_qubits):
                if j != k:
                    qml.CNOT(wires=[j, k])

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):
        # W(𝜃)
        for layer in weights["W"][0]:
            W(layer)
        for i in range(n_qubits):
            qml.RY(weights["final"][0][i][0], wires=i)
            qml.RX(weights["final"][0][i][1], wires=i)
            qml.RY(weights["final"][0][i][2], wires=i)

        S(x)
        # W(𝜃)
        for layer in weights["W"][1]:
            W(layer)
        for i in range(n_qubits):
            qml.RY(weights["final"][1][i][0], wires=i)
            qml.RX(weights["final"][1][i][1], wires=i)
            qml.RY(weights["final"][1][i][2], wires=i)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, "all_to_all_ry_rx_ry"


def all_to_all_rz_rx_rz(x, n_qubits, trainable_layers, scaling, random_key1, random_key2):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    n_rot_params = 3
    W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, n_qubits, n_rot_params), minval=0, maxval=2 * jnp.pi)
    final_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits, n_rot_params), minval=0, maxval=2 * jnp.pi)
    weights = {"W": W_weights, "final": final_weights}

    def S(x):
        """encoding block"""
        for w in range(n_qubits):
            qml.RX((scaling ** w) * x, wires=w)

    def W(theta):
        """train block"""
        for i in range(n_qubits):
            qml.RZ(theta[i][0], wires=i)
            qml.RX(theta[i][1], wires=i)
            qml.RZ(theta[i][2], wires=i)

        for j in range(n_qubits):
            for k in range(n_qubits):
                if j != k:
                    qml.CNOT(wires=[j, k])

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):
        # W(𝜃)
        for layer in weights["W"][0]:
            W(layer)
        for i in range(n_qubits):
            qml.RZ(weights["final"][0][i][0], wires=i)
            qml.RX(weights["final"][0][i][1], wires=i)
            qml.RZ(weights["final"][0][i][2], wires=i)

        S(x)
        # W(𝜃)
        for layer in weights["W"][1]:
            W(layer)
        for i in range(n_qubits):
            qml.RZ(weights["final"][1][i][0], wires=i)
            qml.RX(weights["final"][1][i][1], wires=i)
            qml.RZ(weights["final"][1][i][2], wires=i)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, "all_to_all_rz_rx_rz"


def all_to_all_ry_rx_rz(x, n_qubits, trainable_layers, scaling, random_key1, random_key2):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    n_rot_params = 3
    W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, n_qubits, n_rot_params), minval=0, maxval=2 * jnp.pi)
    final_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits, n_rot_params), minval=0, maxval=2 * jnp.pi)
    weights = {"W": W_weights, "final": final_weights}

    def S(x):
        """encoding block"""
        for w in range(n_qubits):
            qml.RX((scaling ** w) * x, wires=w)

    def W(theta):
        """train block"""
        for i in range(n_qubits):
            qml.RY(theta[i][0], wires=i)
            qml.RX(theta[i][1], wires=i)
            qml.RZ(theta[i][2], wires=i)

        for j in range(n_qubits):
            for k in range(n_qubits):
                if j != k:
                    qml.CNOT(wires=[j, k])

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):
        # W(𝜃)
        for layer in weights["W"][0]:
            W(layer)
        for i in range(n_qubits):
            qml.RY(weights["final"][0][i][0], wires=i)
            qml.RX(weights["final"][0][i][1], wires=i)
            qml.RZ(weights["final"][0][i][2], wires=i)

        S(x)
        # W(𝜃)
        for layer in weights["W"][1]:
            W(layer)
        for i in range(n_qubits):
            qml.RY(weights["final"][1][i][0], wires=i)
            qml.RX(weights["final"][1][i][1], wires=i)
            qml.RZ(weights["final"][1][i][2], wires=i)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, "all_to_all_ry_rx_rz"


def strongly_parallel2(x, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    n_rot_params = 3
    weights = 2 * jnp.pi * jax.random.uniform(random_key, shape=(num_wx, trainable_layers, n_qubits, n_rot_params))

    def S(x):
        """encoding block"""
        for w in range(n_qubits):
            qml.RX((scaling**w) * x, wires=w)

    def W(theta):
        """train block"""
        StronglyEntanglingLayers(theta, wires=range(n_qubits))

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):
        W(weights[0])
        S(x)
        W(weights[1])

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, "strongly_parallel"

