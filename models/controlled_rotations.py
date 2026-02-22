import pennylane as qml
import jax
import jax.numpy as jnp
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
            case qml.CRX:
                rotations_string.append("crx")
            case qml.CRY:
                rotations_string.append("cry")
            case qml.CRZ:
                rotations_string.append("crz")
            case qml.IsingXX:
                rotations_string.append("rxx")
            case qml.IsingYY:
                rotations_string.append("ryy")
            case qml.IsingZZ:
                rotations_string.append("rzz")

    return rotations_string


def build_ansatz_fragments(rotations, n_qubits, theta, entangling=None, permutation=None):
    idx = 0
    for rotation in rotations:
        if rotation in (qml.RX, qml.RY, qml.RZ):
            for i in range(n_qubits):
                rotation(theta[i], wires=i)
                idx += 1
        else:
            match entangling:
                case "basic":
                    for i in range(n_qubits):
                        j = (i + 1) % n_qubits
                        rotation(theta[n_qubits + i], wires=[i, j])
                case "strongly":
                    for i, j in zip(range(n_qubits), permutation):
                        rotation(theta[n_qubits + i], wires=[i, j])
                case "permutated":
                    idx_list = list(zip(range(n_qubits), permutation))
                    sort_cyclic_permutation(idx_list)
                    for i, j in idx_list:
                        rotation(theta[n_qubits + i], wires=[i, j])
                case "all-to-all":
                    for i in range(n_qubits):
                        for j in range(n_qubits):
                            if i != j:
                                rotation(theta[idx], wires=[i, j])
                                idx += 1


def basic_rotational_entangling(encoding, n_qubits, trainable_layers, scaling, random_key, entangling, rotations):
    dev = qml.device(dev_name, wires=n_qubits)
    num_rotations = len(rotations)
    rotations_string = toString_operations(rotations)
    name = f"{entangling}_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[0]}" if num_rotations < 3 else f"{entangling}_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[2]}"
    layer_size = 2 * n_qubits if num_rotations < 3 else 3 * n_qubits
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + n_qubits
    total_size = trainblock_size + w_theta_size + 1 if num_rotations < 3 else 2 * w_theta_size - n_qubits + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """train block"""
        build_ansatz_fragments(rotations, n_qubits, theta, entangling)

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        if num_rotations < 3:
            w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
            w2 = w[w_theta_size:trainblock_size]
            w3 = w[trainblock_size:-1].reshape(trainable_layers, layer_size)
            # W(𝜃)
            for layer in w1:
                W(layer)
            # final rotation to each qubit
            # for i in range(n_qubits):
            #     rotations[0](w2[i], wires=i)
            build_ansatz_fragments([rotations[0]], n_qubits, w2)

            # S(x)
            encoding(data, n_qubits, scaling)

            # W(𝜃)
            for layer in w3:
                W(layer)
            # final rotation to measure qubit
            rotations[0](w[-1], wires=0)
        else:
            w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
            w2 = w[w_theta_size:-layer_size+n_qubits-1].reshape(trainable_layers-1, layer_size)
            w3 = w[-layer_size+n_qubits-1:-1]
            # W(𝜃)
            for layer in w1:
                W(layer)

            # S(x)
            encoding(data, n_qubits, scaling)

            # W(𝜃)
            for layer in w2:
                W(layer)
            # making sure only the measure qubit gets the last rotation
            build_ansatz_fragments((rotations[0], rotations[1]), n_qubits, w3, entangling)
            # final rotation to measure-qubit
            rotations[2](w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, name


def strongly_rotational_entangling(encoding, n_qubits, trainable_layers, scaling, random_key, entangling, rotations):
    dev = qml.device(dev_name, wires=n_qubits)
    measure_wire = 0 if entangling in "permutated" else trainable_layers - 1
    num_rotations = len(rotations)
    rotations_string = toString_operations(rotations)
    name = f"{entangling}_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[0]}" if num_rotations < 3 else f"{entangling}_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[2]}"
    layer_size = 2 * n_qubits if num_rotations < 3 else 3 * n_qubits
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + n_qubits
    total_size = trainblock_size + w_theta_size + 1 if num_rotations < 3 else 2 * w_theta_size - n_qubits + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta, permutation):
        """train block"""
        build_ansatz_fragments(rotations, n_qubits, theta, entangling, permutation)

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        if num_rotations < 3:
            w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
            w2 = w[w_theta_size:trainblock_size]
            w3 = w[trainblock_size:-1].reshape(trainable_layers, layer_size)
            # W(𝜃)
            for layer in range(trainable_layers):
                W(w1[layer], cyclic_permutation(layer, n_qubits))
            # final rotation to each qubit
            # for i in range(n_qubits):
            #     rotations[0](w2[i], wires=i)
            build_ansatz_fragments([rotations[0]], n_qubits, w2)

            # S(x)
            encoding(data, n_qubits, scaling)

            # W(𝜃)
            for layer in range(trainable_layers):
                W(w3[layer], cyclic_permutation(layer, n_qubits))
            # final rotation to measure qubit
            rotations[0](w[-1], wires=measure_wire)
        else:
            w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
            w2 = w[w_theta_size:-layer_size+n_qubits-1].reshape(trainable_layers-1, layer_size)
            w3 = w[-layer_size+n_qubits-1:-1]
            # W(𝜃)
            for layer in range(trainable_layers):
                W(w1[layer], cyclic_permutation(layer, n_qubits))

            # S(x)
            encoding(data, n_qubits, scaling)

            # W(𝜃)
            for layer in range(trainable_layers-1):
                W(w2[layer], cyclic_permutation(layer, n_qubits))
            # making sure only the measure qubit gets the last rotation
            build_ansatz_fragments((rotations[0], rotations[1]), n_qubits, w3, entangling, cyclic_permutation(trainable_layers, n_qubits))
            # final rotation to measure-qubit
            rotations[2](w[-1], wires=measure_wire)

        return qml.expval(qml.PauliZ(measure_wire))

    return model, weights, total_size, name


def all_to_all_rotational_entangling(encoding, n_qubits, trainable_layers, scaling, random_key, entangling, rotations):
    dev = qml.device(dev_name, wires=n_qubits)
    num_rotations = len(rotations)
    rotations_string = toString_operations(rotations)
    name = f"{entangling}_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[0]}" if num_rotations < 3 else f"{entangling}_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[2]}"
    layer_size = n_qubits**2 if num_rotations < 3 else n_qubits * (n_qubits + 1)
    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + n_qubits
    total_size = trainblock_size + w_theta_size + 1 if num_rotations < 3 else 2 * w_theta_size - n_qubits + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2 * jnp.pi)

    def W(theta):
        """train block"""
        build_ansatz_fragments(rotations, n_qubits, theta, entangling)

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        if num_rotations < 3:
            w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
            w2 = w[w_theta_size:trainblock_size]
            w3 = w[trainblock_size:-1].reshape(trainable_layers, layer_size)
            # W(𝜃)
            for layer in w1:
                W(layer)
            # final rotation to each qubit
            # for i in range(n_qubits):
            #     rotations[0](w2[i], wires=i)
            build_ansatz_fragments([rotations[0]], n_qubits, w2)

            # S(x)
            encoding(data, n_qubits, scaling)

            # W(𝜃)
            for layer in w3:
                W(layer)
            # final rotation to measure qubit
            rotations[0](w[-1], wires=0)
        else:
            w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
            w2 = w[w_theta_size:-layer_size+n_qubits-1].reshape(trainable_layers-1, layer_size)
            w3 = w[-layer_size+n_qubits-1:-1]
            # W(𝜃)
            for layer in w1:
                W(layer)

            # S(x)
            encoding(data, n_qubits, scaling)

            # W(𝜃)
            for layer in w2:
                W(layer)
            # making sure only the measure qubit gets the last rotation
            build_ansatz_fragments((rotations[0], rotations[1]), n_qubits, w3, entangling)
            # final rotation to measure-qubit
            rotations[2](w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, name















"""old redundant models"""

def basic_crz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 2 * n_qubits
    # W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + n_qubits
    total_size = trainblock_size + w_theta_size + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """
        train block
        RY with CRZ basic entangling layers
        """
        # idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[i], wires=i)
            # idx += 1

        # Ring of CRZ
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.CRZ(theta[n_qubits + i], wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:trainblock_size]
        w3 = w[trainblock_size:-1].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer])

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(w2[i], wires=i)

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w3[layer])

        # final RY to measure qubit
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "basic_crz"


def basic_cry(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 2 * n_qubits
    # W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + n_qubits
    total_size = trainblock_size + w_theta_size + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """
        train block
        RZ with CRY basic entangling layers
        """
        # idx = 0
        # RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(theta[i], wires=i)
            # idx += 1

        # Ring of CRY
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.CRY(theta[n_qubits + i], wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:trainblock_size]
        w3 = w[trainblock_size:-1].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer])

        # final RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(w2[i], wires=i)

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w3[layer])

        # final RZ to measure qubit
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "basic_cry"


def basic_crx_ry(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 2 * n_qubits
    # W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + n_qubits
    total_size = trainblock_size + w_theta_size + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """
        train block
        RY with CRX basic entangling layers
        """
        # idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[i], wires=i)
            # idx += 1

        # Ring of CRX
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.CRX(theta[n_qubits + i], wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:trainblock_size]
        w3 = w[trainblock_size:-1].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer])

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(w2[i], wires=i)

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w3[layer])

        # final RY to measure qubit
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "basic_crx_ry"


def basic_crx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 2 * n_qubits
    # W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + n_qubits
    total_size = trainblock_size + w_theta_size + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """
        train block
        RZ with CRX basic entangling layers
        """
        # idx = 0
        # RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(theta[i], wires=i)
            # idx += 1

        # Ring of CRX
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.CRX(theta[n_qubits + i], wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:trainblock_size]
        w3 = w[trainblock_size:-1].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer])

        # final RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(w2[i], wires=i)

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w3[layer])

        # final RZ to measure qubit
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "basic_crx_rz"


def basic_ry_crx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 3 * n_qubits
    # weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    total_size = 2 * w_theta_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """
        train block
        RY -> CRX basic entangling layers -> RZ
        """
        idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[idx], wires=i)
            idx += 1

        # Ring of CRX
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.CRX(theta[idx], wires=[i, j])
            idx += 1

        for i in range(n_qubits):
            qml.RZ(theta[idx], wires=i)
            idx += 1

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer])

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w2[layer])

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "basic_ry_crx_rz"


def strongly_crz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 2 * n_qubits
    # W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + n_qubits
    total_size = trainblock_size + w_theta_size + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta, permutation):
        """
        train block
        RY with CRZ strongly entangling layers
        """
        # idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[i], wires=i)
            # idx += 1

        # Ring of CRZ (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.CRZ(theta[n_qubits + i], wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:trainblock_size]
        w3 = w[trainblock_size:-1].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer], cyclic_permutation(layer, n_qubits))

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(w2[i], wires=i)

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w3[layer], cyclic_permutation(layer, n_qubits))

        # final RY to measure qubit
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "strongly_crz"


def strongly_cry(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 2 * n_qubits
    # W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + n_qubits
    total_size = trainblock_size + w_theta_size + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta, permutation):
        """
        train block
        RZ with CRY strongly entangling layers
        """
        # idx = 0
        # RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(theta[i], wires=i)
            # idx += 1

        # Ring of CRY (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.CRY(theta[n_qubits + i], wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:trainblock_size]
        w3 = w[trainblock_size:-1].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer], cyclic_permutation(layer, n_qubits))

        # final RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(w2[i], wires=i)

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w3[layer], cyclic_permutation(layer, n_qubits))

        # final RZ to measure qubit
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "strongly_cry"


def strongly_crx_ry(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 2 * n_qubits
    # W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + n_qubits
    total_size = trainblock_size + w_theta_size + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta, permutation):
        """
        train block
        RY with CRX strongly entangling layers
        """
        # idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[i], wires=i)
            # idx += 1

        # Ring of CRX (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.CRX(theta[n_qubits + i], wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:trainblock_size]
        w3 = w[trainblock_size:-1].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer], cyclic_permutation(layer, n_qubits))

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(w2[i], wires=i)

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w3[layer], cyclic_permutation(layer, n_qubits))

        # final RY to measure qubit
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "strongly_crx_ry"


def strongly_crx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 2 * n_qubits
    # W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    trainblock_size = w_theta_size + n_qubits
    total_size = trainblock_size + w_theta_size + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta, permutation):
        """
        train block
        RZ with CRX strongly entangling layers
        """
        # idx = 0
        # RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(theta[i], wires=i)
            # idx += 1

        # Ring of CRX (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.CRX(theta[n_qubits + i], wires=[i, j])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:trainblock_size]
        w3 = w[trainblock_size:-1].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer], cyclic_permutation(layer, n_qubits))

        # final RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(w2[i], wires=i)

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w3[layer], cyclic_permutation(layer, n_qubits))

        # final RZ to measure qubit
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "strongly_crx_rz"


def strongly_ry_crx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 3 * n_qubits
    # weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    total_size = 2 * w_theta_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta, permutation):
        """
        train block
        RY -> CRX strongly entangling layers -> RZ
        """
        idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[idx], wires=i)
            idx += 1

        # Ring of CRX (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            qml.CRX(theta[idx], wires=[i, j])
            idx += 1

        for i in range(n_qubits):
            qml.RZ(theta[idx], wires=i)
            idx += 1

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer], cyclic_permutation(layer, n_qubits))

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w2[layer], cyclic_permutation(layer, n_qubits))

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "strongly_ry_crx_rz"


def all_to_all_crz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = n_qubits + n_qubits * (n_qubits - 1)
    # W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    train_block_size = w_theta_size + n_qubits
    total_size = train_block_size + w_theta_size + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2 * jnp.pi)

    def W(theta):
        """
        train block
        RY with CRZ all-to-all
        """
        idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[idx], wires=i)
            idx += 1

        # CRZ to all qubit pairs (i ≠ j)
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    qml.CRZ(theta[idx], wires=[i, j])
                    idx += 1

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:train_block_size]
        w3 = w[train_block_size:-1].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer])

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(w2[i], wires=i)

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w3[layer])

        # final RY to measure qubit
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "all_to_all_crz"


def all_to_all_cry(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = n_qubits + n_qubits * (n_qubits - 1)
    # W_weights = jax.random.uniform(random_key1, shape=(num_w_theta, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_w_theta, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    train_block_size = w_theta_size + n_qubits
    total_size = train_block_size + w_theta_size + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2 * jnp.pi)

    def W(theta):
        """
        train block
        RZ with CRY all-to-all
        """
        idx = 0
        # RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(theta[idx], wires=i)
            idx += 1

        # CRY to all qubit pairs (i ≠ j)
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    qml.CRY(theta[idx], wires=[i, j])
                    idx += 1

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:train_block_size]
        w3 = w[train_block_size:-1].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer])

        # final RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(w2[i], wires=i)

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w3[layer])

        # final RZ to measure qubit
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "all_to_all_cry"


def all_to_all_crx_ry(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = n_qubits + n_qubits * (n_qubits - 1)
    # W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    train_block_size = w_theta_size + n_qubits
    total_size = train_block_size + w_theta_size + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2 * jnp.pi)

    def W(theta):
        """
        train block
        RY with CRX all-to-all
        """
        idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[idx], wires=i)
            idx += 1

        # CRX to all qubit pairs (i ≠ j)
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    qml.CRX(theta[idx], wires=[i, j])
                    idx += 1

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:train_block_size]
        w3 = w[train_block_size:-1].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer])

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(w2[i], wires=i)

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w3[layer])

        # final RY to measure qubit
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "all_to_all_crx_ry"


def all_to_all_crx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = n_qubits + n_qubits * (n_qubits - 1)
    # W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    train_block_size = w_theta_size + n_qubits
    total_size = train_block_size + w_theta_size + 1
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2 * jnp.pi)

    def W(theta):
        """
        train block
        RZ with CRX all-to-all
        """
        idx = 0
        # RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(theta[idx], wires=i)
            idx += 1

        # CRX to all qubit pairs (i ≠ j)
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    qml.CRX(theta[idx], wires=[i, j])
                    idx += 1

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:train_block_size]
        w3 = w[train_block_size:-1].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer])

        # final RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(w2[i], wires=i)

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w3[layer])

        # final RZ to each qubit
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "all_to_all_crx_rz"


def all_to_all_ry_crx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 2 * n_qubits + n_qubits * (n_qubits - 1)
    # weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_ry_weights}

    w_theta_size = trainable_layers * layer_size
    total_size = 2 * w_theta_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """
        train block
        RY -> CRX all-to-all -> RZ
        """
        idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[idx], wires=i)
            idx += 1

        # CRX to all qubit pairs (i ≠ j)
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    qml.CRX(theta[idx], wires=[i, j])
                    idx += 1

        # RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(theta[idx], wires=i)
            idx += 1

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        w1 = w[:w_theta_size].reshape(trainable_layers, layer_size)
        w2 = w[w_theta_size:].reshape(trainable_layers, layer_size)

        for layer in range(trainable_layers):
            W(w1[layer])

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w2[layer])

        return qml.expval(qml.PauliZ(0))

    return model, weights, total_size, "all_to_all_ry_crx_rz"

