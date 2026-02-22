import pennylane as qml
import jax
import jax.numpy as jnp
from encodings import *


dev_name = "default.qubit"
diff_method = "best"


def cyclic_permutation(layer, n_qubits):
    # cyclic permutation shifted by layer
    return [(i + (layer % (n_qubits-1)) + 1) % n_qubits for i in range(n_qubits)]


def basic_rzz(encoding, n_qubits, trainable_layers, scaling, random_key):
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
        RY with RZZ basic entangling layers
        """
        # idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[i], wires=i)
            # idx += 1

        # Ring of RZZ
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.IsingZZ(theta[n_qubits + i], wires=[i, j])

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

    return model, weights, total_size, "basic_rzz"


def basic_ryy(encoding, n_qubits, trainable_layers, scaling, random_key):
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
        RZ with RYY basic entangling layers
        """
        # idx = 0
        # RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(theta[i], wires=i)
            # idx += 1

        # Ring of RYY
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.IsingYY(theta[n_qubits + i], wires=[i, j])

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

    return model, weights, total_size, "basic_ryy"


def basic_rxx_ry(encoding, n_qubits, trainable_layers, scaling, random_key):
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
        RY with RXX basic entangling layers
        """
        # idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[i], wires=i)
            # idx += 1

        # Ring of RXX
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.IsingXX(theta[n_qubits + i], wires=[i, j])

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

    return model, weights, total_size, "basic_rxx_ry"


def basic_rxx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
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
        RZ with RXX basic entangling layers
        """
        # idx = 0
        # RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(theta[i], wires=i)
            # idx += 1

        # Ring of RXX
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.IsingXX(theta[n_qubits + i], wires=[i, j])

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

    return model, weights, total_size, "basic_rxx_rz"


def basic_ry_rxx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 3 * n_qubits
    # weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)

    w_theta_size = trainable_layers * layer_size
    total_size = 2 * w_theta_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """
        train block
        RY and RZ with RZZ basic entangling layers
        """
        idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[idx], wires=i)
            idx += 1

        # Ring of RZZ
        for i in range(n_qubits):
            j = (i + 1) % n_qubits
            qml.IsingXX(theta[idx], wires=[i, j])
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

    return model, weights, total_size, "basic_ry_rxx_rz"


def strongly_rzz(encoding, n_qubits, trainable_layers, scaling, random_key):
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
        RY with RZZ strongly entangling layers
        """
        # idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[i], wires=i)
            # idx += 1

        # Ring of RZZ (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.IsingZZ(theta[n_qubits + i], wires=[i, j])

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
        qml.RY(w[-1], wires=n_qubits-1)

        return qml.expval(qml.PauliZ(n_qubits-1))

    return model, weights, total_size, "strongly_rzz"


def strongly_ryy(encoding, n_qubits, trainable_layers, scaling, random_key):
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
        RZ with RYY strongly entangling layers
        """
        # idx = 0
        # RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(theta[i], wires=i)
            # idx += 1

        # Ring of RYY (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.IsingYY(theta[n_qubits + i], wires=[i, j])

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
        qml.RZ(w[-1], wires=n_qubits-1)

        return qml.expval(qml.PauliZ(n_qubits-1))

    return model, weights, total_size, "strongly_ryy"


def strongly_rxx_ry(encoding, n_qubits, trainable_layers, scaling, random_key):
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
        RY with RXX strongly entangling layers
        """
        # idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[i], wires=i)
            # idx += 1

        # Ring of RXX (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.IsingXX(theta[n_qubits + i], wires=[i, j])

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
        qml.RY(w[-1], wires=n_qubits-1)

        return qml.expval(qml.PauliZ(n_qubits-1))

    return model, weights, total_size, "strongly_rxx_ry"


def strongly_rxx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
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
        RZ with RXX strongly entangling layers
        """
        # idx = 0
        # RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(theta[i], wires=i)
            # idx += 1

        # Ring of RXX (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.IsingXX(theta[n_qubits + i], wires=[i, j])

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
        qml.RZ(w[-1], wires=n_qubits-1)

        return qml.expval(qml.PauliZ(n_qubits-1))

    return model, weights, total_size, "strongly_rxx_rz"


def strongly_ry_rxx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 3 * n_qubits
    # weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)

    w_theta_size = trainable_layers * layer_size
    total_size = 2 * w_theta_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta, permutation):
        """
        train block
        RY with RZZ strongly entangling layers
        """
        idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[idx], wires=i)
            idx += 1

        # Ring of RZZ (i → i+1 mod n)
        for i, j in zip(range(n_qubits), permutation):
            # j = (i + 1) % n_qubits
            qml.IsingXX(theta[idx], wires=[i, j])
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
            W(w1[layer], cyclic_permutation(layer, n_qubits))

        encoding(data, n_qubits, scaling)

        for layer in range(trainable_layers):
            W(w2[layer], cyclic_permutation(layer, n_qubits))

        return qml.expval(qml.PauliZ(n_qubits-1))

    return model, weights, total_size, "strongly_ry_rxx_rz"


def all_to_all_rzz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = n_qubits + (n_qubits * (n_qubits - 1))//2
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
        RY with RZZ all-to-all
        """
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[i], wires=i)

        # RZZ to all qubit pairs (i ≠ j)
        idx = n_qubits
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                qml.IsingZZ(theta[idx], wires=[i, j])
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
        qml.RY(w[-1], wires=n_qubits-1)

        return qml.expval(qml.PauliZ(n_qubits-1))

    return model, weights, total_size, "all_to_all_rzz"


def all_to_all_ryy(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = n_qubits + (n_qubits * (n_qubits - 1))//2
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
        RZ with RYY all-to-all
        """
        # RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(theta[i], wires=i)

        # RYY to all qubit pairs (i ≠ j)
        idx = n_qubits
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                qml.IsingYY(theta[idx], wires=[i, j])
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
        qml.RZ(w[-1], wires=n_qubits-1)

        return qml.expval(qml.PauliZ(n_qubits-1))

    return model, weights, total_size, "all_to_all_ryy"


def all_to_all_rxx_ry(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = n_qubits + (n_qubits * (n_qubits - 1))//2
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
        RY with RXX all-to-all
        """
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[i], wires=i)

        # RXX to all qubit pairs (i ≠ j)
        idx = n_qubits
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                qml.IsingXX(theta[idx], wires=[i, j])
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
        qml.RY(w[-1], wires=n_qubits-1)

        return qml.expval(qml.PauliZ(n_qubits-1))

    return model, weights, total_size, "all_to_all_rxx_ry"


def all_to_all_rxx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = n_qubits + (n_qubits * (n_qubits - 1))//2
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
        RZ with RXX all-to-all
        """
        # RZ to each qubit
        for i in range(n_qubits):
            qml.RZ(theta[i], wires=i)

        # RXX to all qubit pairs (i ≠ j)
        idx = n_qubits
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                qml.IsingXX(theta[idx], wires=[i, j])
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
        qml.RZ(w[-1], wires=n_qubits-1)

        return qml.expval(qml.PauliZ(n_qubits-1))

    return model, weights, total_size, "all_to_all_rxx_rz"


def all_to_all_ry_rxx_rz(encoding, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device(dev_name, wires=n_qubits)
    layer_size = 2 * n_qubits + (n_qubits * (n_qubits - 1))//2
    # weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)

    w_theta_size = trainable_layers * layer_size
    total_size = 2 * w_theta_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """
        train block
        RY -> RXX all-to-all -> RZ
        """
        idx = 0
        # RY to each qubit
        for i in range(n_qubits):
            qml.RY(theta[idx], wires=i)
            idx += 1

        # RXX to all qubit pairs (i ≠ j)
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                qml.IsingXX(theta[idx], wires=[i, j])
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

        return qml.expval(qml.PauliZ(n_qubits-1))

    return model, weights, total_size, "all_to_all_ry_rxx_rz"

