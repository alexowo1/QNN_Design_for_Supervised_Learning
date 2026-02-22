import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
import jax
import jax.numpy as jnp


def cyclic_permutation(layer, n_qubits):
    # cyclic permutation shifted by layer
    return [(i + (layer % (n_qubits-1)) + 1) % n_qubits for i in range(n_qubits)]


def serial(x, degree, trainable_blocks, scaling, random_key):
    dev = qml.device("default.qubit", wires=2)
    n_rot_params = 3
    weights = 2 * jnp.pi * jax.random.uniform(random_key, shape=((degree + 1), trainable_blocks, n_rot_params))

    def S(x, layer):
        """Data-encoding circuit block"""
        qml.RX((scaling**layer) * x, wires=0)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.Rot(theta[block][0], theta[block][1], theta[block][2], wires=0)
            qml.CNOT(wires=[1, 0])

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):
        idx = 0
        for layer in weights[:-1]:
            W(layer)
            S(x, idx)
            idx += 1

        # (L+1)'th unitary
        W(weights[-1])

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, "serial"

def serial_2qubits(x, degree, trainable_blocks, scaling, random_key):
    dev = qml.device("default.qubit", wires=2)
    n_rot_params = 3
    n_qubits = 2
    weights = 2 * jnp.pi * jax.random.uniform(random_key, shape=((degree + 1), trainable_blocks, n_qubits, n_rot_params))

    def S(x, layer):
        """Data-encoding circuit block"""
        qml.RX((scaling**layer) * x, wires=0)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.Rot(theta[block][0][0], theta[block][0][1], theta[block][0][2], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.Rot(theta[block][1][0], theta[block][1][1], theta[block][1][2], wires=1)
            qml.CNOT(wires=[1, 0])

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):
        idx = 0
        for layer in weights[:-1]:
            W(layer)
            S(x, idx)
            idx += 1

        # (L+1)'th unitary
        W(weights[-1])

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, "serial_2qubits"


def strongly_parallel(x, n_qubits, trainable_layers, scaling, random_key):
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
        # for theta in weights[:-1]:
        #     W(theta)
        #     S(x)
        #
        # # (L+1)'th unitary
        # W(weights[-1])

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, "strongly_parallel"


def all_to_all_parallel(x, n_qubits, trainable_layers, scaling, random_key):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    n_rot_params = 3
    weights = 2 * jnp.pi * jax.random.uniform(random_key, shape=(num_wx, trainable_layers, n_qubits, n_rot_params))

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
        for layer in weights[0]:
            W(layer)
        S(x)
        for layer in weights[1]:
            W(layer)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, "all_to_all_parallel"


def all_to_all_crz(x, n_qubits, trainable_layers, scaling, random_key1, random_key2):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    layer_size = n_qubits + n_qubits * (n_qubits - 1)
    W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    weights = {"W": W_weights, "final": final_ry_weights}

    def S(x):
        """encoding block"""
        for w in range(n_qubits):
            qml.RX((scaling**w) * x, wires=w)

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
        #idx = n_qubits + 1
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    qml.CRZ(theta[idx], wires=[i, j])
                    idx += 1

        # RY to each qubit
        # for i in range(n_qubits):
        #     qml.RY(theta[idx], wires=i)
        #     idx += 1

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):

        for layer in range(trainable_layers):
            W(weights["W"][0][layer])

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(weights["final"][0][i], wires=i)

        S(x)

        for layer in range(trainable_layers):
            W(weights["W"][1][layer])

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(weights["final"][1][i], wires=i)

        return qml.expval(qml.PauliZ(0))

    # @qml.qnode(dev, interface="jax")
    # def quantum_model(weights, x):
    #
    #     W(weights[0])
    #     S(x)
    #     W(weights[1])
    #
    #     return qml.expval(qml.PauliZ(4))

    return model, weights, "all_to_all_crz"


def all_to_all_rzz(x, n_qubits, trainable_layers, scaling, random_key1, random_key2):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    layer_size = n_qubits + (n_qubits * (n_qubits - 1))//2
    W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    weights = {"W": W_weights, "final": final_ry_weights}

    def S(x):
        """encoding block"""
        for w in range(n_qubits):
            qml.RX((scaling**w) * x, wires=w)

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

        # RY to each qubit
        # for i in range(n_qubits):
        #     qml.RY(theta[idx], wires=i)
        #     idx += 1

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):

        for layer in range(trainable_layers):
            W(weights["W"][0][layer])

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(weights["final"][0][i], wires=i)

        S(x)

        for layer in range(trainable_layers):
            W(weights["W"][1][layer])

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(weights["final"][1][i], wires=i)

        return qml.expval(qml.PauliZ(0))

    # @qml.qnode(dev, interface="jax")
    # def quantum_model(weights, x):
    #
    #     W(weights[0])
    #     S(x)
    #     W(weights[1])
    #
    #     return qml.expval(qml.PauliZ(4))

    return model, weights, "all_to_all_rzz"


def strongly_crz(x, n_qubits, trainable_layers, scaling, random_key1, random_key2):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    layer_size = 2 * n_qubits
    W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    weights = {"W": W_weights, "final": final_ry_weights}

    def S(x):
        """encoding block"""
        for w in range(n_qubits):
            qml.RX((scaling**w) * x, wires=w)

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

        # RY to each qubit
        # for i in range(n_qubits):
        #     qml.RY(theta[idx], wires=i)
        #     idx += 1

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):

        for layer in range(trainable_layers):
            W(weights["W"][0][layer], cyclic_permutation(layer, n_qubits))

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(weights["final"][0][i], wires=i)

        S(x)

        for layer in range(trainable_layers):
            W(weights["W"][1][layer], cyclic_permutation(layer, n_qubits))

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(weights["final"][1][i], wires=i)

        return qml.expval(qml.PauliZ(0))

    # @qml.qnode(dev, interface="jax")
    # def quantum_model(weights, x):
    #
    #     W(weights[0])
    #     S(x)
    #     W(weights[1])
    #
    #     return qml.expval(qml.PauliZ(4))

    return model, weights, "strongly_crz"


def strongly_rzz(x, n_qubits, trainable_layers, scaling, random_key1, random_key2):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    layer_size = 2 * n_qubits
    W_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, layer_size), minval=0, maxval=2 * jnp.pi)
    final_ry_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits), minval=0, maxval=2 * jnp.pi)
    weights = {"W": W_weights, "final": final_ry_weights}

    def S(x):
        """encoding block"""
        for w in range(n_qubits):
            qml.RX((scaling**w) * x, wires=w)

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

        # RY to each qubit
        # for i in range(n_qubits):
        #     qml.RY(theta[idx], wires=i)
        #     idx += 1

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):

        for layer in range(trainable_layers):
            W(weights["W"][0][layer], cyclic_permutation(layer, n_qubits))

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(weights["final"][0][i], wires=i)

        S(x)

        for layer in range(trainable_layers):
            W(weights["W"][1][layer], cyclic_permutation(layer, n_qubits))

        # final RY to each qubit
        for i in range(n_qubits):
            qml.RY(weights["final"][1][i], wires=i)

        return qml.expval(qml.PauliZ(0))

    # @qml.qnode(dev, interface="jax")
    # def quantum_model(weights, x):
    #
    #     W(weights[0])
    #     S(x)
    #     W(weights[1])
    #
    #     return qml.expval(qml.PauliZ(4))

    return model, weights, "strongly_rzz"


def basic_mixed(x, n_qubits, trainable_layers, scaling, random_key1, random_key2):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    num_rot_params = 3
    outer_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, n_qubits, num_rot_params), minval=0, maxval=2 * jnp.pi)
    inner_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits, num_rot_params), minval=0, maxval=2 * jnp.pi)
    weights = {"W": outer_weights, "final": inner_weights}

    def inner(x, theta):
        """Rot-RX-Rot with entangling between each qubit"""
        for i in range(n_qubits-1):
            qml.Rot(theta[0][i][0], theta[0][i][1], theta[0][i][2], wires=i)
            qml.RX(scaling * x, wires=i)
            qml.Rot(theta[1][i][0], theta[1][i][1], theta[1][i][2], wires=i)
            qml.CNOT(wires=[i, i+1])

        qml.Rot(theta[0][-1][0], theta[0][-1][1], theta[0][-1][2], wires=n_qubits-1)
        qml.RX(scaling * x, wires=n_qubits-1)
        qml.Rot(theta[1][-1][0], theta[1][-1][1], theta[1][-1][2], wires=n_qubits-1)
        for i in range(n_qubits-1):
            qml.CNOT(wires=[n_qubits-1-i, n_qubits-2-i])

    def outer(theta):
        StronglyEntanglingLayers(theta, wires=range(n_qubits))

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):
        outer(weights["W"][0])
        inner(x, weights["final"])
        outer(weights["W"][1])

        return qml.expval(qml.PauliZ(0))

    return model, weights, "basic_mixed"


def one_to_all_mixed(x, n_qubits, trainable_layers, scaling, random_key1, random_key2):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    num_rot_params = 3
    outer_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, n_qubits, num_rot_params), minval=0, maxval=2 * jnp.pi)
    inner_weights = jax.random.uniform(random_key2, shape=(num_wx, n_qubits, num_rot_params), minval=0, maxval=2 * jnp.pi)
    weights = {"W": outer_weights, "final": inner_weights}

    def inner(x, theta):
        """Rot-RX-Rot on each qubit with entangling to all other qubits"""
        for i in range(n_qubits):
            qml.Rot(theta[0][i][0], theta[0][i][1], theta[0][i][2], wires=i)
            qml.RX(scaling * x, wires=i)
            qml.Rot(theta[1][i][0], theta[1][i][1], theta[1][i][2], wires=i)
            for j in range(n_qubits):
                if i != j:
                    qml.CNOT(wires=[i, j])

    def outer(theta):
        StronglyEntanglingLayers(theta, wires=range(n_qubits))

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):
        outer(weights["W"][0])
        inner(x, weights["final"])
        outer(weights["W"][1])

        return qml.expval(qml.PauliZ(n_qubits-1))

    return model, weights, "one_to_all_mixed"


def all_to_one_mixed(x, n_qubits, trainable_layers, scaling, random_key1, random_key2):
    dev = qml.device("default.qubit", wires=n_qubits)
    num_wx = 2
    num_rot_params = 3
    outer_weights = jax.random.uniform(random_key1, shape=(num_wx, trainable_layers, n_qubits, num_rot_params), minval=0, maxval=2 * jnp.pi)
    inner_weights = jax.random.uniform(random_key2, shape=(2*num_wx, n_qubits, num_rot_params), minval=0, maxval=2 * jnp.pi)
    weights = {"W": outer_weights, "final": inner_weights}

    def inner(x, theta):
        """Rot-RX-Rot on each qubit with entangling to all other qubits"""
        for i in range(n_qubits):
            qml.Rot(theta[0][i][0], theta[0][i][1], theta[0][i][2], wires=i)
        for i in range(n_qubits):
            for j in range(n_qubits):
                if i != j:
                    qml.CNOT(wires=[j, i])
            qml.Rot(theta[1][i][0], theta[1][i][1], theta[1][i][2], wires=i)
            qml.RX(scaling * x, wires=i)
            qml.Rot(theta[2][i][0], theta[2][i][1], theta[2][i][2], wires=i)

        for i in range(n_qubits-1):
            qml.CNOT(wires=[n_qubits-1-i, n_qubits-2-i])
            qml.Rot(theta[3][n_qubits-1-i][0], theta[3][n_qubits-1-i][1], theta[3][n_qubits-1-i][2], wires=n_qubits-1-i)
        qml.Rot(theta[3][0][0], theta[3][0][1], theta[3][0][2], wires=0)

    def outer(theta):
        StronglyEntanglingLayers(theta, wires=range(n_qubits))

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):
        outer(weights["W"][0])
        inner(x, weights["final"])
        outer(weights["W"][1])

        return qml.expval(qml.PauliZ(n_qubits-1))

    return model, weights, "all_to_one_mixed"


def exponential_serial(x, degree, trainable_blocks, random_key, scaling=3):
    dev = qml.device("default.qubit", wires=2)
    n_rot_params = 3
    num_encodings = jnp.ceil(jnp.log(degree) / jnp.log(scaling)).astype(int)
    weights = 2 * jnp.pi * jax.random.uniform(random_key, shape=(num_encodings + 1, trainable_blocks, n_rot_params))

    def S(x, layer):
        """Data-encoding circuit block"""
        qml.RX((scaling**layer) * x, wires=0)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.Rot(theta[block][0], theta[block][1], theta[block][2], wires=0)
            qml.CNOT(wires=[0, 1])

    @qml.qnode(dev, interface="jax")
    def model(weights=weights, x=x):
        idx = 0
        for layer in weights[:-1]:
            W(layer)
            S(x, idx)
            idx += 1

        # (L+1)'th unitary
        W(weights[-1])

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, "exp_serial"


