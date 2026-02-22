import pennylane as qml
from pennylane.templates import StronglyEntanglingLayers
import jax
import jax.numpy as jnp
from encodings import *


dev_name = "default.qubit"
diff_method = "best"


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


def serial_single_qubit(encoding, degree, trainable_blocks, scaling, random_key, rotations):
    dev = qml.device(dev_name, wires=1)
    # for returning name of model for later evaluation
    rotations_string = toString_operations(rotations)
    # calculating length of weight array
    block_size = 2
    layer_size = trainable_blocks * block_size + 1
    total_size = (degree + 1) * layer_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2 * jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            rotations[0](theta[block][0], wires=0)
            rotations[1](theta[block][1], wires=0)
            if len(rotations) > 2:
                rotations[2](theta[block][1], wires=0)

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        if len(rotations) > 2:
            for layer in range(degree):
                w0 = w[layer * layer_size:(layer + 1) * layer_size]
                w1 = w0[:-1].reshape(trainable_blocks, block_size)
                w2 = w0[-1]
                W(w1)
                rotations[1](w2, wires=0)
                encoding(data, layer, scaling)

            # (L+1)'th unitary
            W(w[-layer_size:-1].reshape(trainable_blocks, block_size))
            rotations[1](w[-1], wires=0)
        else:
            for layer in range(degree):
                w0 = w[layer * layer_size:(layer + 1) * layer_size].reshape(trainable_blocks, block_size)
                W(w0)
                encoding(data, layer, scaling)

            # (L+1)'th unitary
            W(w[-layer_size:].reshape(trainable_blocks, block_size))

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, f"serial_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[2]}"


def serial_cnot_entangling(encoding, degree, trainable_blocks, scaling, random_key, rotations):
    dev = qml.device(dev_name, wires=2)
    # for returning name of model for later evaluation
    rotations_string = toString_operations(rotations)
    # calculating length of weight array
    n_rot_params = 3
    n_qubits = 2
    block_size = n_qubits * n_rot_params
    layer_size = trainable_blocks * block_size + n_rot_params
    total_size = (degree + 1) * layer_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            rotations[0](theta[block][0], 0)
            rotations[1](theta[block][1], 0)
            rotations[2](theta[block][2], 0)
            qml.CNOT(wires=[0, 1])
            rotations[0](theta[block][3], 1)
            rotations[1](theta[block][4], 1)
            rotations[2](theta[block][5], 1)
            qml.CNOT(wires=[1, 0])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size]
            w1 = w0[:-n_rot_params].reshape(trainable_blocks, block_size)
            w2 = w0[-n_rot_params:]
            W(w1)
            rotations[0](w2[0], 0)
            rotations[1](w2[1], 0)
            rotations[2](w2[2], 0)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:-n_rot_params].reshape(trainable_blocks, block_size))
        rotations[0](w[-3], 0)
        rotations[1](w[-2], 0)
        rotations[2](w[-1], 0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, f"serial_cnot_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[2]}"


def serial_rotational_entangling(encoding, degree, trainable_blocks, scaling, random_key, rotations):
    dev = qml.device(dev_name, wires=2)
    # for returning name of model for later evaluation
    rotations_string = toString_operations(rotations)
    # calculating length of weight array
    block_size = 4
    layer_size = block_size * trainable_blocks + 1
    total_size = (degree + 1) * layer_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            rotations[0](theta[block][0], wires=0)
            rotations[1](theta[block][1], wires=[0, 1])     # controlled or ising
            rotations[0](theta[block][2], wires=1)
            rotations[1](theta[block][3], wires=[1, 0])     # controlled or ising

    @qml.qnode(dev, interface="jax")
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size]
            w1 = w0[:-1].reshape(trainable_blocks, block_size)
            w2 = w0[-1]
            W(w1)
            qml.RY(w2, wires=0)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:-1].reshape(trainable_blocks, block_size))
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, f"serial_rot_ent_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[0]}"


def serial_multivariate(encoding, rotations, degree, trainable_blocks, scaling, random_key):
    dev = qml.device(dev_name, wires=2)
    # for returning name of model for later evaluation
    rotations_string = toString_operations(rotations)
    # calculating length of weight array
    n_rot_params = 3
    n_qubits = 2
    block_size = n_qubits * n_rot_params
    layer_size = trainable_blocks * block_size + (n_rot_params * 2)
    total_size = (degree + 1) * layer_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            rotations[0](theta[block][0], 0)
            rotations[1](theta[block][1], 0)
            rotations[2](theta[block][2], 0)
            qml.CNOT(wires=[0, 1])
            rotations[0](theta[block][3], 1)
            rotations[1](theta[block][4], 1)
            rotations[2](theta[block][5], 1)
            qml.CNOT(wires=[1, 0])

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size]
            w1 = w0[:-n_rot_params*2].reshape(trainable_blocks, block_size)
            w2 = w0[-n_rot_params*2:]
            W(w1)
            rotations[0](w2[0], 0)
            rotations[1](w2[1], 0)
            rotations[2](w2[2], 0)
            qml.CNOT(wires=[0, 1])
            rotations[0](w2[3], 1)
            rotations[1](w2[4], 1)
            rotations[2](w2[5], 1)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:-n_rot_params*2].reshape(trainable_blocks, block_size))
        rotations[0](w[-6], 0)
        rotations[1](w[-5], 0)
        rotations[2](w[-4], 0)
        qml.CNOT(wires=[0, 1])
        rotations[0](w[-3], 1)
        rotations[1](w[-2], 1)
        rotations[2](w[-1], 1)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, f"serial_cnot_multivariate_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[2]}"


def serial_multivariate_single_qubit(rotations, degree, scaling, random_key):
    dev = qml.device(dev_name, wires=1)
    n_features = 2
    # for returning name of model for later evaluation
    rotations_string = toString_operations(rotations)
    # calculating length of weight array
    block_size = 3
    layer_size = block_size * n_features
    total_size = layer_size * degree + block_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2 * jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        rotations[0](theta[0], wires=0)
        rotations[1](theta[1], wires=0)
        rotations[2](theta[2], wires=0)

    @qml.qnode(dev, interface="jax", diff_method=diff_method)
    def model(data, w):
        x, y = data
        W(w[0:3])
        w0 = w[3:]
        for layer in range(degree):
            w1 = w0[layer * layer_size:(layer + 1) * layer_size]
            qml.RX((scaling**layer) * x, wires=0)   # encoding
            W(w1[0:block_size])
            qml.RX((scaling**layer) * y, wires=0)   # encoding
            W(w1[block_size:n_features * block_size])

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, f"serial_multivariate_{rotations_string[0]}_{rotations_string[1]}_{rotations_string[2]}"













"""old redundant models below"""

def serial_rz_ry_rz(encoding, degree, trainable_blocks, scaling, random_key):
    dev = qml.device("default.qubit", wires=1)
    block_size = 2
    layer_size = trainable_blocks * block_size + 1
    total_size = (degree + 1) * layer_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2 * jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.RZ(theta[block][0], wires=0)
            qml.RY(theta[block][1], wires=0)

    @qml.qnode(dev, interface="jax")
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size]
            w1 = w0[:-1].reshape(trainable_blocks, block_size)
            w2 = w0[-1]
            W(w1)
            qml.RZ(w2, wires=0)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:-1].reshape(trainable_blocks, block_size))
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "serial_rz_ry_rz"


def serial_ry_rz_ry(encoding, degree, trainable_blocks, scaling, random_key):
    dev = qml.device("default.qubit", wires=1)
    block_size = 2
    layer_size = trainable_blocks * block_size + 1
    total_size = (degree + 1) * layer_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2 * jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.RY(theta[block][0], wires=0)
            qml.RZ(theta[block][1], wires=0)

    @qml.qnode(dev, interface="jax")
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size]
            w1 = w0[:-1].reshape(trainable_blocks, block_size)
            w2 = w0[-1]
            W(w1)
            qml.RY(w2, wires=0)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:-1].reshape(trainable_blocks, block_size))
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "serial_ry_rz_ry"


def serial_ry_rx_ry(encoding, degree, trainable_blocks, scaling, random_key):
    dev = qml.device("default.qubit", wires=1)
    block_size = 2
    layer_size = trainable_blocks * block_size + 1
    total_size = (degree + 1) * layer_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2 * jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.RY(theta[block][0], wires=0)
            qml.RX(theta[block][1], wires=0)

    @qml.qnode(dev, interface="jax")
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size]
            w1 = w0[:-1].reshape(trainable_blocks, block_size)
            w2 = w0[-1]
            W(w1)
            qml.RY(w2, wires=0)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:-1].reshape(trainable_blocks, block_size))
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "serial_ry_rx_ry"


def serial_rz_rx_rz(encoding, degree, trainable_blocks, scaling, random_key):
    dev = qml.device("default.qubit", wires=1)
    block_size = 2
    layer_size = trainable_blocks * block_size + 1
    total_size = (degree + 1) * layer_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2 * jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.RZ(theta[block][0], wires=0)
            qml.RX(theta[block][1], wires=0)

    @qml.qnode(dev, interface="jax")
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size]
            w1 = w0[:-1].reshape(trainable_blocks, block_size)
            w2 = w0[-1]
            W(w1)
            qml.RZ(w2, wires=0)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:-1].reshape(trainable_blocks, block_size))
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "serial_rz_rx_rz"


def serial_ry_rx_rz(encoding, degree, trainable_blocks, scaling, random_key):
    dev = qml.device("default.qubit", wires=1)
    block_size = 3
    layer_size = trainable_blocks * block_size
    total_size = (degree + 1) * layer_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2 * jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.RY(theta[block][0], wires=0)
            qml.RX(theta[block][1], wires=0)
            qml.RZ(theta[block][2], wires=0)

    @qml.qnode(dev, interface="jax")
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size].reshape(trainable_blocks, block_size)
            W(w0)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:].reshape(trainable_blocks, block_size))

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "serial_ry_rx_rz"


def serial_2qubits(encoding, degree, trainable_blocks, scaling, random_key):
    dev = qml.device("default.qubit", wires=2)
    n_rot_params = 3
    n_qubits = 2
    # W_weights = jax.random.uniform(random_key1, shape=(degree+1, trainable_blocks, n_qubits, n_rot_params), minval=0, maxval=2 * jnp.pi)
    # final_weights = jax.random.uniform(random_key2, shape=(degree+1, n_rot_params), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_weights}
    block_size = n_qubits * n_rot_params
    layer_size = trainable_blocks * block_size + n_rot_params
    total_size = (degree + 1) * layer_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.Rot(theta[block][0], theta[block][1], theta[block][2], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.Rot(theta[block][3], theta[block][4], theta[block][5], wires=1)
            qml.CNOT(wires=[1, 0])

    @qml.qnode(dev, interface="jax")
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size]
            w1 = w0[:-n_rot_params].reshape(trainable_blocks, block_size)
            w2 = w0[-n_rot_params:]
            W(w1)
            qml.Rot(w2[0], w2[1], w2[2], wires=0)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:-n_rot_params].reshape(trainable_blocks, block_size))
        qml.Rot(w[-3], w[-2], w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "serial_2qubits"


def serial_crz(encoding, degree, trainable_blocks, scaling, random_key):
    dev = qml.device("default.qubit", wires=2)
    block_size = 4
    layer_size = block_size * trainable_blocks + 1
    total_size = (degree + 1) * layer_size
    # W_weights = jax.random.uniform(random_key1, shape=((degree+1), trainable_blocks, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_weights = jax.random.uniform(random_key2, shape=(degree+1,), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_weights}
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.RY(theta[block][0], wires=0)
            qml.CRZ(theta[block][1], wires=[0, 1])
            qml.RY(theta[block][2], wires=1)
            qml.CRZ(theta[block][3], wires=[1, 0])

    @qml.qnode(dev, interface="jax")
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size]
            w1 = w0[:-1].reshape(trainable_blocks, block_size)
            w2 = w0[-1]
            W(w1)
            qml.RY(w2, wires=0)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:-1].reshape(trainable_blocks, block_size))
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "serial_crz"


def serial_cry(encoding, degree, trainable_blocks, scaling, random_key):
    dev = qml.device("default.qubit", wires=2)
    block_size = 4
    layer_size = block_size * trainable_blocks + 1
    total_size = (degree + 1) * layer_size
    # W_weights = jax.random.uniform(random_key1, shape=((degree+1), trainable_blocks, layer_size), minval=0, maxval=2 * jnp.pi)
    # final_weights = jax.random.uniform(random_key2, shape=(degree+1,), minval=0, maxval=2 * jnp.pi)
    # weights = {"W": W_weights, "final": final_weights}
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.RZ(theta[block][0], wires=0)
            qml.CRY(theta[block][1], wires=[0, 1])
            qml.RZ(theta[block][2], wires=1)
            qml.CRY(theta[block][3], wires=[1, 0])

    @qml.qnode(dev, interface="jax")
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size]
            w1 = w0[:-1].reshape(trainable_blocks, block_size)
            w2 = w0[-1]
            W(w1)
            qml.RZ(w2, wires=0)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:-1].reshape(trainable_blocks, block_size))
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "serial_cry"


def serial_crx_ry(encoding, degree, trainable_blocks, scaling, random_key):
    dev = qml.device("default.qubit", wires=2)
    block_size = 4
    layer_size = block_size * trainable_blocks + 1
    total_size = (degree + 1) * layer_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.RY(theta[block][0], wires=0)
            qml.CRX(theta[block][1], wires=[0, 1])
            qml.RY(theta[block][2], wires=1)
            qml.CRX(theta[block][3], wires=[1, 0])

    @qml.qnode(dev, interface="jax")
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size]
            w1 = w0[:-1].reshape(trainable_blocks, block_size)
            w2 = w0[-1]
            W(w1)
            qml.RY(w2, wires=0)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:-1].reshape(trainable_blocks, block_size))
        qml.RY(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "serial_crx_ry"


def serial_crx_rz(encoding, degree, trainable_blocks, scaling, random_key):
    dev = qml.device("default.qubit", wires=2)
    block_size = 4
    layer_size = block_size * trainable_blocks + 1
    total_size = (degree + 1) * layer_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2*jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.RZ(theta[block][0], wires=0)
            qml.CRX(theta[block][1], wires=[0, 1])
            qml.RZ(theta[block][2], wires=1)
            qml.CRX(theta[block][3], wires=[1, 0])

    @qml.qnode(dev, interface="jax")
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size]
            w1 = w0[:-1].reshape(trainable_blocks, block_size)
            w2 = w0[-1]
            W(w1)
            qml.RZ(w2, wires=0)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:-1].reshape(trainable_blocks, block_size))
        qml.RZ(w[-1], wires=0)

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "serial_crx_rz"


def serial_ry_crx_rz(encoding, degree, trainable_blocks, scaling, random_key):
    dev = qml.device("default.qubit", wires=2)
    block_size = 5
    layer_size = block_size * trainable_blocks
    total_size = (degree + 1) * layer_size
    weights = jax.random.uniform(random_key, shape=(total_size,), minval=0, maxval=2 * jnp.pi)

    def W(theta):
        """Trainable circuit block"""
        for block in range(trainable_blocks):
            qml.RY(theta[block][0], wires=0)
            qml.CRX(theta[block][1], wires=[0, 1])
            qml.RY(theta[block][2], wires=1)
            qml.CRX(theta[block][3], wires=[1, 0])
            qml.RZ(theta[block][4], wires=0)

    @qml.qnode(dev, interface="jax")
    def model(data, w):
        for layer in range(degree):
            w0 = w[layer * layer_size:(layer + 1) * layer_size].reshape(trainable_blocks, block_size)
            W(w0)
            encoding(data, layer, scaling)

        # (L+1)'th unitary
        W(w[-layer_size:].reshape(trainable_blocks, block_size))

        return qml.expval(qml.PauliZ(wires=0))

    return model, weights, total_size, "serial_ry_crx_rz"


