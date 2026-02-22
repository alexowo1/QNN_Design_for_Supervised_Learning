import matplotlib.pyplot as plt
import optax
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
import pickle
from target_functions.univariate_target_functions import *
from models.controlled_rotations import *
from models.cnot_models import *
from models.ising_coupling import *
from models.encodings import *
from models.model_builder import *
from models.serial_models import *

# seed = 42

def key_generator(seed):
    key = jax.random.PRNGKey(seed)
    while True:
        key, subkey = jax.random.split(key)
        yield subkey

# keys = key_generator(seed)


def random_coeffs(frequencies, dims, keys):
    match dims:
        case 1:
            return jax.random.normal(next(keys), shape=(frequencies,)) + 1j * jax.random.normal(next(keys), shape=(frequencies,))
        case 2:
            return jax.random.normal(next(keys), shape=(frequencies, frequencies)) + 1j * jax.random.normal(next(keys), shape=(frequencies, frequencies))
        case _:
            raise ValueError(f"Possible Dimensions are 1 or 2, instead got {dims}")


# coeffs = [
#     (1.29 + 1.13j),  # c_1
#     (0.43 + 0.89j),  # c_2
#     (1.97 + 1.03j),  # c_3
#     (0.17 + 0.59j),  # c_4
#     (1.71 + 1.41j),  # c_5
#     (0.61 + 0.37j),  # c_6
#     (1.19 + 1.67j),  # c_7
#     (0.73 + 1.61j),  # c_8
#     (0.23 + 0.47j),  # c_9
#     (1.83 + 0.83j),  # c_10
# ]

coeffs = jnp.array([
    -0.37241954 + 0.23431538j,
    -0.00768933 - 0.02899783j,
    0.2424051 + 0.09402002j,
    -0.00968506 - 0.00708517j,
    0.09149331 + 0.1890211j,
    0.00709596 - 0.00369423j,
    -0.06186681 - 0.07209367j,
    -0.06571778 + 0.02410754j,
    0.00288075 + 0.0052632j,
    -0.15736917 - 0.0288954j], dtype=jnp.complex64)

# coeffs = random_coeffs(10, 1, keys)
# c0 = jax.random.normal(next(keys))
c0 = 0.0


def fourier_2D_target(x1, x2, c):
    Mx, My = c.shape
    m = jnp.arange(1, Mx + 1)  # (Mx,)
    n = jnp.arange(1, My + 1)  # (My,)

    # build phases with broadcasting: (Mx, Ny, Nx) etc., use tensordot to keep it tidy
    # shapes: phaseX -> (Mx, Nx, Ny), phaseY -> (My, Nx, Ny)
    phaseX = jnp.tensordot(m, x1, axes=0)
    phaseY = jnp.tensordot(n, x2, axes=0)

    # Combine to (Mx, My, Nx, Ny)
    phase = phaseX[:, None, :, :] + phaseY[None, :, :, :]

    # Sum_{m,n} C[m,n] * exp(i * scaling * phase)
    scaling = 1
    c0 = 0
    s = jnp.sum(c[:, :, None, None] * jnp.exp(1j * scaling * phase), axis=(0, 1))
    return c0 + 2 * jnp.real(s)


def minmax_scaler(y):
    # Scale y to [0, 1]
    y_min = jnp.min(y)
    y_max = jnp.max(y)
    y_scaled = (y - y_min) / (y_max - y_min)
    return y_scaled


def scaled_data(n_points, seed):
    x = jnp.linspace(0, 2*jnp.pi, n_points)
    # target_y = fourier_function(x, c0, coeffs) * 0.8
    target_y = saw_square(x) * 0.8
    # target_y_scaled = minmax_scaler(target_y) * 2 - 1
    x_train, x_test, y_train, y_test = train_test_split(x, target_y, test_size=0.2, random_state=seed)
    # sort by x, only for convenience/plotting
    s_train = jnp.argsort(x_train)
    s_test = jnp.argsort(x_test)
    x_train_sorted = x_train[s_train]
    y_train_sorted = y_train[s_train]
    x_test_sorted = x_test[s_test]
    y_test_sorted = y_test[s_test]
    return jnp.asarray(x_train_sorted), jnp.asarray(x_test_sorted), jnp.asarray(y_train_sorted), jnp.asarray(y_test_sorted)


def scaled_3D_data(n_points, frequencies, seed):
    x_raw = jnp.linspace(-12, 12, n_points)
    y_raw = jnp.linspace(-12, 12, n_points)
    x = minmax_scaler(x_raw) * 2 * jnp.pi
    y = minmax_scaler(y_raw) * 2 * jnp.pi
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    # C = jnp.outer(jnp.array(coeffs), jnp.array(coeffs))
    # random generated coefficients
    key = jax.random.PRNGKey(0)
    C = (jax.random.cauchy(key, (frequencies, frequencies)) + 1j * jax.random.cauchy(jax.random.split(key)[1], (frequencies, frequencies)))
    z_raw = fourier_2D_target(X, Y, C)
    z = minmax_scaler(z_raw.ravel()) * 2 - 1
    xy = jnp.stack([X.ravel(), Y.ravel()], axis=1)  # (Nx*Ny, 2)
    # split indices, sklearn returns numpy
    idx_all = jnp.arange(xy.shape[0])
    idx_train, idx_test = train_test_split(idx_all, test_size=0.2, random_state=seed)

    xy_train = xy[idx_train]
    z_train = z[idx_train]
    xy_test = xy[idx_test]
    z_test = z[idx_test]
    return xy_train, xy_test, z_train, z_test, idx_train, idx_test, X, Y


def square_loss(targets, predictions):
    return 0.5 * jnp.mean((targets - predictions) ** 2)


def cost(weights, model, x, y):
    predictions = jax.vmap(lambda data: model(data, weights))(x)
    return square_loss(y, predictions)


def cost_2D(weights, model, XY, z_flat):
    # model must accept (weights, xy) where xy is length-2
    preds = jax.vmap(lambda p: model(p, weights))(XY)
    return square_loss(z_flat, preds)


def r2_score(y_true, y_pred):
    ss_resid = jnp.sum((y_true - y_pred) ** 2)
    ss_total = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
    return 1 - ss_resid / ss_total


def initiate_models(frequencies, encoding, scaling, keys):
    if scaling == 1:
        n_qubits = frequencies
    else:
        n_qubits = int(jnp.ceil(jnp.log(frequencies) / jnp.log(scaling))) + 1

    n_qubits = 6
    scaling = 3

    axes = {
        "X": (qml.RX, qml.CRX, qml.IsingXX),
        "Y": (qml.RY, qml.CRY, qml.IsingYY),
        "Z": (qml.RZ, qml.CRZ, qml.IsingZZ)
    }
    entangling_strength = ["basic", "strongly", "all_to_all_down", "all_to_all"]
    operations = [qml.RX, qml.RY, qml.RZ]

    """serial models"""
    serial_models = []
    sequences = []
    for second_rot in operations:
        next_ops = [op for op in operations if op is not second_rot]
        for first_rot in next_ops:
            for third_rot in next_ops:
                sequences.append((first_rot, second_rot, third_rot, qml.CNOT))

    for sequence in sequences:
        serial_models += [serial_single_qubit(encoding=univariate_serial_encoding, degree=n_qubits, trainable_blocks=1, scaling=scaling, random_key=next(keys), rotations=sequence)]
    for sequence in sequences:
        serial_models += [serial_cnot_entangling(encoding=univariate_serial_encoding, degree=n_qubits, trainable_blocks=l, scaling=scaling, random_key=next(keys), rotations=sequence) for l in range(1, 5)]


    """CNOT models"""
    parallel_models = []
    sequences = []
    for second_rot in operations:
        next_ops = [op for op in operations if op is not second_rot]
        for first_rot in next_ops:
            for third_rot in next_ops:
                sequences.append((first_rot, second_rot, third_rot, qml.CNOT))

    for entangling in entangling_strength:
        if entangling == "all_to_all_down":
            measure_wire = n_qubits-1
        else:
            measure_wire = 0

        for sequence in sequences:
            parallel_models += [build_nonparameterized_entangling_model(entangling_strength=entangling, operations=sequence, n_qubits=n_qubits, trainable_layers=l,
                                                                        measure_wire=measure_wire if entangling != "strongly" else (l-1)%(n_qubits-1), measure_axis=qml.PauliZ,
                                                                        encoding=encoding, scaling=scaling, random_key=next(keys)) for l in range(1, 18)]

    """controlled models"""
    controlled_models = []
    sequences_controlled = []
    for axis, (_, ctrl_rot, _) in axes.items():
        # neighbors can't use the same axis as the middle controlled axis
        neighbors = [axes[a][0] for a in axes if a != axis]  # e.g. [RY, RZ] if m='X'
        for first_rot in neighbors:
            for third_rot in neighbors:
                if first_rot == third_rot:
                    sequences_controlled.append((first_rot, ctrl_rot))
                else:
                    sequences_controlled.append((first_rot, ctrl_rot, third_rot))

    for entangling in entangling_strength:
        for sequence in sequences_controlled:
            if entangling == "all_to_all_down":
                measure_wire = n_qubits-1
                trainable_layers_max = 7 - len(sequence) + 2
            elif entangling == "all_to_all":
                measure_wire = 0
                trainable_layers_max = 4
            else:
                measure_wire = 0
                trainable_layers_max = 17 - 5 * (len(sequence) - 2)

            controlled_models += [build_parameterized_entangling_model(entangling_strength=entangling, operations=sequence, n_qubits=n_qubits, trainable_layers=l,
                                                                       measure_wire=measure_wire if entangling != "strongly" else (l-1)%(n_qubits-1), measure_axis=qml.PauliZ,
                                                                       encoding=encoding, scaling=scaling, random_key=next(keys)) for l in range(1, trainable_layers_max)]

    """ising models"""
    ising_models = []
    sequences_ising = []
    for axis, (_, _, ising_rot) in axes.items():
        # neighbors can't use the same axis as the middle controlled axis
        neighbors = [axes[a][0] for a in axes if a != axis]  # e.g. [RY, RZ] if m='X'
        for first_rot in neighbors:
            for third_rot in neighbors:
                if first_rot == third_rot:
                    sequences_ising.append((first_rot, ising_rot))
                else:
                    sequences_ising.append((first_rot, ising_rot, third_rot))

    for entangling in entangling_strength:
        for sequence in sequences_ising:
            if entangling == "all_to_all_down":
                measure_wire = n_qubits-1
                trainable_layers_max = 7 - len(sequence) + 2
            elif entangling == "all_to_all":
                measure_wire = 0
                trainable_layers_max = 4
            else:
                measure_wire = 0
                trainable_layers_max = 17 - 5 * (len(sequence) - 2)

            ising_models += [build_parameterized_entangling_model(entangling_strength=entangling, operations=sequence, n_qubits=n_qubits, trainable_layers=l,
                                                                  measure_wire=measure_wire if entangling != "strongly" else (l-1)%(n_qubits-1), measure_axis=qml.PauliZ,
                                                                  encoding=encoding, scaling=scaling, random_key=next(keys)) for l in range(1, trainable_layers_max)]

    """poor performance models"""
    bad_models = []
    for sequence in sequences_controlled:
        bad_models += [build_parameterized_entangling_model(entangling_strength="strongly", operations=sequence, n_qubits=n_qubits, trainable_layers=l,
                                                            measure_wire=range(n_qubits), measure_axis=qml.PauliZ,
                                                            encoding=univariate_parallel_encoding, scaling=scaling, random_key=next(keys)) for l in range(1, 17)]
        bad_models += [build_parameterized_entangling_model(entangling_strength="strongly", operations=sequence, n_qubits=n_qubits, trainable_layers=l,
                                                            measure_wire=range(n_qubits), measure_axis=qml.PauliZ,
                                                            encoding=univariate_parallel_encoding, scaling=scaling, random_key=next(keys), paired_measurement=True) for l in range(1, 17)]
        bad_models += [build_parameterized_entangling_model(entangling_strength="basic", operations=(qml.RZ, qml.CRY), n_qubits=n_qubits, trainable_layers=l,
                                                            measure_wire=range(n_qubits), measure_axis=qml.PauliY,
                                                            encoding=univariate_parallel_encoding, scaling=scaling, random_key=next(keys)) for l in range(1, 17)]
        bad_models += [build_parameterized_entangling_model(entangling_strength="basic", operations=(qml.RZ, qml.CRY), n_qubits=n_qubits, trainable_layers=l,
                                                            measure_wire=range(n_qubits), measure_axis=qml.PauliY,
                                                            encoding=univariate_parallel_encoding, scaling=scaling, random_key=next(keys), paired_measurement=True) for l in range(1, 17)]
        bad_models += [build_parameterized_entangling_model(entangling_strength="basic", operations=(qml.RZ, qml.CRX), n_qubits=n_qubits, trainable_layers=l,
                                                            measure_wire=range(n_qubits), measure_axis=qml.PauliY,
                                                            encoding=univariate_parallel_encoding, scaling=scaling, random_key=next(keys)) for l in range(1, 17)]
        bad_models += [build_parameterized_entangling_model(entangling_strength="basic", operations=(qml.RZ, qml.CRY), n_qubits=n_qubits, trainable_layers=l,
                                                            measure_wire=range(n_qubits), measure_axis=qml.PauliY,
                                                            encoding=univariate_parallel_encoding, scaling=scaling, random_key=next(keys), paired_measurement=True) for l in range(1, 17)]

        controlled_models = [build_parameterized_entangling_model("basic", (qml.RZ, qml.CRX, qml.RY), 10, l,
                                                                  0, qml.PauliZ, univariate_parallel_encoding, 1, next(keys)) for l in range(3, 4)]

        # serial_models = [serial_multivariate(across_qubits_multivariate_serial_encoding, (qml.RZ, qml.RX, qml.RY), 6, l, 3, next(keys)) for l in range(1, 10)]

        return [controlled_models]


def initiate_multivariate_models(frequencies, encoding, scaling, keys):
    scaling = 3
    n_qubits = 6

    entangling_strength = ["basic", "strongly", "all_to_all_down", "all_to_all"]

    serial_models = []
    parallel_models = []
    controlled_models = []
    ising_models = []

    """serial models"""
    serial_models += [serial_multivariate_single_qubit(rotations=[qml.RY, qml.RX, qml.RZ], degree=10, scaling=1, random_key=next(keys))]
    serial_models += [serial_multivariate(across_qubits_multivariate_serial_encoding, rotations=[qml.RY, qml.RX, qml.RZ], degree=10, trainable_blocks=l,
                                          scaling=1, random_key=next(keys)) for l in range(1, 5)]

    """cnot models"""
    for entangling in entangling_strength:
        if entangling == "all_to_all_down":
            measure_wire = n_qubits-1
        else:
            measure_wire = 0

        parallel_models += [build_nonparameterized_entangling_model(entangling_strength=entangling, operations=[qml.RY, qml.RX, qml.RZ, qml.CNOT], n_qubits=n_qubits, trainable_layers=l,
                                                                    measure_wire=measure_wire if entangling != "strongly" else (l-1)%(n_qubits-1), measure_axis=qml.PauliZ,
                                                                    encoding=encoding, scaling=scaling, random_key=next(keys)) for l in range(1, 18)]

    """controlled models"""
    for entangling in entangling_strength:
        if entangling == "all_to_all_down":
            measure_wire = n_qubits-1
            trainable_layers_max = 13
        elif entangling == "all_to_all":
            measure_wire = 0
            trainable_layers_max = 9
        else:
            measure_wire = 0
            trainable_layers_max = 19

        controlled_models += [build_parameterized_entangling_model(entangling_strength=entangling, operations=[qml.RY, qml.CRX, qml.RZ], n_qubits=n_qubits, trainable_layers=l,
                                                                   measure_wire=measure_wire if entangling != "strongly" else (l-1)%(n_qubits-1), measure_axis=qml.PauliZ,
                                                                   encoding=encoding, scaling=scaling, random_key=next(keys)) for l in range(1, trainable_layers_max)]

    """ising models"""
    for entangling in entangling_strength:
        if entangling == "all_to_all_down":
            measure_wire = n_qubits-1
            trainable_layers_max = 13
        elif entangling == "all_to_all":
            measure_wire = 0
            trainable_layers_max = 9
        else:
            measure_wire = 0
            trainable_layers_max = 19

        ising_models += [build_parameterized_entangling_model(entangling_strength=entangling, operations=[qml.RY, qml.IsingXX, qml.RZ], n_qubits=n_qubits, trainable_layers=l,
                                                              measure_wire=measure_wire if entangling != "strongly" else (l-1)%(n_qubits-1), measure_axis=qml.PauliZ,
                                                              encoding=encoding, scaling=scaling, random_key=next(keys)) for l in range(1, trainable_layers_max)]

    return [serial_models]


class GradientLogger:
    def __init__(self):
        self.log = {
            "step": [],
            "loss": [],
            "grad_mean": [],
            "grad_std": [],
            "grad_min": [],
            "grad_max": [],
        }

    def get_gradients(self, grads):
        # flatten all gradient arrays into a single vector
        flat_grads = jnp.concatenate([jnp.ravel(g) for g in jax.tree_util.tree_leaves(grads)])
        return flat_grads

    def update(self, step, loss_val, flat_grads):
        self.log["step"].append(step)
        self.log["loss"].append(loss_val)
        self.log["grad_mean"].append(jnp.mean(flat_grads))
        self.log["grad_std"].append(jnp.std(flat_grads))
        self.log["grad_min"].append(jnp.min(flat_grads))
        self.log["grad_max"].append(jnp.max(flat_grads))

    def get_logs(self):
        return self.log


def param_count(pytree):
    leaves = jax.tree_util.tree_leaves(pytree)
    return int(sum(jnp.size(leaf) for leaf in leaves))

