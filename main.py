import argparse
import pickle
import warnings
from model_training import *
from model_training_2D import *
from utils import *
from target_functions.univariate_target_functions import *
from target_functions.target_functions_2D import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_input_features", type=int, choices=[1, 2], default=2)
    parser.add_argument("--group", choices=["standard", "exponential"], default="exponential")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    num_axial_datapoints = 150
    frequencies = 10
    # c0, coeffs = coeffs_saw_square(frequencies)

    warnings.filterwarnings(
        "ignore",
        message=r"Explicitly requested dtype .*complex128.* will be truncated to dtype complex64"
    )

    def log_jax_backend():
        print("[JAX] devices:", jax.devices())
        print("[JAX] default backend:", jax.default_backend())
        x = jnp.ones(())
        # handle both APIs (.device attr vs .device() method)
        dev = getattr(x, "device", None)
        if callable(dev):
            dev = dev()
        print("[JAX] sample array device:", dev)

    log_jax_backend()

    keys = key_generator(args.seed)
    if args.num_input_features == 1:
        x_train, x_test, y_train, y_test = scaled_data(num_axial_datapoints, args.seed)
        with open(f"trained_models/target2/controlled/train_datapoints_seed{args.seed}", "wb") as f:
            pickle.dump([x_train, y_train], f)
        with open(f"trained_models/target2/controlled/test_datapoints_seed{args.seed}", "wb") as f:
            pickle.dump([x_test, y_test], f)
        if args.group == "standard":
            models = initiate_models(frequencies=frequencies, encoding=univariate_parallel_encoding, scaling=1, keys=keys)
        else:
            models = initiate_models(frequencies=frequencies, encoding=univariate_parallel_encoding, scaling=3, keys=keys)

        predictions_r2scores_models, gradient_logger, costs_train, costs_test, trained_weights, seed = train_univariate_models(models, args.group, frequencies, x_train, y_train, x_test, y_test, keys, args.seed)

    else:
        xy_train, xy_test, z_train, z_test, idx_train, idx_test, x, y = scaled_3D_data(num_axial_datapoints, frequencies, args.seed)
        # xy_train, xy_test, z_train, z_test, idx_train, idx_test, x, y = make_grid(langermann, (0, 10), num_axial_datapoints, args.seed)
        with open(f"trained_models/2D_models/fourier_2D_cauchy_0/serial/data_seed{args.seed}", "wb") as f:
            pickle.dump((xy_train, xy_test, z_train, z_test, idx_train, idx_test, x, y), f)
        if args.group == "standard":
            models = initiate_multivariate_models(frequencies=frequencies, encoding=across_qubits_multivariate_parallel_encoding, scaling=1, keys=keys)
        else:
            # xy_train, xy_test, z_train, z_test, idx_train, idx_test, x, y = make_grid(langermann, (0, 10), num_axial_datapoints, args.seed)
            models = initiate_multivariate_models(frequencies=frequencies, encoding=across_qubits_multivariate_parallel_encoding, scaling=3, keys=keys)
        predictions_r2scores_models, gradient_logger, costs_train, costs_test, trained_weights, seed = train_multivariate_models(models, args.group, frequencies, xy_train, z_train, xy_test, z_test, idx_train, idx_test, x, y, num_axial_datapoints, keys, args.seed)



if __name__ == "__main__":
    main()
