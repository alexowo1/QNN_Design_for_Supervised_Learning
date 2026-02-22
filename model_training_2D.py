import matplotlib.pyplot as plt
import optax
import jax
import jax.numpy as jnp
import pickle

from utils import *
from models.model_builder import *


def train_multivariate_models(model_group, group_name, frequencies, xy_train, z_train, xy_test, z_test, idx_train, idx_test, x, y, num_1D_datapoints, keys, seed):
    predictions_r2scores_models = []
    gradient_logger = []
    costs_train = []
    costs_test = []
    trained_weights = []

    for models in model_group:
        for model in models:
            qm, weights, num_params, name = model
            fig, ax = qml.draw_mpl(qm, level="device")(xy_test[0], weights)
            fig.savefig(f"plots/fits_and_r2/2D_models/fourier_2D_cauchy_0/serial/circuit_{group_name}_{name}_degree{frequencies}_params{num_params}_seed{seed}.png")
            plt.close()
            qm = jax.jit(qm)
            opt = optax.adam(0.01)
            opt_state = opt.init(weights)
            full_cost = jax.jit(lambda w: cost_2D(w, qm, xy_test, z_test))
            predict = jax.jit(jax.vmap(lambda p, w: qm(p, w), in_axes=(0, None)), static_argnums=())

            @jax.jit
            def update_step(weights, opt_state, xy_batch, z_batch):
                loss_fn = lambda w: cost_2D(w, qm, xy_batch, z_batch)
                loss, grads = jax.value_and_grad(loss_fn)(weights)
                updates, opt_state = opt.update(grads, opt_state)
                weights = optax.apply_updates(weights, updates)
                return weights, opt_state, loss, grads

            epochs = 1000
            batch_size = 2000
            steps_per_epoch = len(xy_train) // batch_size
            cst_train = []
            cst_test = [full_cost(weights)]
            logger = GradientLogger()

            for epoch in range(epochs):
                # shuffle batch indices for each epoch
                perm = jax.random.permutation(next(keys), len(xy_train))
                loss_avg = []

                for step in range(steps_per_epoch):
                    # batch_index = jax.random.choice(next(keys), num_target_points, (batch_size,), replace=False)
                    batch_index = perm[step * batch_size:(step + 1) * batch_size]

                    xy_batch = xy_train[batch_index]
                    z_batch = z_train[batch_index]

                    weights, opt_state, loss, grads = update_step(weights, opt_state, xy_batch, z_batch)

                    # cst_train.append(loss)
                    # flat_grads = logger.get_gradients(grads)
                    # logger.update(step, loss, flat_grads)
                    loss_avg.append(loss)

                # compute average train and overall test loss after each epoch
                mean_train_loss = jnp.mean(jnp.array(loss_avg))
                cst_train.append(mean_train_loss)
                c = full_cost(weights)
                cst_test.append(c)
                if (epoch + 1) % 100 == 0:
                    print("Cost at epoch {0:3} for {1} {2} params: {3}".format(epoch + 1, name, num_params, mean_train_loss), f" -  Cost of test set: {c}")

            # after a model is trained, everything gets saved in a list
            costs_train.append((cst_train, name, num_params, seed))
            costs_test.append((cst_test, name, num_params, seed))
            gradient_logger.append((logger.get_logs(), name, num_params, seed))
            trained_weights.append((weights, name, num_params))

            predictions_train = predict(xy_train, weights)
            predictions_test = predict(xy_test, weights)
            r2_train = r2_score(z_train, predictions_train)
            r2_test = r2_score(z_test, predictions_test)
            predictions_r2scores_models.append((predictions_train, r2_train, r2_test, name, num_params, seed))
            print("R2 train:", r2_train, "R2 test:", r2_test)

            full_pred = jnp.empty((num_1D_datapoints**2,))
            full_pred = full_pred.at[idx_train].set(predictions_train)
            full_pred = full_pred.at[idx_test].set(predictions_test)

            z = full_pred.reshape(num_1D_datapoints, num_1D_datapoints)

            # plt.plot(x, predictions, label=f"{name}_{num_params}: R² = {r2:.4f}")

            # plot
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(121, projection="3d")
            ax.plot_surface(x, y, z, rstride=1, cstride=1, linewidth=0, antialiased=True, cmap="viridis", label=f"{name}_{num_params}: R² = {r2_train:.4f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.legend()
            # plt.tight_layout()

            ax = fig.add_subplot(122, projection='3d')
            # ax.plot_surface(jnp.asarray(X), jnp.asarray(Y), jnp.asarray(Z), edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)

            # plot projections of the contours for each dimension
            ax.contour(x, y, z, zdir='z', offset=jnp.min(z), cmap='coolwarm')
            ax.contour(x, y, z, zdir='x', offset=-0.2, cmap='coolwarm')
            ax.contour(x, y, z, zdir='y', offset=6.2, cmap='coolwarm')
            fig.savefig(f"plots/fits_and_r2/2D_models/fourier_2D_cauchy_0/serial/fit_r2_{group_name}_{name}_degree{frequencies}_params{num_params}_seed{seed}.png")
            plt.close()

            with open(f"preds_and_r2/2D_models/fourier_2D_cauchy_0/serial/predictions_r2scores_{group_name}_degree{frequencies}_seed{seed}", "wb") as f:
                pickle.dump(predictions_r2scores_models, f)
            with open(f"preds_and_r2/2D_models/fourier_2D_cauchy_0/serial/gradients_{group_name}_degree{frequencies}_seed{seed}", "wb") as f:
                pickle.dump(gradient_logger, f)
            with open(f"preds_and_r2/2D_models/fourier_2D_cauchy_0/serial/costs_{group_name}_degree{frequencies}_seed{seed}", "wb") as f:
                pickle.dump(costs_train, f)
            with open(f"preds_and_r2/2D_models/fourier_2D_cauchy_0/serial/costs_{group_name}_degree{frequencies}_seed{seed}", "wb") as f:
                pickle.dump(costs_test, f)
            with open(f"trained_models/2D_models/fourier_2D_cauchy_0/serial/trained_weights_{group_name}_degree{frequencies}_seed{seed}", "wb") as f:
                pickle.dump(trained_weights, f)

    return predictions_r2scores_models, gradient_logger, costs_train, costs_test, trained_weights, seed
