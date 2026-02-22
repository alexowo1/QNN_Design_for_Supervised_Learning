import matplotlib.pyplot as plt
import optax
import jax
import jax.numpy as jnp
import pickle

from utils import *
from models.model_builder import *


def train_univariate_models(model_group, group_name, frequencies, x_train, y_train, x_test, y_test, keys, seed):
    predictions_r2scores_models = []
    gradient_logger = []
    costs_train = []
    costs_test = []
    trained_weights = []

    for models in model_group:
        # plt.plot(x_train, y_train, c="black")
        # plt.scatter(x_train, y_train, facecolor="white", edgecolor="black")

        for model in models:
            qm, weights, num_params, name = model
            fig, ax = qml.draw_mpl(qm, level="device")(x_test, weights)
            fig.savefig(f"plots/fits_and_r2/target2/controlled/fit_r2_{group_name}_{name}_degree{frequencies}_params{num_params}.png")
            plt.close()
            qm = jax.jit(qm)
            opt = optax.adam(0.01)
            opt_state = opt.init(weights)
            # num_params = param_count(weights)
            full_cost = jax.jit(lambda w: cost(w, qm, x_test, y_test))
            predict = jax.jit(jax.vmap(lambda p, w: qm(p, w), in_axes=(0, None)), static_argnums=())

            @jax.jit
            def update_step(weights, opt_state, x_batch, y_batch):
                loss_fn = lambda w: cost(w, qm, x_batch, y_batch)
                loss, grads = jax.value_and_grad(loss_fn)(weights)
                # loss, grads = jax.value_and_grad(cost)(qm, weights, x_batch, y_batch)
                updates, opt_state = opt.update(grads, opt_state)
                weights = optax.apply_updates(weights, updates)
                return weights, opt_state, loss, grads

            epochs = 1200
            batch_size = 30
            steps_per_epoch = len(x_train) // batch_size
            cst_train = []
            cst_test = [full_cost(weights)]  # initial cost
            logger = GradientLogger()

            for epoch in range(epochs):
                # shuffle batch indices for each epoch
                perm = jax.random.permutation(next(keys), len(x_train))

                for step in range(steps_per_epoch):
                    # batch_index = jax.random.choice(next(keys), len(x_train), (batch_size,), replace=False)
                    batch_index = perm[step * batch_size:(step + 1) * batch_size]

                    x_batch = x_train[batch_index]
                    y_batch = y_train[batch_index]

                    weights, opt_state, loss, grads = update_step(weights, opt_state, x_batch, y_batch)

                    # c = full_cost(weights)
                    cst_train.append(loss)

                    grads = logger.get_gradients(grads)
                    logger.update(step, loss, grads)

                # compute test loss after each epoch
                c = full_cost(weights)
                cst_test.append(c)
                if (epoch + 1) % 100 == 0:
                    print("Cost at epoch {0:3} for {1} {2} params: {3}".format(epoch + 1, name, num_params, loss), f" -  Cost of test set: {c}")

            # after a model is trained, everything gets saved in a list
            costs_train.append((cst_train, name, num_params, seed))
            costs_test.append((cst_test, name, num_params, seed))
            gradient_logger.append((logger.get_logs(), name, num_params, seed))
            trained_weights.append((weights, name, num_params))

            predictions_train = predict(x_train, weights)
            predictions_test = predict(x_test, weights)
            r2_train = r2_score(y_train, predictions_train)
            r2_test = r2_score(y_test, predictions_test)
            predictions_r2scores_models.append((predictions_train, r2_train, r2_test, name, num_params, seed))
            print("R2 train:", r2_train, "R2 test:", r2_test)

            plt.plot(x_train, y_train, linestyle="--", linewidth=1.5, label="non-diff function")
            plt.plot(x_train, predictions_train, label=f"{name}_{num_params}: R² = {r2_train:.3f}")

            plt.ylim(-1, 1)
            plt.legend()
            plt.savefig(f"plots/fits_and_r2/target2/controlled/fit_r2_{group_name}_{name}_degree{frequencies}_params{num_params}_seed{seed}.png")
            plt.close()

        with open(f"preds_and_r2/target2/controlled/predictions_r2scores_{group_name}_degree{frequencies}_seed{seed}_1", "wb") as f:
            pickle.dump(predictions_r2scores_models, f)
        with open(f"preds_and_r2/target2/controlled/gradients_{group_name}_degree{frequencies}_seed{seed}_1", "wb") as f:
            pickle.dump(gradient_logger, f)
        with open(f"preds_and_r2/target2/controlled/costs_train_{group_name}_degree{frequencies}_seed{seed}_1", "wb") as f:
            pickle.dump(costs_train, f)
        with open(f"preds_and_r2/target2/controlled/costs_test_{group_name}_degree{frequencies}_seed{seed}_1", "wb") as f:
            pickle.dump(costs_test, f)
        with open(f"trained_models/target2/controlled/trained_weights_{group_name}_degree{frequencies}_seed{seed}_1", "wb") as f:
            pickle.dump(trained_weights, f)

    return predictions_r2scores_models, gradient_logger, costs_train, costs_test, trained_weights, seed
