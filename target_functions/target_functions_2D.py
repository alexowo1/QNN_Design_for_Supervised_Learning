import jax, jax.numpy as jnp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import minmax_scaler


def drop_wave(x):
    x1 = x[..., 0][..., None]
    x2 = x[..., 1][..., None]
    r = jnp.sqrt(x1**2 + x2**2)
    return (- (1.0 + jnp.cos(12.0 * r)) / (0.5 * (x1**2 + x2**2) + 2.0)).squeeze()


A = jnp.array([3., 5., 2., 1., 7.])
B = jnp.array([5., 2., 1., 4., 9.])
C = jnp.array([1., 2., 5., 2., 3.])


def langermann(x, a=A, b=B, c=C):
    x1 = x[..., 0][..., None]
    x2 = x[..., 1][..., None]
    r2 = (x1 - a)**2 + (x2 - b)**2
    return -jnp.sum(c * jnp.exp(-r2/jnp.pi) * jnp.cos(jnp.pi*r2), axis=-1)


def make_grid(func, input_domain, axis_length, seed):
    xs_raw = jnp.linspace(input_domain[0], input_domain[1], axis_length)
    X, Y = jnp.meshgrid(xs_raw, xs_raw, indexing="xy")
    X_scaled = minmax_scaler(X) * 2 * jnp.pi
    Y_scaled = minmax_scaler(Y) * 2 * jnp.pi
    xy = jnp.stack([X.ravel(), Y.ravel()], axis=-1)
    Z = func(xy)
    Z_scaled = minmax_scaler(Z) * 2 - 1
    xy = minmax_scaler(xy) * 2 * jnp.pi

    idx_all = jnp.arange(xy.shape[0])
    idx_train, idx_test = train_test_split(idx_all, test_size=0.2, random_state=seed)

    xy_train = xy[idx_train]
    z_train = Z_scaled[idx_train]
    xy_test = xy[idx_test]
    z_test = Z_scaled[idx_test]
    return xy_train, xy_test, z_train, z_test, idx_train, idx_test, X_scaled, Y_scaled

