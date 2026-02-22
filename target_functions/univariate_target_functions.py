import jax
import jax.numpy as jnp


def fourier_function(x, c0, coeffs):
    res = c0
    for idx, c in enumerate(coeffs):
        exponent = (idx + 1) * x * 1j
        res += c * jnp.exp(exponent) + jnp.conjugate(c) * jnp.exp(-exponent)
    return jnp.real(res)


def square_wave_coeffs(frequencies):
    n = jnp.arange(1, frequencies + 1)
    coeffs = jnp.where(n % 2 == 1, -2j / (jnp.pi * n), 0.0 + 0.0j)
    c0 = 0.0
    return c0, coeffs


def sawtooth(x):
    # map x to (-π, π]
    xr = ((x + jnp.pi) % (2*jnp.pi)) - jnp.pi
    return xr / jnp.pi

def coeffs_sawtooth(frequencies):
    n = jnp.arange(1, frequencies + 1)
    coeffs = 1j * (-1.0)**n / (jnp.pi * n)
    c0 = 0.0
    return c0, coeffs


def saw_square(x):
    x = (x % (2*jnp.pi) + 2*jnp.pi) % (2*jnp.pi)
    return jnp.where(x < jnp.pi, 1.0 - 2.0*x/jnp.pi, 1.0)

def coeffs_saw_square(K):
    n = jnp.arange(1, K+1, dtype=jnp.float32)
    parity = (-1.0) ** n  # = +1 (even n), = -1 (odd n)
    coeffs = -1j * parity / (jnp.pi * n) + (1.0 - parity) / (jnp.pi**2 * n**2)
    c0 = 0.5
    return c0, coeffs
