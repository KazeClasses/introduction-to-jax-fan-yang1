import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dx
from jaxtyping import Float, Array, Int, PRNGKeyArray

class CNNEmulator(eqx.Module):
    layers: list

    def __init__(self, key: PRNGKeyArray, hidden_dim: Int = 4):
        self.layers = [
            eqx.nn.Conv2d(in_channels=2, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1, key=key),
            jax.nn.tanh,
            eqx.nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim * 2, kernel_size=3, stride=1, padding=1, key=key),
            jax.nn.tanh,
            eqx.nn.Conv2d(in_channels=hidden_dim * 2, out_channels=1, kernel_size=3, stride=1, padding=1, key=key),
            jax.nn.tanh
        ]


    def __call__(self, x: Float[Array, "2 n_res n_res"]) -> Float[Array, "1 n_res n_res"]:
        for layer in self.layers:
            x = layer(x)
        return x

