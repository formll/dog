"""
    JAX implementation of polynomial decay averaging (Shamir & Zhang, 2013)
"""
import chex
import jax
import jax.numpy as jnp
import optax
from optax._src import base
from optax._src import utils
from typing import NamedTuple


class PolynomialAveragingState(NamedTuple):
    """State for the Polynomial decay averaging algorithm (Shamir & Zhang, 2013)."""
    count: chex.Array
    av_model: base.Params


# the implementation is based on the transform.ema
def polynomial_decay_averaging(
        gamma: float = 8,
) -> base.GradientTransformation:
    """
    Averaging model weights using a polynomial decay, as described in Shamir & Zhang, 2013.

    Given parameters x_t at iteration t, the averaged parameters are updated as follows:
    .. math::
        \begin{aligned}
            \bar{x}_t = (1 - \frac{1+\gamma}{t+\gamma}) \cdot \bar{x}_{t-1} + \frac{1+\gamma}{t+\gamma} \cdot x_t
        \end{aligned}
    """
    accumulator_dtype = utils.canonicalize_dtype(None)

    def init_fn(params):
        return PolynomialAveragingState(
            count=jnp.zeros([], jnp.int32),
            av_model=jax.tree_util.tree_map(
                lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params))

    def update_fn(updates, state, params=None):
        count_inc = optax.safe_int32_increment(state.count)

        t = state.count
        av_coef = (1 - ((gamma + 1) / (gamma + t)))
        old_coef = ((gamma + 1) / (gamma + t))
        new_av = jax.tree_util.tree_map(
            lambda av, p, u: av_coef * av + old_coef * (p + u), state.av_model, params, updates)

        av_model = utils.cast_tree(new_av, accumulator_dtype)
        return updates, PolynomialAveragingState(count=count_inc, av_model=av_model)

    return base.GradientTransformation(init_fn, update_fn)


def get_av_model(opt_state) -> base.Params:
    """
    Given an optimizer state, return the averaged model.
    """
    for sub_state in opt_state:
        if isinstance(sub_state, PolynomialAveragingState):
            return sub_state.av_model
        if isinstance(sub_state, tuple):
            av_model = get_av_model(sub_state)
            if av_model is not None:
                return av_model
    return None
