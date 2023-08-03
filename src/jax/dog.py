import chex
import jax
import jax.lax as lax
import jax.numpy as jnp
import operator
import optax
from optax import ScalarOrSchedule
from optax._src import base, combine
from optax._src import utils
from optax._src.alias import _scale_by_learning_rate
from typing import Optional, NamedTuple


# modified from https://github.com/deepmind/optax/blob/master/optax/_src/contrib/mechanic.py ###########
@jax.jit
def _tree_sum(tree_x):
    """Compute sum(tree_x)."""
    sums = jax.tree_util.tree_map(jnp.sum, tree_x)
    return jax.tree_util.tree_reduce(operator.add, sums)


@jax.jit
def _tree_squared_norm(tree, tree2=None):
    """Compute the l2 norm ||tree1 - tree2|| if tree2 is given,
  otherwise compute ||tree1||."""
    if tree2 is not None:
        tree = jax.tree_util.tree_map(lambda x, y: x - y, tree, tree2)
    return _tree_sum(jax.tree_map(lambda x: jnp.sum(x ** 2), tree))


@jax.jit
def _tree_norm(tree, tree2=None):
    return jnp.sqrt(_tree_squared_norm(tree, tree2))


#################################


class ScaleByDogState(NamedTuple):
    """State for the DoG algorithm."""
    step_count: chex.Array  # shape=(), dtype=jnp.int32.
    rbar: chex.Array
    g: chex.Array
    init_buffer: base.Params


def scale_by_dog(
        reps_rel: float = 1e-6,
        eps: float = 1e-8,
        init_eta: Optional[float] = None,
        weight_decay: float = 0.0,
) -> base.GradientTransformation:
    accumulator_dtype = utils.canonicalize_dtype(None)

    def init_fn(params):
        return ScaleByDogState(step_count=jnp.zeros([], jnp.int32),
                               rbar=jnp.zeros([], jnp.float32),
                               g=jnp.zeros([], jnp.float32),
                               init_buffer=jax.tree_util.tree_map(
                                   lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params))

    def update_fn(updates, state, params=None):
        # updates are the gradients if they were not scaled yet
        if weight_decay > 0.0:
            raise NotImplementedError('weight decay is not implemented yet')
            # updates = jax.tree_multimap(lambda p, g: g + weight_decay * p, params, updates)

        def first_update(_):
            init_buffer = jax.tree_util.tree_map(jnp.copy, params)
            params_norm = _tree_norm(params)
            rbar = reps_rel * (1 + params_norm)

            g = state.g + _tree_squared_norm(updates)
            eta = jnp.array(init_eta if init_eta is not None else rbar / jnp.sqrt(g + eps))

            return rbar, init_buffer, g, eta

        def general_update(_):
            init_buffer = state.init_buffer
            rbar = jnp.maximum(state.rbar, _tree_norm(params, tree2=init_buffer))

            g = state.g + _tree_squared_norm(updates)
            eta = rbar / jnp.sqrt(g + eps)

            return rbar, init_buffer, g, eta

        rbar, init_buffer, g, eta = lax.cond(state.step_count == 0, first_update, general_update, None)

        step_count = optax.safe_int32_increment(state.step_count)  # should be used in dog as well
        updates = jax.tree_util.tree_map(lambda u: eta * u, updates)  # scaling by lr will come later if needed

        return updates, ScaleByDogState(step_count=step_count, rbar=rbar, g=g, init_buffer=init_buffer)

    return base.GradientTransformation(init_fn, update_fn)


def DoG(
        learning_rate: ScalarOrSchedule,
        reps_rel: float = 1e-6,
        eps: float = 1e-8,
        init_eta: Optional[float] = None,
        weight_decay: float = 0.0,
) -> base.GradientTransformation:
    return combine.chain(
        scale_by_dog(reps_rel=reps_rel, eps=eps, init_eta=init_eta, weight_decay=weight_decay),
        _scale_by_learning_rate(learning_rate),
    )


############################################


class ScaleByLDogState(NamedTuple):
    """State for the DoG algorithm."""
    step_count: chex.Array  # shape=(), dtype=jnp.int32.
    rbar: base.Params
    g: base.Params
    init_buffer: base.Params


def scale_by_ldog(
        reps_rel: float = 1e-6,
        eps: float = 1e-8,
        init_eta: Optional[float] = None,
        weight_decay: float = 0.0,
) -> base.GradientTransformation:
    accumulator_dtype = utils.canonicalize_dtype(None)

    def init_fn(params):
        return ScaleByLDogState(step_count=jnp.zeros([], jnp.int32),
                               rbar=jax.tree_util.tree_map(
                                   lambda t: jnp.zeros([], jnp.float32), params),
                               g=jax.tree_util.tree_map(
                                   lambda t: jnp.zeros([], jnp.float32), params),
                               init_buffer=jax.tree_util.tree_map(
                                   lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params))

    def update_fn(updates, state, params=None):
        # updates are the gradients if they were not scaled yet
        if weight_decay > 0.0:
            raise NotImplementedError('weight decay is not implemented yet')
            # updates = jax.tree_multimap(lambda p, g: g + weight_decay * p, params, updates)

        def first_update(_):
            init_buffer = jax.tree_util.tree_map(jnp.copy, params)
            rbar = jax.tree_util.tree_map(lambda x: reps_rel * (1+ jnp.sqrt(jnp.sum(x ** 2))), params)

            g = jax.tree_util.tree_map(lambda old_g, params_g: old_g + jnp.sum(params_g ** 2), state.g, params)
            if init_eta is not None:
                eta = jax.tree_util.tree_map(lambda t: jnp.array(init_eta, jnp.float32), params)
            else:
                eta = jax.tree_util.tree_map(lambda r, gi: r / jnp.sqrt(gi + eps), rbar, g)

            return rbar, init_buffer, g, eta

        def general_update(_):
            init_buffer = state.init_buffer

            curr_rbar = jax.tree_util.tree_map(lambda p, ib: jnp.sqrt(jnp.sum((p-ib) ** 2)), params, init_buffer)
            rbar = jax.tree_util.tree_map(jnp.maximum, state.rbar, curr_rbar)

            g = jax.tree_util.tree_map(lambda old_g, params_g: old_g + jnp.sum(params_g ** 2), state.g, params)
            eta = jax.tree_util.tree_map(lambda r, gi: r / jnp.sqrt(gi + eps), rbar, g)

            return rbar, init_buffer, g, eta

        rbar, init_buffer, g, eta = lax.cond(state.step_count == 0, first_update, general_update, None)

        step_count = optax.safe_int32_increment(state.step_count)  # should be used in dog as well
        updates = jax.tree_util.tree_map(lambda u, etai: etai * u, updates, eta)  # scaling by lr will come later if needed

        return updates, ScaleByDogState(step_count=step_count, rbar=rbar, g=g, init_buffer=init_buffer)

    return base.GradientTransformation(init_fn, update_fn)


def LDoG(
        learning_rate: ScalarOrSchedule,
        reps_rel: float = 1e-6,
        eps: float = 1e-8,
        init_eta: Optional[float] = None,
        weight_decay: float = 0.0,
) -> base.GradientTransformation:
    return combine.chain(
        scale_by_ldog(reps_rel=reps_rel, eps=eps, init_eta=init_eta, weight_decay=weight_decay),
        _scale_by_learning_rate(learning_rate),
    )