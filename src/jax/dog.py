from typing import Optional, NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.lax as lax
from optax import ScalarOrSchedule
from optax._src import base, combine, transform
from optax._src.alias import _scale_by_learning_rate


class ScaleByDogState(NamedTuple):
    """State for the Adam algorithm."""
    step_count: chex.Array  # shape=(), dtype=jnp.int32.  # TODO - seems like this is not the way to define scalar?
    rbar: chex.Array
    g: chex.Array
    init_buffer: chex.Array


def scale_by_dog(
        reps_rel: float = 1e-6,
        eps: float = 1e-8,
        init_eta: Optional[float] = None,
        weight_decay: float = 0.0,
) -> base.GradientTransformation:

    def init_fn(params):
        return ScaleByDogState(step_count=jnp.zeros([], jnp.int32),
                               rbar=jnp.zeros([], jnp.float32),
                               g=jnp.zeros([], jnp.float32),
                               init_buffer=jnp.hstack([v.flatten() for p in params.values() for v in p.values()]))
        # mu = jax.tree_util.tree_map(  # First moment
        #     lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
        # nu = jax.tree_util.tree_map(jnp.zeros_like, params)  # Second moment
        # return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        # updates are the gradients if they were not scaled yet
        if weight_decay > 0.0:
            raise NotImplementedError('weight decay is not implemented yet')
            # updates = jax.tree_multimap(lambda p, g: g + weight_decay * p, params, updates)

        def first_update(_):
            init_buffer = jnp.hstack([v.flatten() for p in params.values() for v in p.values()])
            params_norm = jnp.linalg.norm(
                jnp.hstack([v.flatten() for p in params.values() for v in p.values()]))  # biases and kernels
            rbar = reps_rel * (1 + params_norm)

            grads_flat = jnp.hstack([v.flatten() for g in updates.values() for v in g.values()])
            g = state.g + jnp.sum(grads_flat ** 2)

            eta = jnp.array(init_eta if init_eta is not None else rbar / jnp.sqrt(g + eps))

            return rbar, init_buffer, g, eta

        def general_update(_):
            init_buffer = state.init_buffer
            params_flat = jnp.hstack(jnp.hstack([v.flatten() for p in params.values() for v in p.values()]))
            rbar = jnp.maximum(state.rbar, jnp.linalg.norm(params_flat - init_buffer))

            grads_flat = jnp.hstack([v.flatten() for g in updates.values() for v in g.values()])
            g = state.g + jnp.sum(grads_flat ** 2)

            eta = rbar / jnp.sqrt(g + eps)

            return rbar, init_buffer, g, eta

        rbar, init_buffer, g, eta = lax.cond(state.step_count == 0, first_update, general_update, None)

        step_count = state.step_count + 1
        updates = jax.tree_util.tree_map(lambda u: eta*u, updates)  # scaling by lr will come later if needed

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


def init_ldog_params(params, reps_rel=1e-6, lr=1.0, eps=1e-8, init_eta=None):
    rbar = jax.tree_map(lambda p: reps_rel * (1 + jnp.linalg.norm(p)), params)
    g = jax.tree_map(lambda p: jnp.array([0.]), params)
    eta = jax.tree_map(lambda p: jnp.array([init_eta if init_eta is not None else lr * rbar[p] / jnp.sqrt(g[p] + eps)]),
                       params)

    return dict(rbar=rbar, g=g, eta=eta)


def update_ldog_params(params, grads, state, weight_decay=0.0, eps=1e-8):
    g = jax.tree_multimap(lambda gi, p: gi + jnp.sum(p ** 2), state['g'], grads)
    rbar = jax.tree_multimap(lambda rb, p1, p2: jnp.maximum(rb, jnp.linalg.norm(p1 - p2)), state['rbar'], params, grads)
    eta = jax.tree_multimap(lambda r, gi, ei: ei[0] * r / jnp.sqrt(gi[0] + eps), rbar, g, state['eta'])

    if weight_decay > 0.0:
        grads = jax.tree_multimap(lambda p, g: g + weight_decay * p, params, grads)

    params = jax.tree_multimap(lambda p, g, e: p - e[0] * g, params, grads, eta)

    return params, dict(rbar=rbar, g=g, eta=eta)
