"""
    JAX implementation of the DoG/LDoG optimizers (Ivgi et al., 2023)
"""

import chex
import jax
import jax.lax as lax
import jax.numpy as jnp
import operator
import optax
from optax import ScalarOrSchedule
from optax._src import base, combine, utils
from optax._src.alias import _scale_by_learning_rate
from typing import Optional, NamedTuple




@jax.jit
def _params_squared_norm(params: base.Params, other: Optional[base.Params]=None) -> chex.Array:
    """Compute the l2 norm ||params - other||^2 if other is given, otherwise compute ||params||^2."""
    if other is not None:
        params = jax.tree_util.tree_map(lambda x, y: x - y, params, other)
    squared_params_sums = jax.tree_map(lambda x: jnp.sum(x**2), params)
    return jax.tree_reduce(jnp.add, squared_params_sums)


@jax.jit
def _params_norm(params: base.Params, other: Optional[base.Params]=None) -> chex.Array:
    """Compute the l2 norm ||params - other|| if other is given, otherwise compute ||params||."""
    return jnp.sqrt(_params_squared_norm(params, other))



class ScaleByDogState(NamedTuple):
    """State for the DoG algorithm."""
    step_count: chex.Array
    rbar: chex.Array
    G: chex.Array
    init_buffer: base.Params


def scale_by_dog(
        reps_rel: float = 1e-6,
        eps: float = 1e-8,
        init_eta: Optional[float] = None,
        weight_decay: float = 0.0,
) -> base.GradientTransformation:
    """
        DoG (Distance over Gradients) is a parameter-free adaptive optimizer, proposed in
         `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule` (Ivgi et al., 2023).
       IMPORTANT: for best performance, DoG must be combined with iterate averaging.
    """


    accumulator_dtype = utils.canonicalize_dtype(None)

    def init_fn(params):
        return ScaleByDogState(step_count=jnp.zeros([], jnp.int32),
                               rbar=jnp.zeros([], jnp.float32),
                               G=jnp.zeros([], jnp.float32),
                               init_buffer=jax.tree_util.tree_map(
                                   lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params))

    def update_fn(updates, state, params=None):
        # assuming updates are the gradients (and that they were not scaled yet)
        if weight_decay > 0.0:
            raise NotImplementedError('weight decay is not implemented yet')
            # updates = jax.tree_multimap(lambda p, g: g + weight_decay * p, params, updates)

        def first_update(_):
            init_buffer = jax.tree_util.tree_map(jnp.copy, params)
            params_norm = _params_norm(params)
            rbar = reps_rel * (1 + params_norm)

            G = state.G + _params_squared_norm(updates)
            eta = jnp.array(init_eta if init_eta is not None else rbar / jnp.sqrt(G + eps))

            return rbar, init_buffer, G, eta

        def general_update(_):
            init_buffer = state.init_buffer
            rbar = jnp.maximum(state.rbar, _params_norm(params, tree2=init_buffer))

            G = state.G + _params_squared_norm(updates)
            eta = rbar / jnp.sqrt(G + eps)

            return rbar, init_buffer, G, eta

        rbar, init_buffer, G, eta = lax.cond(state.step_count == 0, first_update, general_update, None)

        step_count = optax.safe_int32_increment(state.step_count)
        updates = jax.tree_util.tree_map(lambda u: eta * u, updates)  # scaling by lr will come later if needed

        return updates, ScaleByDogState(step_count=step_count, rbar=rbar, G=G, init_buffer=init_buffer)

    return base.GradientTransformation(init_fn, update_fn)


def DoG(
        learning_rate: ScalarOrSchedule = 1.0,
        reps_rel: float = 1e-6,
        eps: float = 1e-8,
        init_eta: Optional[float] = None,
        weight_decay: float = 0.0,
) -> base.GradientTransformation:
    r"""Distance over Gradients - an adaptive stochastic optimizer.

            DoG updates parameters x_t with stochastic gradients g_t according to:
            .. math::
                \begin{aligned}
                    eta_t & = \frac{ max_{i \le t}{\|x_i - x_0\|} }{ \sqrt{\sum_{i \le t }{\|g_i\|^2 + eps}} }, \\
                    x_{t+1} & = x_{t} - eta_t * g_t,
                \end{aligned}

            IMPORTANT: Since we do not employ a step-size decay scheme, ITERATE AVERAGING IS CRUCIAL to obtain
            the best performance. This package provides an implementation of the polynomial decay averaging
            that is effective and does not require tuning.

            Args:
                learning_rate (ScalarOrSchedule): learning rate (referred to as c in the paper). The default value
                                                    is 1.0 and changing it is not recommended, nor is using a schedule.
                reps_rel (float): value to use to compute the  initial distance (r_epsilon in the paper).
                                            Namely, the first step size is given by:
                                            (reps_rel * (1+\|x_0\|)) / (\|g_0\|^2 + eps)^{1/2}  where x_0 are the
                                            initial weights of  the model (or the parameter group), and g_0 is the
                                            gradient of the first step.
                                            As discussed in the paper, this value should be small enough to ensure
                                            that the first update step will be small enough to not cause the model
                                            to diverge.

                                            Suggested value is 1e-6, unless the model uses batch-normalization,
                                            in which case the suggested value is 1e-4. (default: 1e-6)

                eps (float, optional): epsilon used for numerical stability - added to the sum of gradients
                                        (default: 1e-8)
                init_eta (floar, optional):  if specified, this value will be used the the initial eta (i.e.
                                            first step size), and will override the value of reps_rel (default: None)
                weight_decay (float, optional): weight decay (L2 penalty). weight_decay * x_t is added directly
                                to the gradient (default: 0)

            __ https://arxiv.org/pdf/2302.12022.pdf
            """
    return combine.chain(
        scale_by_dog(reps_rel=reps_rel, eps=eps, init_eta=init_eta, weight_decay=weight_decay),
        _scale_by_learning_rate(learning_rate),
    )



class ScaleByLDogState(NamedTuple):
    """State for the LDoG algorithm."""
    step_count: chex.Array
    rbar: base.Params
    G: base.Params
    init_buffer: base.Params


def scale_by_ldog(
        reps_rel: float = 1e-6,
        eps: float = 1e-8,
        init_eta: Optional[float] = None,
        weight_decay: float = 0.0,
) -> base.GradientTransformation:
    """
        Layer-wise DoG, as described in:
       `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule` (Ivgi et al., 2023).
        LDoG applies the DoG formula defined in the DoG class, but for each layer separately.
        IMPORTANT: for best performance, L-DoG must be combined with iterate averaging.
    """
    accumulator_dtype = utils.canonicalize_dtype(None)

    def init_fn(params):
        return ScaleByLDogState(step_count=jnp.zeros([], jnp.int32),
                               rbar=jax.tree_util.tree_map(
                                   lambda t: jnp.zeros([], jnp.float32), params),
                               G=jax.tree_util.tree_map(
                                   lambda t: jnp.zeros([], jnp.float32), params),
                               init_buffer=jax.tree_util.tree_map(
                                   lambda t: jnp.zeros_like(t, dtype=accumulator_dtype), params))

    def update_fn(updates, state, params=None):
        # assuming updates are the gradients (and that they were not scaled yet)
        if weight_decay > 0.0:
            raise NotImplementedError('weight decay is not implemented yet')
            # updates = jax.tree_multimap(lambda p, g: g + weight_decay * p, params, updates)

        def first_update(_):
            init_buffer = jax.tree_util.tree_map(jnp.copy, params)
            rbar = jax.tree_util.tree_map(lambda x: reps_rel * (1 + jnp.sqrt(jnp.sum(x ** 2))), params)

            G = jax.tree_util.tree_map(lambda G_i, g_i: G_i + jnp.sum(g_i ** 2), state.G, updates)
            if init_eta is not None:
                eta = jax.tree_util.tree_map(lambda t: jnp.array(init_eta, jnp.float32), params)
            else:
                eta = jax.tree_util.tree_map(lambda r, G_i: r / jnp.sqrt(G_i + eps), rbar, G)

            return rbar, init_buffer, G, eta

        def general_update(_):
            init_buffer = state.init_buffer

            curr_rbar = jax.tree_util.tree_map(lambda p, ib: jnp.sqrt(jnp.sum((p-ib) ** 2)), params, init_buffer)
            rbar = jax.tree_util.tree_map(jnp.maximum, state.rbar, curr_rbar)

            G = jax.tree_util.tree_map(lambda G_i, g_i: G_i + jnp.sum(g_i ** 2), state.G, updates)
            eta = jax.tree_util.tree_map(lambda r, G_i: r / jnp.sqrt(G_i + eps), rbar, G)

            return rbar, init_buffer, G, eta

        rbar, init_buffer, G, eta = lax.cond(state.step_count == 0, first_update, general_update, None)

        step_count = optax.safe_int32_increment(state.step_count)
        updates = jax.tree_util.tree_map(lambda u, etai: etai * u, updates, eta)  # scaling by lr will come later if needed

        return updates, ScaleByDogState(step_count=step_count, rbar=rbar, G=G, init_buffer=init_buffer)

    return base.GradientTransformation(init_fn, update_fn)


def LDoG(
        learning_rate: ScalarOrSchedule,
        reps_rel: float = 1e-6,
        eps: float = 1e-8,
        init_eta: Optional[float] = None,
        weight_decay: float = 0.0,
) -> base.GradientTransformation:
    """
        Layer-wise DoG, as described in:
       `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule` (Ivgi et al., 2023).
        LDoG applies the DoG formula defined in the DoG class, but for each layer separately.
        IMPORTANT: for best performance, L-DoG must be combined with iterate averaging.
    """
    return combine.chain(
        scale_by_ldog(reps_rel=reps_rel, eps=eps, init_eta=init_eta, weight_decay=weight_decay),
        _scale_by_learning_rate(learning_rate),
    )
