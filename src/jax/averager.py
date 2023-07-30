import copy
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import Model, state_dict, restore_state_dict

class PolynomialDecayAverager:
    """
    Averaging model weights using a polynomial decay, as described in Shamir & Zhang, 2013.

    Given parameters x_t at iteration t, the averaged parameters are updated as follows:
    .. math::
        \begin{aligned}
            \bar{x}_t = (1 - \frac{1+\gamma}{t+\gamma}) \cdot \bar{x}_{t-1} + \frac{1+\gamma}{t+\gamma} \cdot x_t
        \end{aligned}
    """

    def __init__(self, model: nn.Model, gamma: float = 8.):
        self.t = 1
        self.model = model
        self.av_model = copy.deepcopy(model)
        self.gamma = gamma

    def step(self, params):
        t = self.t
        model_sd = state_dict(self.model)
        av_sd = state_dict(self.av_model)

        for k in model_sd.keys():
            av_sd[k] = (1 - ((self.gamma + 1) / (self.gamma + t))) * av_sd[k] + ((self.gamma + 1) / (self.gamma + t)) * model_sd[k]

        self.av_model = restore_state_dict(self.av_model, av_sd)

        self.t += 1

    def reset(self):
        self.t = 1

    @property
    def averaged_model(self):
        """
        @return: returns the averaged model (the polynomial decay averaged model)
        """
        return self.av_model

    @property
    def base_model(self):
        """
        @return: returns the base model (the one that is being trainer)
        """
        return self.model

    def state_dict(self):
        """
        @return: returns the state dict of the averager.
        Note that if you wish to save the averaged model itself, as a loadable weights checkpoint,
        you should use averager.averaged_model.state_dict().
        """
        return {
            't': self.t,
            'av_model': state_dict(self.av_model)
        }

    def load_state_dict(self, state_dict):
        """
        Loads the state dict of the averager.
        @param state_dict: A state dict as returned by averager.state_dict()
        """
        self.t = state_dict['t']
        self.av_model = restore_state_dict(self.av_model, state_dict['av_model'])
