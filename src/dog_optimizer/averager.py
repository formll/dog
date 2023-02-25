"""
PyTorch implementation of polynomial decay averaging (Shamir & Zhang, 2013)
"""
import logging

import torch
from torch import nn

from copy import deepcopy

logger = logging.getLogger(__name__)


class PolynomialDecayAverager:
    """
    Averaging model weights using a polynomial decay, as described in Shamir & Zhang, 2013.

    Given parameters x_t at iteration t, the averaged parameters are updated as follows:
    .. math::
        \begin{aligned}
            \bar{x}_t = (1 - \frac{1+\gamma}{t+\gamma}) \cdot \bar{x}_{t-1} + \frac{1+\gamma}{t+\gamma} \cdot x_t
        \end{aligned}
    """

    def __init__(self, model: nn.Module, gamma: float = 8.):
        self.t = 1
        self.model = model
        self.av_model = deepcopy(model)
        self.gamma = gamma
        self._matched_devices = False

    def step(self):
        self._match_models_devices()

        t = self.t
        model_sd = self.model.state_dict()
        av_sd = self.av_model.state_dict()

        for k in model_sd.keys():
            if isinstance(av_sd[k], (torch.LongTensor, torch.cuda.LongTensor)):
                # these are buffers that store how many batches batch norm has seen so far
                av_sd[k].copy_(model_sd[k])
                continue
            av_sd[k].mul_(1 - ((self.gamma + 1) / (self.gamma + t))).add_(
                model_sd[k], alpha=(self.gamma + 1) / (self.gamma + t)
            )

        self.t += 1

    def _match_models_devices(self):
        if self._matched_devices:
            return

        # nn.Module does not always have a device attribute, so we check if the model has one and use it to match
        # the device where the av_model is stored
        if hasattr(self.model, 'device'):
            if self.model.device != self.av_model.device:
                self.av_model = self.av_model.to(self.model.device)
        else:
            # This could be a problem if the model is split across multiple devices in a distributed manner
            model_device, av_device = next(self.model.parameters()).device, next(self.av_model.parameters()).device
            if model_device != av_device:
                self.av_model = self.av_model.to(model_device)

        self._matched_devices = True

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
            'av_model': self.av_model.state_dict()
        }

    def load_state_dict(self, state_dict):
        """
        Loads the state dict of the averager.
        @param state_dict: A state dict as returned by averager.state_dict()
        """
        self.t = state_dict['t']
        self.av_model.load_state_dict(state_dict['av_model'])
