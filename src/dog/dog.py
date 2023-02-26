"""
PyTorch implementation of the DoG/LDoG optimizers (Ivgi et al., 2023)
"""
import logging
from typing import Optional

import torch
from torch.optim import Optimizer

logger = logging.getLogger(__name__)


class DoG(Optimizer):
    """
        DoG (Distance over Gradients) is a parameter-free adaptive optimizer, proposed in
         `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule` (Ivgi et al., 2023)
    """

    __version__ = '1.0.2'

    def __init__(self, params, reps_rel: float = 1e-6, lr: float = 1.0,
                 weight_decay: float = 0.0, eps: float = 1e-8, init_eta: Optional[float] = None):
        r"""Distance over Gradients - an adaptive stochastic optimizer.

        DoG updates parameters x_t with stochastic gradients g_t according to:
        .. math::
            \begin{aligned}
                eta_t & = \frac{ max_{i \le t}{\|x_i - x_0\|} }{ \sqrt{\sum_{i \le t }{\|g_i\|^2 + eps}} }, \\
                x_{t+1} & = x_{t} - eta_t * g_t,
            \end{aligned}


        Args:
            params (iterable): iterable of parameters to optimize or dicts defining parameter groups
            reps_rel (float): value to use to compute the  initial distance (r_epsilon in the paper).
                                        Namely, the first step size is given by:
                                        (reps_rel * (1+\|x_0\|)) / (\|g_0\|^2 + eps)^{1/2}  where x_0 are the initial
                                        weights of  the model (or the parameter group), and g_0 is the gradient of the
                                        first step.
                                        As discussed in the paper, this value should be small enough to ensure that the
                                        first update step will be small enough to not cause the model to diverge.

                                        Suggested value is 1e-6, unless the model uses batch-normalization,
                                        in which case the suggested value is 1e-4. (default: 1e-6)

            lr (float, optional): learning rate (referred to as c in the paper). The default value is 1.0 and changing
                                        it is not recommended.
            weight_decay (float, optional): weight decay (L2 penalty). weight_decay * x_t is added directly
                                            to the gradient (default: 0)
            eps (float, optional): epsilon used for numerical stability - added to the sum of gradients (default: 1e-8)
            init_eta (floar, optional):  if specified, this value will be used the the initial eta (i.e.
                                        first step size), and will override the value of reps_rel (default: None)

        Example:
            >>> optimizer = DoG(model.parameters(), reps_rel=1e-6)
            >>> optimizer.zero_grad()
            >>> loss_fn(model(input), target).backward()
            >>> optimizer.step()

        __ https://arxiv.org/pdf/2302.12022.pdf
        """

        if lr <= 0.0:
            raise ValueError(f'Invalid learning rate ({lr}). Suggested value is 1.')
        if lr != 1.0:
            logger.warning(f'We do not recommend changing the lr parameter from its default value of 1')
        if init_eta is not None:
            if init_eta <= 0:
                raise ValueError(f'Invalid value for init_eta ({init_eta})')
            logger.info(f'Ignoring reps_rel since will be explicitly set init_eta to be {init_eta} (first step size)')
            reps_rel = 0
        else:
            if reps_rel <= 0.0:
                raise ValueError(f'Invalid reps_rel value ({reps_rel}). Suggested value is 1e-6 '
                                 '(unless the model uses batch-normalization, in which case suggested value is 1e-4)')

        if weight_decay < 0.0:
            raise ValueError(f'Invalid weight_decay value: {weight_decay}')

        self._first_step = True

        defaults = dict(reps_rel=reps_rel, lr=lr, weight_decay=weight_decay, eps=eps, init_eta=init_eta)
        super(DoG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DoG, self).__setstate__(state)

    def state_dict(self) -> dict:
        state_dict = super(DoG, self).state_dict()
        logger.info('retrieving DoG state dict')
        state_dict['state']['_first_step'] = self._first_step
        return state_dict

    def load_state_dict(self, state_dict: dict) -> None:
        super(DoG, self).load_state_dict(state_dict)
        self._first_step = state_dict['state']['_first_step']
        logger.info(f'loaded DoG state dict')
        cuda = self.param_groups[0]['params'][0].device
        for group in self.param_groups:
            cuda_buffers = {'init_buffer'}
            for tgroup in group.keys():
                # this can cast all the tensors to the device. However, as it turns out,
                # we need ONLY the init_buffer to be on the params' device
                if tgroup != 'params':
                    device = cuda if tgroup in cuda_buffers else 'cpu'
                    if isinstance(group[tgroup], list) and len(group[tgroup]) > 0 and \
                            isinstance(group[tgroup][0], torch.Tensor):
                        group[tgroup] = [i.to(device) for i in group[tgroup]]
                    elif isinstance(group[tgroup], torch.Tensor):
                        group[tgroup] = group[tgroup].to(device)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        first_step = self._first_step

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            if first_step:
                init = group['init_buffer'] = [torch.clone(p).detach() for p in group['params']]
            else:
                init = group['init_buffer']

            if weight_decay > 0:
                for p in group['params']:
                    p.grad.add_(p, alpha=weight_decay)

            self._update_group_state(group, init)
            self._override_init_eta_if_needed(group)

            for p, eta in zip(group['params'], group['eta']):
                if p.grad is None:
                    continue
                else:
                    p.add_(p.grad, alpha=-eta)

        self._first_step = False

        return loss

    def _update_group_state(self, group, init):
        # treat all layers as one long vector
        if self._first_step:
            group['rbar'] = group['reps_rel'] * (1 + torch.stack([p.norm() for p in group['params']]).norm())
            group['G'] = torch.stack([(p.grad.detach() ** 2).sum() for p in group['params']]).sum() + group['eps']
        else:
            curr_d = torch.stack([torch.norm(p.detach() - pi) for p, pi in zip(group['params'], init)]).norm()
            group['rbar'] = torch.maximum(group['rbar'], curr_d)
            group['G'] += torch.stack([(p.grad.detach() ** 2).sum() for p in group['params']]).sum()
        assert group['G'] > 0, \
            f'DoG cannot work when G is not strictly positive. got: {group["G"]}'
        group['eta'] = [group['lr'] * group['rbar'] / torch.sqrt(group['G'])] * len(group['params'])

    def _override_init_eta_if_needed(self, group):
        # Override init_eta if needed
        if self._first_step and group['init_eta'] is not None:
            init_eta = group['init_eta']
            logger.info(f'Explicitly setting init_eta value to {init_eta}')
            group['eta'] = [eta * 0 + init_eta for eta in group['eta']]


class LDoG(DoG):
    """
        Layer-wise DoG, as described in:
       `DoG is SGD's Best Friend: A Parameter-Free Dynamic Step Size Schedule` (Ivgi et al., 2023).
        LDoG applies the DoG formula defined in the DoG class, but for each layer separately.
    """
    def _update_group_state(self, group, init):
        # treat each layer in the group as a separate block
        if self._first_step:
            group['rbar'] = group['reps_rel'] * (1 + torch.stack([p.norm() for p in group['params']]))
            group['G'] = torch.stack([(p.grad ** 2).sum() for p in group['params']]) + group['eps']
        else:
            curr_d = torch.stack([torch.norm(p - pi) for p, pi in zip(group['params'], init)])
            group['rbar'] = torch.maximum(group['rbar'], curr_d)
            group['G'] += torch.stack([(p.grad ** 2).sum() for p in group['params']])
        assert torch.all(group['G'] > 0).item(), \
            f'DoG cannot work when g2 is not strictly positive. got: {group["G"]}'
        group['eta'] = list(group['lr'] * group['rbar'] / torch.sqrt(group['G']))


