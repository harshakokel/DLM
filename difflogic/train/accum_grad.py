#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : accum_grad.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 01/24/2018
#
# This file is part of Jacinle.
# Distributed under terms of the MIT license.

__all__ = ['AccumGrad']

import torch
import numpy as np
from jactorch.utils.meta import as_tensor

class AccumGrad():
    def __init__(self, base_optimizer, nr_acc):
        self._base_optimizer = base_optimizer
        self._nr_acc = nr_acc
        self._current = 0
        self.batch_sizes = []

    @property
    def state(self):
        return self._base_optimizer.state

    @property
    def param_groups(self):
        return self._base_optimizer.param_groups

    def state_dict(self):
        # TODO(Jiayuan Mao @ 05/08): use a separate method to store all grad_buffer.
        return {
            'base_optimizer': self._base_optimizer.state_dict(),
            'current': self._current
        }

    def load_state_dict(self, state_dict):
        self._current = state_dict['current']
        return self._base_optimizer.load_state_dict(state_dict['base_optimizer'])

    def zero_grad(self):
        return self._base_optimizer.zero_grad()

    def provide_batch_size(self, b):
        self.batch_sizes.append(b)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self._current += 1

        for group in self._base_optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                #source gradient
                d_p = p.grad.data
                param_state = self._base_optimizer.state[p]

                # MJY:: we ensure that grad_buffer does not require grad.
                if 'grad_buffer' not in param_state:
                    buf = param_state['grad_buffer'] = []
                else:
                    buf = param_state['grad_buffer']
                    #MZ: cannot simply add cause of different batch size
                    #buf.add_(d_p)
                buf.append(d_p.clone())

                #MZ: FIX
                if 'exp_avg' not in param_state:
                    self._base_optimizer.state[p]['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    self._base_optimizer.state[p]['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    self._base_optimizer.state[p]['step'] = 0

                if self._current >= self._nr_acc:
                    assert len(self.batch_sizes) == self._current
                    #buf.mul_(1. / self._current)
                    r = (torch.stack(buf, -1) * as_tensor(np.array(self.batch_sizes)/sum(self.batch_sizes)).to(buf[0].device)).sum(-1)
                    p.grad.data.copy_(r)
                    #buf.zero_()
                    buf.clear()

        if self._current >= self._nr_acc:
            self._base_optimizer.step()
            self._current = 0
            self.batch_sizes.clear()

        return loss
