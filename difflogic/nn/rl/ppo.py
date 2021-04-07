#! /usr/bin/env python3
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Implement PPO loss."""

import torch.nn as nn
import torch

__all__ = ['PPOLoss']




class PPOLoss(nn.Module):
  """Implement the loss function for PPO algorithm."""

  def __init__(self, entropy_beta=None):
    super().__init__()
    self.entropy_beta = entropy_beta

  def forward(self, policy, old_policy, action, advantage, epsilon, entropy_beta=None):
#   ratio=torch.exp(nll_old - nll)
#   previous ratio was wrong because NLLLoss don't apply a log
    ratio = (policy.gather(1, action.unsqueeze(1))/old_policy.gather(1, action.unsqueeze(1)))[:,0]
    monitors = dict()
    entropy = -(policy * policy.log()).sum(dim=1).mean()
    loss1 = advantage * ratio
    loss2 = advantage * torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
    loss = -torch.min(loss1,loss2).mean()
    if entropy_beta is None:
      entropy_beta = self.entropy_beta
    if entropy_beta is not None:
      monitors['ppo_loss'] = loss
      monitors['entropy_loss'] = -entropy * entropy_beta
      loss -= entropy * entropy_beta
    monitors['entropy'] = entropy
    return loss, monitors
