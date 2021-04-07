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
"""Quick access for blocksworld environments."""

import numpy

from jaclearn.rl.proxy import LimitLengthProxy

from .envs import FinalBlocksWorldEnv, StackBlocksWorldEnv
from ..utils import get_action_mapping_blocksworld
from ..utils import MapActionProxy

from .symbolicEnvironment import *

__all__ = ['get_final_env', 'get_stack_env', 'make', 'make_nlrl']


def get_final_env(nr_blocks,
                  random_order=False,
                  exclude_self=True,
                  shape_only=False,
                  fix_ground=False,
                  limit_length=None):
  """Get the blocksworld environment for the final task."""
  p = FinalBlocksWorldEnv(
      nr_blocks,
      random_order=random_order,
      shape_only=shape_only,
      fix_ground=fix_ground)
  p = LimitLengthProxy(p, limit_length or nr_blocks * 4)
  mapping = get_action_mapping_blocksworld(nr_blocks, exclude_self=exclude_self)
  p = MapActionProxy(p, mapping)
  return p

def get_stack_env(nr_blocks,
                  random_order=False,
                  exclude_self=True,
                  shape_only=False,
                  fix_ground=False,
                  limit_length=None):
  """Get the blocksworld environment for the final task."""
  p = StackBlocksWorldEnv(
      nr_blocks,
      random_order=random_order,
      shape_only=shape_only,
      fix_ground=fix_ground)
  p = LimitLengthProxy(p, limit_length or nr_blocks * 4)
  mapping = get_action_mapping_blocksworld(nr_blocks, exclude_self=exclude_self)
  p = MapActionProxy(p, mapping)
  return p

def make(task, *args, **kwargs):
  if task == 'final':
    return get_final_env(*args, **kwargs)
  elif task == 'stack':
    return get_stack_env(*args, **kwargs)
  elif 'nlrl' in task:
    return make_nlrl(task.split('-')[1], *args, **kwargs)
  else:
    raise ValueError('Unknown task: {}.'.format(task))

class NLRLEnvInterface:
    def __init__(self, nlrl_env, task):
        self.nlrl_env = nlrl_env
        self.task = task
        self.decoding = {}
        for s,i in self.nlrl_env._block_encoding.items():
            self.decoding[i] = s
        self.decoding[0] = 'floor'

    def restart(self):
        return self.nlrl_env.reset()

    @property
    def unwrapped(self):
        return self

    @property
    def world(self):
        return self

    def moveable(self, a, b):
        # judge whether an action is valid
        state = self.nlrl_env._state

        if a == 0 or a == b:
            return 0

        for block_state in state:
            if self.decoding[a] in block_state:
                ind = block_state.index(self.decoding[a])
                if ind != len(block_state) - 1:
                    return 0

            if self.decoding[b] in block_state:
                ind = block_state.index(self.decoding[b])
                if ind != len(block_state) - 1:
                    return 0
        return 1

    def action(self, a):
        return self.nlrl_env.next_step(Atom(MOVE, [self.decoding[a[0]],self.decoding[a[1]]]))

    @property
    def current_state(self):
        st = self.nlrl_env.state2atoms(self.nlrl_env.state)
        st = [str(a) for a in list(st)]
        n = self.nlrl_env._block_n + 1

        if self.task == 'On':
            goal_on = np.zeros((n,n), dtype=np.float32)
            # goal_on(a,b)
            goal_on[1,2] = 1
        
        on = np.zeros((n, n), dtype=np.float32)
        top = np.zeros((n), dtype=np.float32)
        isFloor = np.zeros(n, dtype=np.float32)
        isFloor[0] = 1
        #TODO:goalOn(a,b) for On task

        for i in range(n):
            if ('top(%s)' % self.decoding[i]) in st:
                top[i] = 1

            for j in range(n):
                if ('on(%s,%s)' % (self.decoding[i], self.decoding[j]) ) in st:
                    on[i,j] = 1

        #matrix shapes: batch size, obj..., nb predicates
        unaryPred = np.expand_dims(np.array([top, isFloor]).transpose(), 0)
        if self.task == 'On':
            binaryPred = np.expand_dims(np.array([on, goal_on]).transpose(), 0).transpose(0,2,1,3)
        else:
            binaryPred = np.expand_dims(np.array([on]).transpose(), 0).transpose(0,2,1,3)
        return [unaryPred, binaryPred]

def make_nlrl(task='Stack', nr_blocks=0, variation_index=None, exclude_self=True, *args, **kwargs):
  #ignore nr_blocks

  env_fns = {'Stack': Stack, 'Unstack': Unstack, 'On':On}
  init_states = {'Stack' : INI_STATE2, 'Unstack':INI_STATE, 'On':INI_STATE}

  init_state = init_states[task]
  env_fn = env_fns[task]

  env = env_fn(init_state)
  if variation_index:
    env = env.vary(env_fn.all_variations[variation_index])

  p = NLRLEnvInterface(env, task)
  nr_blocks = env._block_n

  p = LimitLengthProxy(p, 50)
  mapping = get_action_mapping_blocksworld(nr_blocks, exclude_self=exclude_self)
  p = MapActionProxy(p, mapping)
  return p

