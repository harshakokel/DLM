import torch
import numpy as np
from collections import deque

def update_dict_list(dict_full, dict_current):
    for k in dict_current.keys():
        if k in dict_full.keys():
            dict_full[k].extend(dict_current[k])
        else:
            dict_full[k] = deque()
            dict_full[k].extend(dict_current[k])

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta.square() * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-7, shape=()):
        self.mean = torch.zeros(shape, dtype=torch.float)
        self.var = torch.ones(shape, dtype=torch.float)
        self.count = epsilon
        self.epsilon = 1e-7

    def update(self, x):
        batch_mean = x.mean()
        batch_var = x.var(unbiased=(x.shape[0]!=1))
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def obs_filter(self, obs):
        self.update(obs)
        obs = (obs - self.mean) / np.sqrt(self.var + self.epsilon)
        return obs


