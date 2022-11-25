import numpy as np
from collections import namedtuple

Batch = namedtuple("Batch", "states ref_values advs probs actions")


class BatchAccumulator:
    def __init__(self, n, d, obs_shape, n_actions, gamma):
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.n = n
        self.d = d
        self.mb_states = np.zeros((n, d) + obs_shape, dtype=np.uint8)
        self.mb_values = np.zeros((n, d), dtype=np.float32)
        self.mb_probs = np.zeros((n, d, n_actions), dtype=np.float32)
        self.mb_actions = np.zeros((n, d), dtype=np.int32)
        self.mb_rewards = np.zeros((n, d), dtype=np.float32)
        self.mb_dones = np.full((n, d), True, dtype=np.bool_)
        
    def add(self, idx, states, values, probs, actions, rewards, dones):
        self.mb_states[idx] = states
        self.mb_values[idx] = values
        self.mb_probs[idx] = probs
        self.mb_actions[idx] = actions
        self.mb_rewards[idx] = rewards
        self.mb_dones[idx] = dones
        
    def compute_ref_values(self, values):
        discounted = np.zeros((self.n, self.d), dtype=np.float32)
        discounted[self.n - 1] = self.mb_rewards[self.n - 1] + \
                    self.gamma*np.where(self.mb_dones[self.n-1], np.zeros(self.d), values)
        for i in range(1, self.n):
            discounted[self.n - 1 - i] = self.mb_rewards[self.n - 1 - i] + \
                        self.gamma*np.where(self.mb_dones[self.n - 1 - i], np.zeros(self.d), discounted[self.n - i])
        return discounted
    
    
    def get_batch(self, values):
        ref_values = self.compute_ref_values(values)
        advs = ref_values - self.mb_values
        
        states = np.reshape(self.mb_states, (-1,) + self.obs_shape)
        ref_values = ref_values.flatten()
        advs = advs.flatten()
        probs = np.reshape(self.mb_probs, (-1, self.n_actions))
        actions = self.mb_actions.flatten()
        
        return Batch(states, ref_values, advs, probs, actions)