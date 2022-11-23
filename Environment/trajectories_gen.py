import gym
import numpy as np
from collections import namedtuple

from Environment.utils.accumulator import BatchAccumulator
        
class TrajectoriesGenerator:
    def __init__(self, envs, agent, n_steps: int, gamma: float):        
        self.envs = envs
        self.agent = agent
        
        self.n_envs = self.envs.n
        self.n_steps = n_steps
        self.obs_shape = self.envs.obs_shape
        self.gamma = gamma
        
        
    def __iter__(self):
        states = self.envs.reset()
        batch_accumulator = BatchAccumulator(self.n_steps, self.n_envs, self.obs_shape, self.gamma)
        
        while True:
            for idx in range(self.n_steps):
                values, actions = self.agent(states)
                new_states, rewards, dones = self.envs.step(actions)
                batch_accumulator.add(idx, states, values, actions, rewards, dones)
                states = new_states 
            
            values = self.agent(states, only_values = True)
            
            yield batch_accumulator.get_batch(values)

