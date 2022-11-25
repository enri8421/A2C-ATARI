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
        self.n_actions = self.envs.n_actions
        self.gamma = gamma
        
        
    def __iter__(self):
        states = self.envs.reset()
        batch_accumulator = BatchAccumulator(self.n_steps, self.n_envs, self.obs_shape, self.n_actions, self.gamma)
        
        while True:
            for idx in range(self.n_steps):
                values, actions, probs = self.agent(states)
                new_states, rewards, dones = self.envs.step(actions)
                batch_accumulator.add(idx, states, values, probs, actions, rewards, dones)
                states = new_states 
            
            values = self.agent(states, only_values = True)
            
            yield batch_accumulator.get_batch(values)
            
            


class MultiAgentTransitionGenerator:
    def __init__(self, envs, agents):
        assert len(envs) == len(agents)
        
        self.envs = envs
        self.agents = agents        
        
    def __iter__(self):
        states = [env.reset() for env in self.envs]
        while True:
            actions = [agent(state)[1] for agent, state in zip(self.agents, states)]
            new_states, rewards, dones = (list(t) for t in zip(*(
                                (env.step(action) for env, action in zip(self.envs, actions)))))
            yield np.vstack(states), np.hstack(actions), np.hstack(rewards), \
                 np.hstack(dones), np.vstack(new_states)
            states = new_states

