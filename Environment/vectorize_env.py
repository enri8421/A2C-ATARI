import numpy as np
import gym

from Environment.utils.wrappers import wrap_dqn


class BasicVec:
    def __init__(self, env_name, n):
        self.envs = [wrap_dqn(gym.make(env_name), 2, True, True) for _ in range(n)]
        
        self.n = n
        self.obs_shape = self.envs[0].observation_space.shape
        self.n_actions = self.envs[0].action_space.n
           
    def reset(self):
        return np.array([env.reset() for env in self.envs])
    
    def autoreset(self, idx, action):
        new_state, reward, done, _ = self.envs[idx].step(action)
        if done:
            new_state = self.envs[idx].reset()
        return new_state, reward, done
    
    def step(self, actions):
        new_states, rewards, dones = (np.array(t) for t in 
                                      zip(*(self.autoreset(idx, action) for idx, action
                                            in enumerate(actions))))
        return new_states, rewards, dones
    
            
