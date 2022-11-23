import gym
import numpy as np

from Environment.utils.wrappers import wrap_dqn

class TestPerformance:
    def __init__(self, env_name, agent):
        
        self.env = wrap_dqn(gym.make(env_name), 2, False, False)
        self.agent = agent
        
    
    def __call__(self, n_trials = 5):
        tot_rewards = 0
        for _ in range(n_trials):
            done = False
            state = self.env.reset()
            while not done:
                _, action = self.agent(np.expand_dims(state, 0))
                state, reward, done, _ = self.env.step(action.squeeze())
                tot_rewards += reward
        return tot_rewards/n_trials