import torch
import wandb

from Environment.trajectories_gen import TrajectoriesGenerator
from Agent.agent import Agent
from Environment.vectorize_env import BasicVec
from Environment.test_perf import TestPerformance

wandb.init(project="Breakout-a2c")

env_name = "BreakoutNoFrameskip-v4"
gamma = 0.99
n_envs = 16
n_steps = 5
beta = 0.01 
coef = 0.5 
lr =  1e-4
clip_grad = 0.5


envs = BasicVec(env_name, n_envs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = Agent(envs.obs_shape, envs.n_actions, device, beta, coef, lr, clip_grad)

trajectories = TrajectoriesGenerator(envs, agent, n_steps, gamma)
tester = TestPerformance(env_name, agent)


for idx, batch in enumerate(trajectories):
    
    agent.update(batch)
    if (str(idx)[-3:]) == '000':
        score = tester(3)
        wandb.log({'score':score})
    if (str(idx)[-4:]) == '0000':
        agent.save(f"{idx}_")
