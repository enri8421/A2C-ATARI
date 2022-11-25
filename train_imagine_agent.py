import torch
import wandb

from Environment.trajectories_gen import TrajectoriesGenerator
from Agent.imagine_agent import ImagineAgent
from Environment.vectorize_env import BasicVec
from Environment.test_perf import TestPerformance
from Agent.configuration import DefaultImagineAgent
from Environment.env_model import EnvModel

wandb.init(project="Breakout-a2c")

env_name = "BreakoutNoFrameskip-v4"
gamma = 0.99
n_envs = 16
n_steps = 5
idx = 400000

config = DefaultImagineAgent()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


envs = BasicVec(env_name, n_envs)
env_model = EnvModel(envs.obs_shape, envs.n_actions).to(device)
env_model.load_state_dict(torch.load(f"envmodel_weight/{idx}_EM.pt", map_location=torch.device(device)))
agent = ImagineAgent(envs.obs_shape, envs.n_actions, device, config, env_model, (1,84,84))

trajectories = TrajectoriesGenerator(envs, agent, n_steps, gamma)
tester = TestPerformance(env_name, agent)


for idx, batch in enumerate(trajectories):
    agent.update(batch)
    if (str(idx)[-3:]) == '000':
        score = tester(3)
        wandb.log({'score':score})
    if (str(idx)[-5:]) == '00000':
        agent.save(f"{idx}_")
        
        
