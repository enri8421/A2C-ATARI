import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb

from Environment.trajectories_gen import MultiAgentTransitionGenerator
from Agent.agent import BaseAgent
from Agent.module import EnvModel
from Environment.vectorize_env import BasicVec
from Agent.configuration import DefaultBaseAgent
from Agent.utils import unpack_transition, pred_state_loss

wandb.init(project="Breakout-a2c")

env_name = "BreakoutNoFrameskip-v4"
gamma = 0.99
n_envs = 20

lr = 5e-4


config = DefaultBaseAgent()

num_agents = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


IDX = [400000, 420000, 440000, 460000, 480000]
num_agents = len(IDX)

paths = [f"base_weight/_{idx}" for idx in IDX]
envs = [BasicVec(env_name, n_envs) for _ in range(num_agents)]
agents = [BaseAgent(envs[0].obs_shape, envs[0].n_actions, device, config, load_path=path) for path in paths]


net = EnvModel(envs[0].obs_shape, envs[0].n_actions).to(device)
optimizer = optim.Adam(net.parameters(), lr=lr)

transitions = MultiAgentTransitionGenerator(envs, agents)

for idx, transition in enumerate(transitions):
    
    states, actions, rewards, dones, delta = unpack_transition(*transition, device)
    
    optimizer.zero_grad()
    
    pred_delta, pred_rew = net(states, actions)
    
    reward_loss = F.mse_loss(pred_rew.squeeze(), rewards)
    delta_loss = pred_state_loss(pred_delta.squeeze(), delta, dones)
    
    loss = 10*delta_loss + reward_loss
    loss.backward()
    
    optimizer.step()
    
    if (str(idx)[-3:]) == '000':
        wandb.log({'state_loss':delta_loss, 'reward_loss':reward_loss})
    if (str(idx)[-5:]) == '00000':
        torch.save(net.state_dict(), f"{idx}_EM.pt")
    
    
    